# Copyright (c) 2024, Tri Dao, Albert Gu.
# Extended to 2d using VMamba method by Nicolas Carpenter

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from functools import partial

from einops import rearrange, repeat

from timm.models.layers import DropPath

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,        # NOTE: vmamba elects to use 96
        d_state=64,
        d_conv=3,
        conv_init=None,
        expand=2,
        headdim=96,     # NOTE: might be mismatched with the model dimension
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        # alternate architecture
        disable_z = False,
        oact = False,
        dropout=0.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim        
        self.D_has_hdim = D_has_hdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.act = activation    # NOTE: VMamba uses nn.GELU(), Mamba2 uses nn.SILU()
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        
        # VMamba inits 
        self.k_groups = 4
        self.disable_z = disable_z
        self.oact = oact

        if self.disable_z:
            # Order: [z, x, B, C, dt]
            d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        else:
            # Order: [x, B, C, dt]
            d_in_proj = self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
  
        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.k_groups, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

       
        
        # applied to [x, B, C]
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state

        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=(d_conv - 1) // 2,  # NOTE: might need to adjust this
            **factory_kwargs,
        )
    
        if self.conv_init is not None:
            nn.init.uniform_(self.conv2d.weight, -self.conv_init, self.conv_init)
        # self.conv2d.weight._no_weight_decay = True

        # Initialize log dt bias
        dts = torch.exp(
            torch.rand(self.k_groups, self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dts = torch.clamp(dts, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dts = dts + torch.log(-torch.expm1(-dts))
        self.dts_bias = nn.Parameter(inv_dts)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dts_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        As = torch.empty(self.k_groups, self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_logs = torch.log(As).to(dtype=dtype)
        self.A_logs = nn.Parameter(A_logs)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_logs._no_weight_decay = True

        # D "skip" parameter
        self.Ds = nn.Parameter(
            torch.ones(
                self.k_groups, 
                self.d_inner if self.D_has_hdim else self.nheads,
                device=device
            )
        )
        self.Ds._no_weight_decay = True

        # NOTE: need to change the normalization (this is built for a 1d input)
        # Extra normalization layer right before output projection
        # assert RMSNormGated is not None
        # self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        # NOTE: we will just use the VMamba setup for now
        self.out_norm = nn.LayerNorm(self.d_inner, **factory_kwargs)
        self.out_act = nn.GELU() if self.oact else nn.Identity()

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(
        self, 
        u, 
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        selective_scan_backend = None,
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        seq_idx=None
    ):
        """
        u: (B, H, W, C)
        Returns: (B, H, W, C)
        """        
        batch, H, W, _ = u.shape

        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]        
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        # Input Projection
        if not self.disable_z:
            zxbcdt = self.in_proj(u)  # (B, H, W, d_in_proj)
            
            z, xBC, dt = torch.split(
                    zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], 
                    dim=-1
                )
        
        else:
            xbcdt = self.in_proj(u)  # (B, H, W, d_in_proj)
            
            xBC, dt = torch.split(
                    xbcdt, [self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], 
                    dim=-1
                )

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        As = -torch.exp(self.A_logs.float())  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # TODO: will need to look to see if a fused conv, scan kernal is feasible
        if self.use_mem_eff_path:
            # Fully fused path
            pass
            # out = mamba_split_conv1d_scan_combined(
            #     zxbcdt,
            #     rearrange(self.conv2d.weight, "d 1 w -> d w"),
            #     self.conv2d.bias,
            #     self.dt_bias,
            #     A,
            #     D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            #     chunk_size=self.chunk_size,
            #     seq_idx=None,
            #     activation=self.activation,
            #     rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            #     rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            #     outproj_weight=self.out_proj.weight,
            #     outproj_bias=self.out_proj.bias,
            #     headdim=None if self.D_has_hdim else self.headdim,
            #     ngroups=self.ngroups,
            #     norm_before_gate=self.norm_before_gate,
            #     **dt_limit_kwargs,
            # )

        else:
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            
            # 2D Convolution
            # (B, H, W, self.d_inner + 2 * ngroups * d_state)
            xBC = rearrange(xBC, "b h w c -> b c h w")
            xBC = self.conv2d(xBC)  #NOTE: electing not to use truncation, should be covered by padding [:, :-(self.d_conv - 1)]
            xBC = rearrange(xBC, "b c h w -> b h w c")
            xBC = self.act(xBC)
            
            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

            # Expand B, C, dt to K directions
            Bs = B.unsqueeze(3).expand(-1, -1, -1, self.k_groups, -1)  # (B, H, W, K, ngroups*d_state)
            Cs = C.unsqueeze(3).expand(-1, -1, -1, self.k_groups, -1)  # (B, H, W, K, ngroups*d_state)
            dts = dt.unsqueeze(3).expand(-1, -1, -1, self.k_groups, -1)  # (B, H, W, K, nheads)
            xs = cross_scan_fn(x.view(B, H, W, self.d_inner), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch) # (B, H, W, K, d_inner)

            # Find the sequence length
            L = H * W

            # Flatten spatial dims
            Bs = Bs.contiguous().view(B, L, self.k_groups, self.ngroups, self.d_state)  # (B, L, K, ngroups, d_state)
            Cs = Cs.contiguous().view(B, L, self.k_groups, self.ngroups, self.d_state)
            dts = dts.contiguous().view(B, L, self.k_groups, self.nheads)  # (B, L, K, nheads)
            xs = xs.contiguous().view(B, L, self.k_groups, self.d_inner)  # (B, L, K, d_inner)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys = mamba_chunk_scan_combined(
                rearrange(xs, "b l k (h p) -> b l (k h) p", p=self.headdim),
                dts,
                As,
                rearrange(Bs, "b l k (g n) -> b l (k g) n", g=self.ngroups),
                rearrange(Cs, "b l k (g n) -> b l (k g) n", g=self.ngroups),
                chunk_size=self.chunk_size,
                Ds=rearrange(self.Ds, "k (h p) -> (k h) p", p=self.headdim) if self.D_has_hdim else self.D,
                z=None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=None,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            ys = rearrange(ys, "b l (k h) p -> b l k (h p)")
            y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, self.k_groups, self.d_inner), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)

            # NOTE: again we are not using the rmsnorm as it is implemented only for 1D
            # Multiply "gate" branch and apply extra normalization layer
            # y = self.norm(y, z)

            y = (self.out_norm(y.view(B, H, W, -1))).to(x.dtype)
            y = self.out_act(y)

            if not self.disable_z:
                y = y * z

            out = self.dropout(self.out_proj(y))

        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 96,
        d_state: int = 64,
        headdim: int = 96,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ssm = SS2D(d_model=hidden_dim, d_state=d_state, headdim=headdim, dropout=attn_drop_rate, bias=bias, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor):
        x_ln = x.permute(0, 2, 3, 1).contiguous()       # [B,C,H,W] --> [B,H,W,C]
        x_ln = self.ln_1(x_ln)                          # norm across C    
        out = self.drop_path(self.ssm(x_ln))
        out = out.permute(0, 3, 1, 2).contiguous()    # [B,H,W,C] --> [B,C,H,W]
        
        return x + out