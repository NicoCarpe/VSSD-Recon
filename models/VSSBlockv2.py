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
        activation=nn.GELU(),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=64,
        use_mem_eff_path=False,
        layer_idx=None,  # Absorb kwarg for general module
        oact = True,
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
        self.act = activation    # NOTE: VMamba uses nn.GELU(), Mamba2 uses nn.SiLU()
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        
        # VMamba inits 
        self.K = 4
        self.oact = oact

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias, **factory_kwargs)
  
        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.K, self.headdim, self.d_state, **factory_kwargs))
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
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dts = dt + torch.log(-torch.expm1(-dt))
        # dt_bias: (nheads,)
        self.dt_bias = nn.Parameter(inv_dts)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # As parameter:
        #  (K, nheads)
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        As = torch.empty(self.K, self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_logs = torch.log(As).to(dtype=dtype)
        # A_logs:
        self.A_logs = nn.Parameter(A_logs)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_logs._no_weight_decay = True

        # Ds "skip" parameter:
        #  (K, nheads)  or (K, d_inner) if D_has_hdim
        self.Ds = nn.Parameter(
            torch.ones(
                self.K, 
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
        selective_scan_backend = "triton",
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        seq_idx=None
    ):
        """
        u: (batch, H, W, channel)
        Returns: (batch, H, W, channel)
        """        
        batch, H, W, _ = u.shape

        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]        
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        # Input Projection
        zxbcdt = self.in_proj(u)  # (batch, H, W, d_in_proj)
        
        z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], 
                dim=-1
            )  

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        As = -torch.exp(self.A_logs.float())  # (K, nheads)
        As = As.view(-1) # (K * nheads)

        # dt_bias: (nheads) → (K * nheads)
        dts_bias = self.dt_bias.float().repeat(self.K)

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
            # 2D Convolution
            # (batch, H, W, self.d_inner + 2 * ngroups * d_state)
            xBC = rearrange(xBC, "b h w c -> b c h w")
            xBC = self.conv2d(xBC)  #NOTE: electing not to use truncation, should be covered by padding [:, :-(self.d_conv - 1)]
            xBC = rearrange(xBC, "b c h w -> b h w c")
            xBC = self.act(xBC)
            
            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

            # Find the sequence length
            L = H*W

            # Expand to K directions
            Bs = B.unsqueeze(3).expand(-1, -1, -1, self.K, -1)  # (batch, H, W, K, ngroups*d_state)
            Cs = C.unsqueeze(3).expand(-1, -1, -1, self.K, -1)  # (batch, H, W, K, ngroups*d_state)
            dts = dt.unsqueeze(3).expand(-1, -1, -1, self.K, -1)  # (batch, H, W, K, nheads)

            # build the four sequences
            x_chw  = x.permute(0,3,1,2).contiguous()                        # [B, C, H, W]
            x_flat = x_chw.view(batch, -1, L)                               # [B, C, L]
            x_t    = x_chw.transpose(2,3).contiguous().view(batch, -1, L)   # [B, C, L]

            xs = torch.stack([
                x_flat,            # → 
                x_t,               # ↓
                x_flat.flip(-1),   # ←
                x_t   .flip(-1),   # ↑
            ], dim=1)                    # [B, K=4, C, L]

            # xs = cross_scan_fn(x.view(batch, H, W, self.d_inner), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch) # (batch, H, W, K, d_inner)

            # Flatten spatial dims
            Bs = Bs.contiguous().view(batch, L, self.K, self.ngroups*self.d_state)  # (batch, L, K, ngroups, d_state)
            Cs = Cs.contiguous().view(batch, L, self.K, self.ngroups*self.d_state)  # (batch, L, K, ngroups, d_state)
            dts = dts.contiguous().view(batch, L, self.K, self.nheads)  # (batch, L, K, nheads)
            xs = xs.permute(0,1,3,2).reshape(batch, L, self.K, self.d_inner)
            
            # xs = xs.contiguous().view(batch, L, self.K, self.d_inner)  # (batch, L, K, d_inner)

            Ds = rearrange(self.Ds, "k (h p) -> (k h) p", p=self.headdim) if self.D_has_hdim else self.Ds.view(-1)
            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys = mamba_chunk_scan_combined(
                rearrange(xs, "b l k (h p) -> b l (k h) p", p=self.headdim),
                rearrange(dts, "b l k h -> b l (k h)"),
                As,
                rearrange(Bs, "b l k (g n) -> b l (k g) n", g=self.ngroups),
                rearrange(Cs, "b l k (g n) -> b l (k g) n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=Ds, 
                z=None,
                dt_bias=dts_bias,
                dt_softplus=True,
                seq_idx=None,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )

            # ys has shape [B, L, K·H, P] after the scan, where H=self.nheads, 
            # P=self.headdim so that H*P = self.d_inner
            # First pull it back to [B, L, K, H, P]:
            out_y = ys.view(batch, L, self.K, self.nheads, self.headdim)

            # Then swap L↔(H·P) and split out the four directions:
            # We want [B, K, H·P, L], so:
            out_y = out_y.permute(0, 2, 3, 4, 1).reshape(batch, self.K, self.d_inner, L)

            # Now reconstruct the four directional maps without ever naming “C”:
            # → (0), ↓ (1), ← (2), ↑ (3)
            y0 = out_y[:, 0].view(batch, -1, H, W)                     # →  [B, C, H, W]
            y1 = out_y[:, 1].view(batch, -1, W, H).transpose(2, 3)     # ↓  [B, C, H, W]
            y2 = out_y[:, 2].view(batch, -1, H, W).flip(-1)            # ←  [B, C, H, W]
            y3 = out_y[:, 3].view(batch, -1, W, H).transpose(2, 3).flip(-1)  # ↑  [B, C, H, W]

            # Sum them and put back into [B, H, W, C]
            y = (y0 + y1 + y2 + y3).permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

            # ys = rearrange(ys, "b l (k h) p -> b l k (h p)", k=self.K)
            # y: torch.Tensor = cross_merge_fn(ys.view(batch, H, W, self.K, self.d_inner), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)

            # NOTE: again we are not using the rmsnorm as it is implemented only for 1D
            # Multiply "gate" branch and apply extra normalization layer
            # y = self.norm(y, z)
            # y = (self.out_norm(y.view(batch, H, W, -1))).to(x.dtype)
            y = self.out_norm(y).to(x.dtype)

            # Apply output activation and gating
            y = self.out_act(y) * z

            # Apply output projection and dropout
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
