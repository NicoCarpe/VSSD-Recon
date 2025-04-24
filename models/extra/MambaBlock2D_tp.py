# Copyright (c) 2024, Tri Dao, Albert Gu.

# 2D extension following VMamba formulation: 2025, Nicolas Carpenter 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class MambaBlock2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=3,   # NOTE: 4 used in mamba, look to see why
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
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
        process_group=None,
        sequence_parallel=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        #======================================================================
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        #======================================================================
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.d_inner = (self.expand * self.d_model) // self.world_size
        self.headdim = headdim
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # number of sequences
        # NOTE: will likely need to make consideration for worldsize based off K
        self.K = 4

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv2d = nn.squential(
            Permute(0, 3, 1, 2),  # [B,H,W,C] -> [B,C,H,W]
            nn.Conv2d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                # NOTE: apparently convolutions dont need this explicit padding?
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            ),
            Permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv2d.weight, -self.conv_init, self.conv_init)
        # self.conv2d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dts = torch.exp(
            torch.rand(self.K, self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
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
        As = torch.empty(self.K, self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_logs = torch.log(As).to(dtype=dtype)
        self.A_logs = nn.Parameter(A_logs)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_logs._no_weight_decay = True

        # D "skip" parameter
        self.Ds = nn.Parameter(
            torch.ones(
                self.K, 
                self.d_inner if self.D_has_hdim else self.nheads,
                device=device
            )
        )
        self.Ds._no_weight_decay = True

        # Extra normalization layer right before output projection
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_inner // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)


    def forward(
        self, 
        u, 
        # ==============================
        selective_scan_backend = None,
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        seq_idx=None
    ):
        """
        u: (B, H, W, D)
        Returns: same shape as u
        """        
        batch, seqlen, dim = u.shape
        
        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        # Input Projection
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)

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

            # if self.process_group is not None:
            #     reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
            #     out = reduce_fn(out, self.process_group)

        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], 
                dim=-1
            )

            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 2D Convolution
            xBC = self.act(
                self.conv2d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
            )  # (B, H, W, self.d_inner + 2 * ngroups * d_state)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

            B, H, W, _ = x.shape

            # Expand B, C, dt to K directions
            Bs = B.unsqueeze(3).expand(-1, -1, -1, self.K, -1)  # (B, H, W, K, ngroups*d_state)
            Cs = C.unsqueeze(3).expand(-1, -1, -1, self.K, -1)  # (B, H, W, K, ngroups*d_state)
            dts = dt.unsqueeze(3).expand(-1, -1, -1, self.K, -1)  # (B, H, W, K, nheads)
            xs = cross_scan_fn(x.view(B, H, W, self.d_inner), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch) # (B, H, W, K, d_inner)

            L = H * W

            # Flatten spatial dims
            Bs = Bs.contiguous().view(B, L, self.K, self.ngroups, self.d_state)  # (B, L, K, ngroups, d_state)
            Cs = Cs.contiguous().view(B, L, self.K, self.ngroups, self.d_state)
            dts = dts.contiguous().view(B, L, self.K, self.nheads)  # (B, L, K, nheads)
            xs = xs.contiguous().view(B, L, self.K, self.d_inner)  # (B, L, K, d_inner)

            ys = mamba_chunk_scan_combined(
                rearrange(xs, "b l k (h p) -> b l (k h) p", p=self.headdim),
                dts,
                As,
                rearrange(Bs, "b l k (g n) -> b l (k g) n", g=self.ngroups),
                rearrange(Cs, "b l k (g n) -> b l (k g) n", g=self.ngroups),
                chunk_size=self.chunk_size,
                Ds=rearrange(self.Ds, "k (h p) -> (k h) p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l k (h p) -> b l (k h) p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=None,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            ys = rearrange(y, "b l (k h) p -> b l k (h p)")
            y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, self.K, self.d_inner), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)

            # Multiply "gate" branch and apply extra normalization layer
            if self.rmsnorm:
                y = self.norm(y, z)

            out = self.out_proj(y)

        return out