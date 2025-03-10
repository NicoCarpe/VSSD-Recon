# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined



class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
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
        self.dt_limit = dt_limit
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(
                self.d_inner if self.D_has_hdim else self.nheads,
                device=device
            )
        )
        self.D._no_weight_decay = True

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


    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=None,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )

            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)

        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], 
                dim=-1
            )

            # this seems additional to the simple version
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
            )  # (B, L, self.d_ssm + 2 * ngroups * d_state)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=None,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            if self.rmsnorm:
                y = self.norm(y, z)

            out = self.out_proj(y)

        return out