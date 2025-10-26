# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

class Mamba2Simple(nn.Module):
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
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

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

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

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
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)


    # ---------- Higher-Order Interactions (state-space, reference) ----------
    def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None, return_V=False, U_override=None):
        """
        Arguments:
            X: (batch, length, n_heads, d_head)
            A: (batch, length, n_heads)                # log-decays; a_t = exp(A_t)
            B: (batch, length, n_heads, d_state)
            C: (batch, length, n_heads, d_state)
        Returns:
            Y: (batch, length, n_heads, d_head)
            final_state: (batch, n_heads, d_head, d_state)
            [optional] state_features: (batch, length, n_heads, d_head, d_state)  # pre-readout per-timestep states
        """
        assert X.dtype == A.dtype == B.dtype == C.dtype
        assert X.shape[1] % block_len == 0

        if U_override is None:
            X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
        else:
            X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
            U = rearrange(U_override, "b (c s) h p n -> b c s h p n", s=block_len)  # (b,c,s,h,p,n)

        A = rearrange(A, "b c l h -> b h c l")           # (b,h,c,l)
        A_cumsum = torch.cumsum(A, dim=-1)               # (b,h,c,l)

        # 1) Intra-chunk (diagonal blocks)
        L = torch.exp(segsum(A))                         # (b,h,c,l,l)
        if U_override is None:
            Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
        else:
            Y_diag = torch.einsum("bclhn,bhcls,bcshpn->bclhp", C, L, U)

        # 2) Right term for off-diagonal (B terms)
        decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)  # (b,h,c,l)
        if U_override is None:
            states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)   # (b,c,h,p,n)
        else:
            states = torch.einsum("bhcl,bcshpn->bchpn", decay_states, U)           # (b,c,h,p,n)

        # 3) Inter-chunk SSM recurrence (A terms)
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])     # (b,1,h,p,n)
        states = torch.cat([initial_states, states], dim=1)      # (b,c+1,h,p,n)
        decay_chunk = torch.exp(segsum(F.pad(A_cumsum[..., -1], (1, 0))))  # (b,h,c+1,c+1)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)        # (b,c+1,h,p,n)
        states, final_state = new_states[:, :-1], new_states[:, -1]                # (b,c,h,p,n), (b,h,p,n)

        # 4) Left term: state -> output per chunk (C terms)
        state_decay_out = torch.exp(A_cumsum)                                      # (b,h,c,l)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

        # Output
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

        if not return_V:
            return Y, final_state

        # Per-timestep states before contracting with C
        if U_override is None:
            state_features_diag = torch.einsum("bcshn,bhcls,bcshp->bclhpn", B, L, X)        # (b,c,l,h,p,n)
        else:
            state_features_diag = torch.einsum("bhcls,bcshpn->bclhpn", L, U)                 # (b,c,l,h,p,n)
        state_features_off = torch.einsum("bchpn,bhcl->bclhpn", states, state_decay_out)     # (b,c,l,h,p,n)
        state_features = rearrange(state_features_diag + state_features_off, "b c l h p n -> b (c l) h p n")

        return Y, final_state, state_features


    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states = repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            x, B, C = torch.split(
                xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
            )

            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # --- Higher-order interactions via SSD (P scans) ---
            P = 2  # set self.hoi_order > 1 to enable HOI; default=1 (off)
            if P >= 2:
                X = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)              # (B,L,H,P)
                B_ = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)            # (B,L,H,N)
                C_ = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)            # (B,L,H,N)

                # Discrete form consistent with SSD parity: a_t = exp(A * dt); scale X by dt
                A_eff = A[None, None, :] * dt                                       # (B,L,H)
                X_eff = X * dt.unsqueeze(-1)                                        # (B,L,H,P)

                # Order-1: get per-timestep pre-readout states for shifting
                _, _, state_features_prev = Mamba2Simple.ssd_minimal_discrete(
                    X_eff, A_eff, B_, C_, self.chunk_size, return_V=True
                )

                # Precompute (B ⊗ X) once
                BX = B_.unsqueeze(-2) * X_eff.unsqueeze(-1)                          # (B,L,H,P,N)

                acc = None
                for _order in range(2, P + 1):
                    # shift in time (zero at t=0)
                    state_features_prev_shift = F.pad(
                        state_features_prev[:, :-1], (0, 0, 0, 0, 0, 0, 0, 1)
                    )  # (B,L,H,P,N)

                    # U_order = (B X) ⊙ v^{(order-1)}_{t-1}
                    Up = BX * state_features_prev_shift                              # (B,L,H,P,N)

                    # Order-k output using the same SSD routine, passing U_override
                    Yp, _, state_features_prev = Mamba2Simple.ssd_minimal_discrete(
                        X_eff, A_eff, B_, C_, self.chunk_size,
                        initial_states=None, return_V=True, U_override=Up
                    )  # Yp: (B,L,H,P); state_features_prev: (B,L,H,P,N)

                    acc = Yp if acc is None else acc + Yp

                if acc is not None:
                    y = y + rearrange(acc, "b l h p -> b l (h p)")
            # --- end HOI ---

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)

        return out
