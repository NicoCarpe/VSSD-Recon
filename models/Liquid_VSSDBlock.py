import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import DropPath
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


class tTensor(torch.Tensor):
    """Torch tensor with .shape -> plain ints (mirrors some Mamba kernels)."""
    @property
    def shape(self):
        shape = super().shape
        return tuple(int(s) for s in shape)


def to_ttensor(*args):
    if len(args) == 1:
        return tTensor(args[0])
    return tuple(tTensor(x) for x in args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class StandardAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **_):
        super().__init__()
        inner_dim    = dim_head * heads
        self.heads   = heads
        self.scale   = dim_head ** -0.5
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out  = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, H=None, W=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=self.heads) for t in qkv]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out  = torch.einsum('bhij,bhjd->bhid', attn, v)
        out  = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Liquid_VSSD(nn.Module):
    def __init__(
        self,
        d_model,
        d_conv=3,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="silu",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=False,
        layer_idx=None,
        device=None,
        dtype=None,                 # assume global default is fp32
        linear_attn_duality=True,
        d_state=64,

        # higher-order controls
        max_order=1,
        learnable_order_weights=False,  # NEW: only create gamma if True
        per_order_dropout=0.0,     # e.g., 0.05–0.1
        per_order_norm=False,       # small LN on Pv per order
        center_tokens=False,        # mean-centre over tokens (no RMS over L)
        disable_skip=False,

        # stabilizers
        ssd_positive_dA=False,
        log_block=False,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.expand  = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        self.ngroups = (self.d_inner // self.headdim) if ngroups == -1 else ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads  = self.d_inner // self.headdim

        self.dt_limit  = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.kwargs = kwargs

        # in-proj over [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        # depthwise conv over [x, B, C]
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim, out_channels=conv_dim, groups=conv_dim,
            bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs
        )
        if conv_init is not None:
            nn.init.uniform_(self.conv2d.weight, -conv_init, conv_init)

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # dt bias (per head)
        dt = torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt); self.dt_bias._no_weight_decay = True

        # A (per head) & D (skip)
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype)); self.A_log._no_weight_decay = True
        self.D     = nn.Parameter(torch.ones(self.nheads, device=device)); self.D._no_weight_decay = True

        self.norm     = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.linear_attn_duality = linear_attn_duality
        self.ssd_positive_dA = ssd_positive_dA

        # higher-order knobs
        self.P = int(max_order)
        self.center_tokens = center_tokens
        self.disable_skip = disable_skip
        self.learnable_order_weights = learnable_order_weights
        self.log_block = log_block

        # softmax gating over orders (only if enabled)
        if self.P > 0 and self.learnable_order_weights:
            self.gamma_logits = nn.Parameter(torch.zeros(self.P, device=device, dtype=dtype))  # init ~ uniform
        else:
            self.gamma_logits = None

        # optional per-order norm/drop
        self.per_order_norm = per_order_norm
        self.per_order_dropout_p = per_order_dropout
        if self.P > 0 and (per_order_norm or per_order_dropout > 0.0):
            self.order_norms = nn.ModuleList([
                nn.LayerNorm(self.headdim) if per_order_norm else nn.Identity()
                for _ in range(self.P)
            ])
            self.order_drop  = nn.ModuleList([
                nn.Dropout(per_order_dropout) if per_order_dropout > 0.0 else nn.Identity()
                for _ in range(self.P)
            ])
        else:
            self.order_norms = None
            self.order_drop  = None

    # --------- utils ---------
    @staticmethod
    def _center_over_dim(x, dim):
        return x - x.mean(dim=dim, keepdim=True)

    def _maybe_center_tokens(self, K, V_scaled):
        if not self.center_tokens:
            return K, V_scaled
        # mean-center over token dimension only (no RMS over L)
        if K.dim() == 4:   # (B,1,L,N)
            Kc = self._center_over_dim(K, dim=2)
        else:              # (B,1,g,L,n_g)
            Kc = self._center_over_dim(K, dim=3)
        Vc = self._center_over_dim(V_scaled, dim=2)  # (B,H,L,Pv) or (B,Hg,g,L,Pv)
        return Kc, Vc

    # --------- core higher-order block ---------
    def _higher_order_nc_ssd(self, x, dt, A, B, C, D, P: int, H_img=None, W_img=None):
        """
        x:  (B, L, H, Pv)
        dt: (B, L, H)
        A:  (H,)
        B:  (B, L, N)
        C:  (B, L, N)
        D:  (H,)
        """
        batch, seq_len, nheads, head_dim = x.shape
        assert self.nheads == nheads
        assert (B.shape[-1] % self.ngroups) == 0
        assert (nheads % self.ngroups) == 0

        # Values and scales
        V   = x.permute(0, 2, 1, 3)     # (B,H,L,Pv)
        dth = dt.permute(0, 2, 1)       # (B,H,L)

        # In forward() we define A = -exp(A_log) to match NC-SSD; allow flipping to positive if requested
        A_eff = -A if getattr(self, "ssd_positive_dA", False) else A
        m = dth * A_eff.view(1, -1, 1)  # (B,H,L)
        V_scaled = V * m.unsqueeze(-1)  # (B,H,L,Pv)

        # Compute gamma weights only if enabled
        gamma_w = None
        if self.gamma_logits is not None:
            gamma_w = self.gamma_logits.softmax(dim=0)  # (P,)

        if self.ngroups == 1:
            N = B.shape[-1]
            K = B.view(batch, 1, seq_len, N)   # (B,1,L,N)
            Q = C.view(batch, 1, seq_len, N)   # (B,1,L,N)

            # center-only over tokens
            K_base, V_base = self._maybe_center_tokens(K, V_scaled)

            # powers start at p=1
            K_pow = K_base
            V_pow = V_base

            # e0 buffer and S buffer
            e_prev = [torch.ones((batch, nheads, N, head_dim), device=K.device, dtype=K.dtype)]
            S_list = []

            # STREAMED ACCUMULATOR (lower memory)
            contracted = 0.0
            order_maps_2d = [] if (self.log_block and H_img is not None and W_img is not None) else None

            for p in range(1, P + 1):
                # S_p = (K_pow^T) @ V_pow  -> (B,H,N,Pv) via broadcast on H)
                S_p = K_pow.transpose(-2, -1) @ V_pow
                S_list.append(S_p)

                # Newton recurrence for e_p (exact, no S_p RMS norm)
                # e_p = (1/p) * Σ_{k=1..p} (-1)^{k-1} e_{p-k} ⊙ S_k
                e_p = torch.zeros_like(S_p)
                for m_idx in range(1, p + 1):
                    sign = 1.0 if (m_idx & 1) else -1.0
                    e_p = e_p.addcmul(sign * e_prev[p - m_idx], S_list[m_idx - 1])
                e_p = e_p / float(p)
                e_prev.append(e_p)

                # project to tokens: (B,1,L,N) @ (B,H,N,Pv) -> (B,H,L,Pv)
                y_p = Q @ e_p

                if self.order_norms is not None:
                    y_p = self.order_norms[p - 1](y_p)
                if self.order_drop is not None:
                    y_p = self.order_drop[p - 1](y_p)

                # softmax weight for this order (only if enabled)
                if gamma_w is not None:
                    y_p = gamma_w[p - 1] * y_p

                # stream add
                contracted = contracted + y_p

                # lightweight per-order map (CPU, detached)
                if order_maps_2d is not None:
                    mag = y_p.pow(2).sum(dim=(1, 3)).sqrt()                 # (B, L)
                    order_maps_2d.append(mag.view(batch, H_img, W_img).detach().cpu())

                # next powers
                if p < P:
                    K_pow = K_pow * K_base
                    V_pow = V_pow * V_base

            D_eff = self.D if not self.disable_skip else (self.D * 0.0)
            skip = V * D_eff.view(1, nheads, 1, 1)
            out_heads  = contracted + skip
            out = out_heads.permute(0, 2, 1, 3).contiguous()  # (B,L,H,Pv)

            logs = {"gamma": gamma_w}

            if self.log_block and H_img is not None and W_img is not None:
                m_map = m.mean(dim=1).view(batch, H_img, W_img)
                order_sum = out_heads.pow(2).sum(dim=(1, 3)).sqrt().view(batch, H_img, W_img)
                # `order_maps_2d` was filled in the loop (already detach().cpu() there)
                logs.update({
                    "m_map": m_map.detach().cpu(),
                    "order_maps": order_maps_2d,
                    "order_sum": order_sum.detach().cpu(),
                })

            return out, logs

        # -------- grouped case --------
        N = B.shape[-1]
        N_group = N // self.ngroups
        Hg = nheads // self.ngroups

        K = B.view(batch, 1, self.ngroups, seq_len, N_group)     # (B,1,g,L,n_g)
        Q = C.view(batch, 1, self.ngroups, seq_len, N_group)     # (B,1,g,L,n_g)

        Vg = V.view(batch, Hg, self.ngroups, seq_len, head_dim)  # (B,Hg,g,L,Pv)
        mg = m.view(batch, Hg, self.ngroups, seq_len, 1)         # (B,Hg,g,L,1)
        Vg_scaled = Vg * mg

        # center-only over tokens
        if self.center_tokens:
            K_base = K - K.mean(dim=3, keepdim=True)
            V_base = Vg_scaled - Vg_scaled.mean(dim=3, keepdim=True)
        else:
            K_base, V_base = K, Vg_scaled

        K_pow = K_base
        V_pow = V_base

        e_prev = [torch.ones((batch, Hg, self.ngroups, N_group, head_dim), device=K.device, dtype=K.dtype)]
        S_list = []
        contracted = 0.0
        order_maps_2d = [] if (self.log_block and H_img is not None and W_img is not None) else None

        for p in range(1, P + 1):
            # S_p: (B,1,g,L,n_g)^T @ (B,Hg,g,L,Pv) -> (B,Hg,g,n_g,Pv)
            S_p = K_pow.transpose(-2, -1) @ V_pow
            S_list.append(S_p)

            e_p = torch.zeros_like(S_p)
            for m_idx in range(1, p + 1):
                sign = 1.0 if (m_idx & 1) else -1.0
                e_p = e_p.addcmul(sign * e_prev[p - m_idx], S_list[m_idx - 1])
            e_p = e_p / float(p)
            e_prev.append(e_p)

            # Project: (B,1,g,L,n_g) @ (B,Hg,g,n_g,Pv) -> (B,Hg,g,L,Pv)
            y_p = Q @ e_p
            if self.order_norms is not None:
                y_p = self.order_norms[p - 1](y_p)
            if self.order_drop is not None:
                y_p = self.order_drop[p - 1](y_p)

            if gamma_w is not None:
                y_p = gamma_w[p - 1] * y_p

            contracted = contracted + y_p

            if order_maps_2d is not None:
                mag = y_p.pow(2).sum(dim=(1, 2, 4)).sqrt()                # (B, L)
                order_maps_2d.append(mag.view(batch, H_img, W_img).detach().cpu())

            if p < P:
                K_pow = K_pow * K_base
                V_pow = V_pow * V_base

        D_eff = self.D if not self.disable_skip else (self.D * 0.0)
        V_skip = (V * D_eff.view(1, nheads, 1, 1)).view(batch, Hg, self.ngroups, seq_len, head_dim)
        out_grouped = contracted + V_skip
        out = out_grouped.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seq_len, nheads, head_dim).contiguous()

        logs = {"gamma": gamma_w}

        if self.log_block and H_img is not None and W_img is not None:
            m_map = m.mean(dim=1).view(batch, H_img, W_img)
            order_sum = out_grouped.pow(2).sum(dim=(1, 2, 4)).sqrt().view(batch, H_img, W_img)
            # `order_maps_2d` was filled in the loop (already detach().cpu() there)
            logs.update({
                "m_map": m_map.detach().cpu(),
                "order_maps": order_maps_2d,
                "order_sum": order_sum.detach().cpu(),
            })

        return out, logs


    def forward(self, u, H_img, W_img, seq_idx=None):
        """
        u: (B, L, C) where L = H_img * W_img
        returns: (B, L, C)
        """
        B, L, _ = u.shape

        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        dt = F.softplus(dt + self.dt_bias)   # (B, L, H_heads)
        A  = -torch.exp(self.A_log)          # (H_heads,)  (sign matches NC-SSD convention)

        # depthwise conv over [x,B,C]
        xBC = xBC.view(B, H_img, W_img, -1).permute(0, 3, 1, 2).contiguous()   # (B, conv_dim, H, W)
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(B, H_img * W_img, -1).contiguous()  # (B, L, conv_dim)

        # split -> x=V, B=K, C=Q
        x, Bk, Cq = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        x, dt, A, Bk, Cq = to_ttensor(x, dt, A, Bk, Cq)

        if self.linear_attn_duality:
            y, logs = self._higher_order_nc_ssd(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt, A, Bk, Cq, self.D, self.P, H_img, W_img
            )
        else:
            y = mamba_chunk_scan_combined(
                to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)),
                to_ttensor(dt),
                to_ttensor(A),
                to_ttensor(rearrange(Bk, "b l (g n) -> b l g n", g=self.ngroups)),
                to_ttensor(rearrange(Cq, "b l (g n) -> b l g n", g=self.ngroups)),
                chunk_size=self.chunk_size,
                D=to_ttensor(self.D),
                z=None,
                seq_idx=seq_idx,
                initial_states=repeat(self.init_states, "... -> b ...", b=B) if self.learnable_init_states else None,
                **({} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)),
            )
            logs = None

        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y)
        y = y * F.silu(z)
        return self.out_proj(y), logs


class Liquid_VSSDBlock(nn.Module):
    """
    ViT-like block:
      x = x + depthwise_conv(x)
      x = x + Liquid_VSSD(norm(x))
      x = x + MLP(norm(x))
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=True,
        drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,

        ssd_expansion=2,
        ssd_ngroups=1,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        d_state=64,
        attn_type='liquid_vssd',

        max_order=4,
        ssd_positive_dA=False,
        log_block=True,
        disable_skip=False,

        # stabilizers in use
        per_order_dropout=0.00,
        per_order_norm=False,       # Really expensive + causes NaN errors
        center_tokens=False,
        learnable_order_weights=False,  
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm1 = norm_layer(dim)

        if attn_type == 'standard':
            self.attn = StandardAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, dropout=drop)
        elif attn_type == 'liquid_vssd':
            self.attn = Liquid_VSSD(
                d_model=dim,
                expand=ssd_expansion,
                headdim=dim * ssd_expansion // num_heads,
                ngroups=ssd_ngroups,
                chunk_size=ssd_chunk_size,
                linear_attn_duality=linear_attn_duality,
                d_state=d_state,
                max_order=max_order,
                ssd_positive_dA=ssd_positive_dA,
                log_block=log_block,
                disable_skip=disable_skip,
                learnable_order_weights=learnable_order_weights,
                per_order_dropout=per_order_dropout,
                per_order_norm=per_order_norm,
                center_tokens=center_tokens,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown attn_type={attn_type}")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2  = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm2 = norm_layer(dim)
        self.mlp   = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H=None, W=None):
        """
        x: [B, C, H, W] → returns: [B, C, H, W]
        """
        B, C, H_img, W_img = x.shape

        x = x.flatten(2).transpose(1, 2)

        x = x + self.cpe1(x.reshape(B, H_img, W_img, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        x, logs = self.attn(x, H_img, W_img)
        x = shortcut + self.drop_path(x)

        x = x + self.cpe2(x.reshape(B, H_img, W_img, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.transpose(1, 2).view(B, C, H_img, W_img)
        return x, logs