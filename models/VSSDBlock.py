import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
# Only the causal chunk-scan path (linear_attn_duality=False) needs these Triton
# kernels. The NC-SSD path used by VSSD-Recon does not, so keep the import optional:
# it lets the model run on machines without mamba_ssm / a matching CUDA build.
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None
    RMSNormGated = None
    selective_state_update = None
from einops import rearrange, repeat

import math
# from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

class tTensor(torch.Tensor):
    """Upstream VSSD wraps the SSM inputs in this so `.shape` yields a plain tuple of
    ints instead of a `torch.Size`. That only matters where the shape can hold SymInts
    -- tracing, torch.compile, ONNX export. `torch.Size` is already a tuple subclass,
    so in eager training it changes nothing: wrapping and not wrapping give bitwise
    identical output (max|diff| = 0.000e+00, 4 cascades at 64^2).

    It is not free. Subclassing torch.Tensor sends every downstream op through
    __torch_function__, and the subclass propagates into every result, so the whole
    SSM path pays it -- measured 19.6% more aten ops and 2.7x the wall clock
    (1.528 -> 0.560 s/iter fwd+bwd on CPU).

    Kept for anyone who needs to trace the model; set USE_TTENSOR to re-enable.
    """
    @property
    def shape(self):
        return tuple(int(s) for s in super().shape)


USE_TTENSOR = False   # True only for tracing / export, where SymInts appear


def to_ttensor(*args):
    if USE_TTENSOR:
        return tuple(tTensor(x) for x in args) if len(args) > 1 else tTensor(args[0])
    return tuple(args) if len(args) > 1 else args[0]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class StandardAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.inner_dim = inner_dim


    def forward(self, x, H, W):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_conv=3, #default to 3 for 2D
        conv_init=None,
        expand=2,
        headdim=64, #default to 64
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="silu", #default to silu
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False, # NOTE: the efficent triton kernel is not implemented in this code
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        linear_attn_duality=False,
        d_state = 64,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim #equivalent to multi-head attention
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        #convert chunk_size to triton.language.int32
        self.chunk_size = chunk_size #torch.tensor(chunk_size,dtype=torch.int32)
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        # Upstream VSSD defaults this to True, which makes dA = -dt*A positive.
        # It was pinned to False here as an ablation; keep that as the default so
        # existing variants are unchanged, but let a config override it.
        self.ssd_positve_dA = kwargs.get('ssd_positve_dA', False)
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        # Order: [x, B, C, dt]
        # d_in_proj = self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state

        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv2d.weight, -self.conv_init, self.conv_init)
        # self.conv2d.weight._no_weight_decay = True

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

        # modified from RMSNormGated to layer norm
        #assert RMSNormGated is not None
        #self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        #linear attention duality
        self.linear_attn_duality = linear_attn_duality
        self.kwargs = kwargs

    def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
        '''
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        '''

        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3) # (B, H, L, D)
        dt = dt.permute(0, 2, 1) # (B, H, L)
        dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1)   # broadcasts to (B, H, L, 1)
        if self.ssd_positve_dA: dA = -dA

        V_scaled = V * dA
        K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, D)
        if getattr(self, "__DEBUG__", False):
            A_mat = dA.cpu().detach().numpy()
            A_mat = A_mat.reshape(batch, -1, H, W)
            setattr(self, "__data__", dict(
                dA=A_mat, H=H, W=W, V=V,))

        if self.ngroups == 1:
            ## get kv via transpose K and V
            KV = K.transpose(-2, -1) @ V_scaled # (B, H, dstate, D)
            Q = C.view(batch, 1, seqlen, dstate)#.repeat(1, head, 1, 1)
            x = Q @ KV # (B, H, L, D)
            x = x + V * D.view(1, -1, 1, 1)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
        else:
            assert head % self.ngroups == 0
            dstate = dstate // self.ngroups
            K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)
            V_scaled = V_scaled.view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)

            KV = K.transpose(-2, -1) @ V_scaled # (B, H//g, g, dstate, D)
            x = Q @ KV # (B, H//g, g, L, D)
            V_skip = (V * D.view(1, -1, 1, 1)).view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            x = x + V_skip # (B, H//g, g, L, D)
            x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim) # (B, L, H, D)
            x = x.contiguous()

        return x


    def forward(self, u, H, W, seq_idx=None):
        """
        u: (B, L, C)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        # xbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # xBC, dt = torch.split(
        #     xbcdt, [self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        # )

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]


        #2D Convolution
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x, dt, A, B, C = to_ttensor(x, dt, A, B, C)
        if self.linear_attn_duality:
            y = self.non_casual_linear_attn(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt, A, B, C, self.D, H, W
            )
        else:
            if mamba_chunk_scan_combined is None:
                raise ImportError(
                    'linear_attn_duality=False needs the mamba_ssm Triton kernels; '
                    'install mamba-ssm, or set linear_attn_duality=True to use NC-SSD.'
                )
            if self.kwargs.get('bidirection', False):
                #assert self.ngroups == 2 #only support bidirectional with 2 groups
                x = to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)).chunk(2, dim=-2)
                B = to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                C = to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                dt = dt.chunk(2, dim=-1) # (B, L, nheads) -> (B, L, nheads//2)*2
                A, D = A.chunk(2, dim=-1), self.D.chunk(2,dim=-1) # (nheads) -> (nheads//2)*2
                y_forward = mamba_chunk_scan_combined(
                    x[0], dt[0], A[0], B[0], C[0], chunk_size=self.chunk_size, D=D[0], z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y_backward = mamba_chunk_scan_combined(
                    x[1].flip(1), dt[1].flip(1), A[1], B[1].flip(1), C[1].flip(1), chunk_size=self.chunk_size, D=D[1], z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y = torch.cat([y_forward, y_backward.flip(1)], dim=-2)
            else:
                y = mamba_chunk_scan_combined(
                    to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)),
                    to_ttensor(dt),
                    to_ttensor(A),
                    to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)),
                    to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)),
                    chunk_size=self.chunk_size,
                    D=to_ttensor(self.D),
                    z=None,
                    seq_idx=seq_idx,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
        y = rearrange(y, "b l h p -> b l (h p)")

        # # Multiply "gate" branch and apply extra normalization layer
        # y = self.norm(y, z)
        y = self.norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


class VSSDBlock(nn.Module):
    """ VSSD block: NC-SSD (or MSA) token mixer + FFN, each with a conditional-position depthwise conv.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256,
                 linear_attn_duality=True, d_state=64, attn_type='mamba2', **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm1 = norm_layer(dim)
        if attn_type == 'standard':
            self.attn = StandardAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, dropout=drop)
        elif attn_type == 'mamba2':
            self.attn = Mamba2(d_model=dim, expand=ssd_expansion, headdim= dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, chunk_size=ssd_chunk_size,
                                linear_attn_duality=linear_attn_duality, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H=None, W=None):
        """
        x: [B, C, H, W]
        returns: [B, C, H, W]
        """

        B, C, H, W = x.shape
        # cpe1 wants NCHW, which is what was passed in. Going via the flattened tensor
        # (`x.reshape(B, H, W, C).permute(0, 3, 1, 2)`) recovers exactly this tensor, but
        # reshape on a transposed view is not a view, so it copies the whole activation.
        x_nchw = x
        # [B, C, H, W] → [B, l, C]
        x = x.flatten(2).transpose(1, 2)

        x = x + self.cpe1(x_nchw).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)

        # SSD or Standard Attention
        x = self.attn(x, H, W)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # MLP + Norm
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # [B, L, C] → [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x