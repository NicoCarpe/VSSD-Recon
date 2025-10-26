"""
This file contains the basic modules for the model. 
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import matplotlib.pyplot as plt

from data import transforms
#from .VSSBlock import VSSBlock
from .VSSDBlock import VSSDBlock

def erf_backbone_central(promptmr, masked_kspace, mask, num_low_frequencies,
                         cascade_idx=0, device=None, average_over=None):
    """
    Computes ERF of the backbone (PromptUnet/liquid_VSSDBlock stack) w.r.t. the
    aliased, coil-combined input image that enters PatchEmbed. DC is disabled.
    - promptmr: your PromptMR model (in eval mode).
    - masked_kspace: (B, Nc, H, W, 2)
    - mask: (B, 1, H, W, 1) -> we will replace by zeros to disable DC
    - num_low_frequencies: (B,)
    - cascade_idx: which cascade’s backbone to probe (default 0)
    - average_over: if not None, iterate a dataloader and average ERFs
    """
    was_training = promptmr.training
    promptmr.eval()

    def _one_erf(masked_kspace, mask, nlf):
        # --- disable DC by zeroing the mask ---
        mask0 = torch.zeros_like(mask)

        # --- capture the image at the backbone input (before PatchEmbed) ---
        holder = {"x_in": None}
        hook = promptmr.cascades[cascade_idx].model.unet.patch_embed.register_forward_hook(
            lambda mod, inp, out: (inp[0].retain_grad(), holder.__setitem__("x_in", inp[0]))
        )

        # Forward full model (sens maps are computed normally upstream)
        out = promptmr(masked_kspace, mask0, nlf)
        img = out["img_pred"]                  # (B, 1, H, W), real image
        B, _, H, W = img.shape
        cy, cx = H // 2, W // 2
        scalar = img[0, 0, cy, cx]

        # Backprop to the backbone input image
        promptmr.zero_grad(set_to_none=True)
        scalar.backward()

        hook.remove()
        x_in = holder["x_in"]                  # (B, C_in, H', W')
        assert x_in is not None and x_in.grad is not None, "Hook/grad missing"

        # Grad magnitude over channels
        g = x_in.grad.detach().abs().sum(1, keepdim=True)   # (B,1,H',W')
        g = g / (g.max() + 1e-12)
        return g[0, 0]                                      # (H', W')

    # Single batch
    if average_over is None:
        gmap = _one_erf(masked_kspace, mask, num_low_frequencies)
        plt.figure(figsize=(4, 4)); plt.imshow(gmap.cpu(), cmap="magma"); plt.axis("off")
        plt.title("ERF (backbone only, central pixel)"); plt.show()
        if was_training: promptmr.train()
        return gmap

    # Average over a dataloader of test samples
    acc = None
    with torch.no_grad():
        for i, (mk, m, nlf) in enumerate(average_over):
            g = _one_erf(mk, m, nlf)   # this call backprops, so remove no_grad if you use your DataLoader
            acc = g if acc is None else acc + g
    gmean = acc / (i + 1)
    plt.figure(figsize=(4, 4)); plt.imshow(gmean.cpu(), cmap="magma"); plt.axis("off")
    plt.title("ERF (backbone only, averaged)"); plt.show()
    if was_training: promptmr.train()
    return gmean




##########################################################################
# ---------- Down Block -----------------------

class DownBlock(nn.Module):
    def __init__(self, in_dim, d_state, n_block, num_heads, dropout, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential(*[
            VSSDBlock(
                dim=in_dim,
                d_state=d_state,
                num_heads=num_heads,                
                drop=dropout,
                attn_type='mamba2',
                **kwargs
            ) for _ in range(n_block)
        ])
        
        self.down = PatchMerge(dim=in_dim)

    def forward(self, x):
        enc = self.encoder(x)  # Shape: (B, C, H, W)
        x_down = self.down(enc)  # (B, H//2, W//2, 2*C)

        return x_down, enc  # enc is (B, C, H, W) for skip connection


##########################################################################
# ---------- Up Block -----------------------

class UpBlock(nn.Module):
    def __init__(self, in_dim, d_state, n_block, num_heads, bias, dropout, n_history=0, **kwargs):
        super().__init__()
        # momentum layer
        self.n_history = n_history
        if n_history > 0:
            self.momentum = nn.Sequential(
                nn.Conv2d(in_dim*(n_history+1), in_dim, kernel_size=1, bias=bias),
                VSSDBlock(
                    dim=in_dim,
                    d_state = d_state,
                    num_heads = num_heads,
                    drop = dropout,
                    attn_type='mamba2',
                    **kwargs
                )
            )

        self.decoder = nn.Sequential(*[
            VSSDBlock(
                dim=in_dim//2,          # this operation happens after patch expand
                d_state = d_state,
                num_heads = num_heads,
                drop = dropout,
                attn_type='mamba2',
                **kwargs
            ) for _ in range(n_block)
        ])

        self.fuse = nn.Conv2d(in_dim, in_dim//2, kernel_size=1, bias=True)
        self.up = PatchExpand(dim=in_dim, dim_scale=2)

    def forward(self, x, skip, history_feat: Optional[torch.Tensor] = None):
        
        # momentum layer
        if self.n_history > 0:
            if history_feat is None:
                x = torch.cat([torch.tile(x, (1, self.n_history+1, 1, 1))], dim=1)
            else:
                x = torch.cat([x, history_feat], dim=1)

            x = self.momentum(x)

        x_up = self.up(x)
        x_cat = torch.cat([x_up, skip], dim=1)
        x_fused = self.fuse(x_cat)
        dec = self.decoder(x_fused)

        # x = self.up(x) + skip 
        # dec = self.decoder(x)


        return dec


##########################################################################
# ---------- Skip Block -----------------------

class SkipBlock(nn.Module):
    def __init__(self, enc_dim, d_state, n_cab, num_heads, dropout, **kwargs):
        super().__init__()
        if n_cab == 0:
            self.skip_attn = nn.Identity()
        else:
            self.skip_attn = nn.Sequential(*[
                VSSDBlock(
                    dim=enc_dim,
                    d_state = d_state,
                    num_heads = num_heads,
                    drop = dropout,
                    attn_type='mamba2',
                    **kwargs
                ) for _ in range(n_cab)
            ])

    def forward(self, x):
        x = self.skip_attn(x)
        return x


##########################################################################
# ---------- Kspace ACS Extractor -----------------------   

class KspaceACSExtractor:
    '''
    Extract ACS lines from k-space data
    '''

    def __init__(self, mask_center):
        self.mask_center = mask_center
        self.low_mask_dict = {}  # avoid repeated calculation

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        get the padding size and number of low frequencies for the center mask. For fastmri and cmrxrecon dataset
        '''
        if num_low_frequencies is None or (num_low_frequencies == -1).all():
            # get low frequency line locations (ny) and mask them out
            squeezed_mask = mask[:, 0, :, 0, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        # compute pad along H axis (shape[-3])
        pad = (mask.shape[-3] - num_low_frequencies_tensor + 1) // 2
        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def circular_centered_mask(self, shape, radius):
        '''
        generate a circular mask centered at the center of the image. For calgary-campinas dataset
        -shape: the shape of the mask
        -radius: the radius of the circle (ACS region)

        '''
        # radius is a tensor or int
        if type(radius) == torch.Tensor:
            # radius[0].item() # assume batch have the same radius
            radius = int(radius[0])

        center = torch.tensor(shape) // 2
        Y, X = torch.meshgrid(torch.arange(
            shape[0]), torch.arange(shape[1]), indexing='ij')
        dist_from_center = torch.sqrt(
            (X - center[1]) ** 2 + (Y - center[0]) ** 2)
        mask = (dist_from_center <= radius).float()
        return mask.unsqueeze(0).unsqueeze(-1)

    def __call__(self, masked_kspace: torch.Tensor,
                 mask: torch.Tensor,
                 num_low_frequencies: Optional[int] = None,
                 mask_type: Tuple[str] = ("cartesian",),
                 ) -> torch.Tensor:
        if self.mask_center:
            mask_type = mask_type[0] # assume the same type in a batch
            mask_type = 'cartesian' if mask_type in ['uniform', 'kt_uniform', 'kt_random'] else mask_type
            if mask_type == 'kt_radial':  # cmrxrecon24 pseudo radial
                mask_low = torch.zeros_like(mask)
                b, adj_nc, h, w, two = mask.shape
                h_left = h//2 - num_low_frequencies//2
                w_left = w//2 - num_low_frequencies//2
                mask_low[:, :, h_left:h_left+num_low_frequencies, w_left:w_left+num_low_frequencies, :] \
                    = mask[:, :, h_left:h_left+num_low_frequencies, w_left:w_left+num_low_frequencies, :]
                masked_kspace_acs = masked_kspace*mask_low
            elif mask_type  == 'cartesian': # fastmri and cmrxrecon (exclude kt_radial)
                pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                    mask, num_low_frequencies
                )
                masked_kspace_acs = transforms.batched_mask_center(
                    masked_kspace, pad, pad + num_low_freqs
                )
            elif mask_type == 'poisson_disc': # cc-brain
                ss = masked_kspace.shape[-3:-1]  # (h,w)
                # * cache low mask in dict to avoid repeated calculation for the same input shape.
                if ss not in self.low_mask_dict:
                    mask_low = self.circular_centered_mask(masked_kspace.shape[-3:-1], num_low_frequencies)  # shape (1, 218, 180, 1)
                    mask_low = mask_low[None].to(masked_kspace.device)
                    self.low_mask_dict[ss] = mask_low
                else:
                    mask_low = self.low_mask_dict[ss]
                masked_kspace_acs = masked_kspace * mask_low
            else:
                raise ValueError('mask_type should be cartesian or poisson_disc')
            return masked_kspace_acs
        else:
            return masked_kspace


##########################################################################
# -------- Conv Layer -----------------------
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.InstanceNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

# ##########################################################################
# # -------- Patch Embed -----------------------

# class PatchEmbed(nn.Module):
#     r""" Stem

#     Args:
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#     """

#     def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
#         super().__init__()
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim


#         self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
#         self.conv2 = nn.Sequential(
#             ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
#             ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
#         )
#         self.conv3 = nn.Sequential(
#             ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
#             ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
#         )


#     def forward(self, x):
#         """
#         # x: [B, C, H, W]
#         """
#         x = self.conv1(x)
#         x = self.conv2(x) + x
#         x = self.conv3(x)
#         return x
    

# ##########################################################################
# # -------- Patch Merge -----------------------
# class PatchMerge(nn.Module):
#     r""" Patch Merging Layer.

#     Args:
#         dim (int): Number of input channels.
#     """

#     def __init__(self, dim, ratio=4.0):
#         super().__init__()
#         self.dim = dim
#         in_channels = dim
#         out_channels = 2 * dim
#         self.conv = nn.Sequential(
#             ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
#             ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
#             ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
#         )

#     def forward(self, x):
#         """
#         # x: [B, C, H, W]
#         """
#         x = self.conv(x)
#         return x
    
# ##########################################################################
# # -------- Patch Expand -----------------------

# class PatchExpand(nn.Module):
#     """
#     Inverse of PatchMerge: upsamples spatial dims by dim_scale and halves channels.
#     Args:
#         dim (int): number of input channels
#         dim_scale (int): upsampling factor per spatial dimension
#     """
#     def __init__(self, dim, dim_scale=2, ratio=4.0, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.dim = dim
#         in_channels = dim
#         out_channels = dim // 2  # 因为我们要扩展空间维度，所以通道数减半
#         self.norm = norm_layer(out_channels)
#         self.conv = nn.Sequential(
#             ConvLayer(in_channels, int(in_channels * ratio), kernel_size=1, norm=None),
#             nn.ConvTranspose2d(int(in_channels * ratio), int(in_channels * ratio), kernel_size=3, stride=2, padding=1,
#                                output_padding=1, groups=int(in_channels * ratio), bias=False),
#             ConvLayer(int(in_channels * ratio), out_channels, kernel_size=1, act_func=None)
#         )


#     def forward(self, x):
#         """
#         # x: [B, C, H, W]
#         """
#         x = self.conv(x)
#         x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         return x


# ##########################################################################
# # -------- Final Projection -----------------------
# class FinalProjection(nn.Module):
#     """
#     Inverse of PatchEmbed: upsamples spatial dims by dim_scale and
#     projects to exactly out_chans real channels.
#     Args:
#         in_dim (int): number of input feature channels
#         out_chans (int): desired number of output (real) channels
#         dim_scale (int): upsampling factor per spatial dimension
#         norm_layer (nn.Module): normalization over the last channel axis
#     """
#     def __init__(self, in_dim: int, out_chans: int, dim_scale: int = 4, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim_scale = dim_scale
#         self.out_chans = out_chans
#         # project from in_dim → (dim_scale^2 * out_chans)
#         self.expand = nn.Linear(in_dim, (dim_scale ** 2) * out_chans, bias=False)
#         # normalize over the final out_chans axis
#         self.norm = norm_layer(out_chans)

#     def forward(self, x):
#         # x: [B, in_dim, H, W]
#         # → [B, H, W, in_dim] so we can do a point-wise linear
#         x = x.permute(0, 2, 3, 1).contiguous()  

#         # → [B, H, W, dim_scale^2 * out_chans]
#         x = self.expand(x)

#         # split that last axis into (p1, p2, out_chans)
#         # which upsamples H,W by p1,p2 and leaves out_chans channels
#         x = rearrange(
#             x,
#             'b h w (p1 p2 c) -> b (h p1) (w p2) c',
#             p1=self.dim_scale,
#             p2=self.dim_scale,
#             c=self.out_chans
#         )  # → [B, H*dim_scale, W*dim_scale, out_chans]

#         # normalize over the channel axis
#         x = self.norm(x)

#         # → [B, out_chans, H*dim_scale, W*dim_scale]
#         return x.permute(0, 3, 1, 2).contiguous()


##########################################################################
# -------- Patch Embed -----------------------

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): size of each patch (patch_size x patch_size)
        in_chans (int): number of input channels
        embed_dim (int): number of output embedding channels
        norm_layer (nn.Module, optional): normalization layer applied to embeddings
    """
    def __init__(self, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        # project non-overlapping patches to embed_dim channels
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        # optional normalization over embedding dimension
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        # x: [B, in_chans, H, W]
        x = self.proj(x)                  # → [B, embed_dim, H/patch_size, W/patch_size]
        if self.norm:
            x = x.permute(0, 2, 3, 1).contiguous()  # → [B, H/ps, W/ps, embed_dim]
            x = self.norm(x)             # normalize along embedding dim
            x = x.permute(0, 3, 1, 2).contiguous()  # → [B, embed_dim, H/ps, W/ps]
        return x                         # [B, embed_dim, H/ps, W/ps]


##########################################################################
# -------- Patch Merge -----------------------

class PatchMerge(nn.Module):
    r""" Patch Merging Layer: downsample spatial dims by 2 and increase channels by 2x.
    Args:
        dim (int): number of input channels
        norm_layer (nn.Module, optional): normalization layer applied before reduction
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # combine 2x2 neighbors: 4*dim channels → reduce to 2*dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        # x: [B, C, H, W]
        """
        x = x.permute(0, 2, 3, 1).contiguous()  # → [B, H, W, C]
        B, H, W, C = x.shape

        # split into 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]

        # concatenate channel-wise: [B, H/2, W/2, 4*C]
        x = torch.cat([x0, x1, x2, x3], dim=-1)

        x = self.norm(x)             # normalize 4*C channels
        x = self.reduction(x)        # project 4*C → 2*C
        x = x.permute(0, 3, 1, 2).contiguous()  # → [B, 2*C, H/2, W/2]

        return x                     # [B, 2*C, H/2, W/2]


##########################################################################
# -------- Patch Expand -----------------------

class PatchExpand(nn.Module):
    """
    Inverse of PatchMerge: upsamples spatial dims by dim_scale and halves channels.
    Args:
        dim (int): number of input channels
        dim_scale (int): upsampling factor per spatial dimension
        norm_layer (nn.Module, optional): normalization layer applied after rearrange
    """
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_scale = dim_scale
        self.dim = dim
        # project C → (dim_scale^2 * C)
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False)
        # normalize over output channels after halving
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert C == self.dim, f"Expected {self.dim} channels, got {C}"

        # → [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()

        # → [B, H, W, C * dim_scale]
        x = self.expand(x)

        # rearrange: split last axis into (dim_scale, dim_scale, C/dim_scale)
        # and upscale spatial dims accordingly
        x = rearrange(
            x,
            'b h w (p1 p2 c) -> b (h p1) (w p2) c',
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=self.dim // self.dim_scale
        )  # → [B, H*dim_scale, W*dim_scale, C/ dim_scale]

        x = self.norm(x)  # normalize C/dim_scale channels

        # → [B, C/ dim_scale, H*dim_scale, W*dim_scale]
        return x.permute(0, 3, 1, 2).contiguous()


##########################################################################
# -------- Final Projection -----------------------

class FinalProjection(nn.Module):
    """
    Inverse of PatchEmbed: upsamples spatial dims by dim_scale and
    projects to exactly out_chans real channels.
    Args:
        in_dim (int): number of input feature channels
        out_chans (int): desired number of output (real) channels
        dim_scale (int): upsampling factor per spatial dimension
        norm_layer (nn.Module): normalization over the last channel axis
    """
    def __init__(self, in_dim: int, out_chans: int, dim_scale: int = 4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_scale = dim_scale
        self.out_chans = out_chans
        # project from in_dim → (dim_scale^2 * out_chans)
        self.expand = nn.Linear(in_dim, (dim_scale ** 2) * out_chans, bias=False)
        # normalize over the final out_chans axis
        self.norm = norm_layer(out_chans)

    def forward(self, x):
        # x: [B, in_dim, H, W]
        # → [B, H, W, in_dim] so we can do a point-wise linear
        x = x.permute(0, 2, 3, 1).contiguous()  

        # → [B, H, W, dim_scale^2 * out_chans]
        x = self.expand(x)

        # split that last axis into (p1, p2, out_chans)
        # which upsamples H,W by p1,p2 and leaves out_chans channels
        x = rearrange(
            x,
            'b h w (p1 p2 c) -> b (h p1) (w p2) c',
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=self.out_chans
        )  # → [B, H*dim_scale, W*dim_scale, out_chans]

        # normalize over the channel axis
        x = self.norm(x)

        # → [B, out_chans, H*dim_scale, W*dim_scale]
        return x.permute(0, 3, 1, 2).contiguous()