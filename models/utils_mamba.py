"""
This file contains the basic modules for the model. 
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from data import transforms
from .MambaBlock2D import MambaBlock2D


##########################################################################
# ---------- Prompt Block -----------------------

class PromptBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, learnable_prompt=False):
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size), 
                                         requires_grad=learnable_prompt)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = torch.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.dec_conv3x3(prompt)

        return prompt


##########################################################################
# ---------- Down Block -----------------------

class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_block, kernel_size, bias, act, first_act=True):
        super().__init__()
        if first_act:
            self.encoder = [
                MambaBlock2D(
                    d_model=in_dim,
                    d_conv=kernel_size,
                    activation=nn.PReLU(),
                    bias=bias
                )]
            self.encoder = nn.Sequential(*[
                MambaBlock2D(
                    d_model=in_dim,
                    d_conv=kernel_size,
                    activation=act,
                    bias=bias
                ) for _ in range(n_block-1)
            ])
        else:
            self.encoder = nn.Sequential(*[
                MambaBlock2D(
                    d_model=in_dim,
                    d_conv=kernel_size,
                    activation=act,
                    bias=bias
                ) for _ in range(n_block)
            ])
        
        self.down = PatchMerge(dim=in_dim, norm_layer=nn.LayerNorm)

    def forward(self, x):
        enc = self.encoder(x)  # Shape: (B, C, H, W)
        # Permute to channel-last for PatchMerge
        enc_permuted = enc.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_down = self.down(enc_permuted)  # (B, H//2, W//2, 2*C)
        # Permute back to channel-first
        x_down = x_down.permute(0, 3, 1, 2)  # (B, 2*C, H//2, W//2)
        return x_down, enc  # enc is (B, C, H, W) for skip connection


##########################################################################
# ---------- Up Block -----------------------

class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, n_block, kernel_size, bias, act, n_history=0):
        super().__init__()
        # momentum layer
        self.n_history = n_history
        if n_history > 0:
            self.momentum = nn.Sequential(
                nn.Conv2d(in_dim*(n_history+1), in_dim, kernel_size=1, bias=bias),
                MambaBlock2D(
                    d_model=in_dim,
                    d_conv=kernel_size,
                    activation=act,
                    bias=bias
                )
            )

        self.fuse = nn.Sequential(*[
            MambaBlock2D(
                d_model=in_dim+prompt_dim,
                d_conv=kernel_size,
                activation=act,
                bias=bias
            ) for _ in range(n_block)
        ])
        self.reduce = nn.Conv2d(in_dim+prompt_dim, in_dim, kernel_size=1, bias=bias)

        self.up = PatchExpand(dim=in_dim, dim_scale=2, norm_layer=nn.LayerNorm)

        # why this one
        self.ca = MambaBlock2D(
                d_model=out_dim,
                d_conv=kernel_size,
                activation=act,
                bias=bias
            )

    def forward(self, x, prompt_dec, skip, history_feat: Optional[torch.Tensor] = None):
        # momentum layer
        if self.n_history > 0:
            if history_feat is None:
                x = torch.cat([torch.tile(x, (1, self.n_history+1, 1, 1))], dim=1)
            else:
                x = torch.cat([x, history_feat], dim=1)

            x = self.momentum(x)

        x = torch.cat([x, prompt_dec], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)

        # Permute for PatchExpand (B, C, H, W) -> (B, H, W, C)
        x_permuted = x.permute(0, 2, 3, 1)
        x_up = self.up(x_permuted)  # (B, 2H, 2W, C_out)
        # Permute back to (B, C_out, 2H, 2W)
        x_up = x_up.permute(0, 3, 1, 2)
        x = x_up + skip  # Ensure skip has shape (B, C_out, 2H, 2W)
        x = self.ca(x)

        return x


##########################################################################
# ---------- Skip Block -----------------------

class SkipBlock(nn.Module):
    def __init__(self, enc_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        super().__init__()
        if n_cab == 0:
            self.skip_attn = nn.Identity()
        else:
            self.skip_attn = nn.Sequential(*[
                MambaBlock2D(
                    d_model=enc_dim,
                    d_conv=kernel_size,
                    activation=act,
                    bias=bias
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
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
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

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2
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
# -------- Patch Embed -----------------------

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    # TODO: in_chans and embedding dim might not be correct here
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        # Projects each non‑overlapping patch into an embed_dim vector
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        # Optional normalization (e.g. LayerNorm) on the embedding dimension
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        # x: [B, in_chans, H, W]
        x = self.proj(x)                  # → [B, embed_dim, H/ps, W/ps]
        x = x.permute(0, 2, 3, 1)         # → [B, H/ps, W/ps, embed_dim]
        if self.norm:
            x = self.norm(x)              # normalize along embed_dim
        return x
    

##########################################################################
# -------- Patch Merge -----------------------

class PatchMerge(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # After concatenating 4 neighbor patches, reduce from 4*dim → 2*dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        # split into four sub‑grids (2×2)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        # cat along the channel dimension → [B, H/2, W/2, 4*C]
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)        # normalize 4*C
        x = self.reduction(x)   # project 4*C → 2*C

        return x                # [B, H/2, W/2, 2*C]
    

##########################################################################
# -------- Patch Expand -----------------------

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        # When upsampling, we first increase channels from dim → dim_scale*dim
        self.expand    = nn.Linear(dim * dim_scale, dim_scale * dim, bias=False)
        self.norm      = norm_layer(dim)

    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        # project C → dim_scale * C
        x = self.expand(x)           # → [B, H, W, p1*p2*C] where p1=p2=dim_scale

        # rearrange so that channels become extra spatial dims:
        #   from [B, H, W, (p1*p2*C)] → [B, H*p1, W*p2, C]
        x = rearrange(x,
                      'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                      p1=self.norm.normalized_shape[0]//C if False else self.dim_scale,
                      p2=self.dim_scale,
                      c=C)
        x = self.norm(x)             # normalize over C
        return x                     # [B, H*2, W*2, C]
    

##########################################################################
# -------- Final Projection -----------------------

class FinalProjection(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # Similar to PatchExpand2D but with a larger scale (e.g. 4×)
        self.expand    = nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm      = norm_layer(dim)
        self.dim_scale = dim_scale
        self.dim       = dim

    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x = self.expand(x)            # [B, H, W, scale² * C]
        # pixel‑shuffle to [B, H*scale, W*scale, C]
        x = rearrange(x,
                      'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                      p1=self.dim_scale,
                      p2=self.dim_scale,
                      c=C)
        x = self.norm(x)              # normalize over C
        return x                      # [B, H*4, W*4, C]