"""
This file contains one implementation of the PromptMR+ model
"""
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from mri_utils import ifft2c, rss, complex_abs, rss_complex, sens_expand, sens_reduce
from .utils_mamba import KspaceACSExtractor, DownBlock, UpBlock, SkipBlock, PromptBlock, PatchEmbed, FinalProjection
from .VSSBlock import VSSBlock


class PromptUnet(nn.Module): 
    def __init__(self,
                 in_chans: int,
                 out_chans: int,
                 n_feat0: int,
                 feature_dim: List[int],
                 prompt_dim: List[int],
                 len_prompt: List[int],
                 prompt_size: List[int],
                 n_enc_cab: List[int],
                 n_dec_cab: List[int],
                 n_skip_cab: List[int],
                 n_bottleneck_cab: int,
                 bias=False,
                 learnable_prompt=False,
                 adaptive_input=False,
                 d_state=16,
                 dropout=0,
                 n_buffer=0,
                 n_history=0,
                 ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_history = n_history
        self.n_buffer = n_buffer if adaptive_input else 0
        
        in_chans = in_chans * (1+self.n_buffer) if adaptive_input else in_chans 
        out_chans = out_chans * (1+self.n_buffer) if adaptive_input else in_chans 
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(patch_size=4, in_chans=in_chans, embed_dim=n_feat0)
        
        # Encoder - 3 DownBlocks
        self.enc_level1 = DownBlock(n_feat0, d_state, n_enc_cab[0], bias, dropout)
        self.enc_level2 = DownBlock(feature_dim[0], d_state, n_enc_cab[1], bias, dropout)
        self.enc_level3 = DownBlock(feature_dim[1], d_state, n_enc_cab[2],  bias, dropout)

        # Skip Connections - 3 SkipBlocks
        self.skip_attn1 = SkipBlock(n_feat0, d_state, n_skip_cab[0], bias, dropout)
        self.skip_attn2 = SkipBlock(feature_dim[0], d_state, n_skip_cab[1], bias, dropout)
        self.skip_attn3 = SkipBlock(feature_dim[1], d_state, n_skip_cab[2], bias, dropout)

        # Bottleneck 
        self.bottleneck = nn.Sequential(*[
            VSSBlock(
                hidden_dim = feature_dim[2],
                d_state = d_state,
                drop_path = dropout,
                bias = bias
            ) for _ in range(n_bottleneck_cab)
        ])

        # Decoder - 3 UpBlocks
        self.prompt_level3 = PromptBlock(prompt_dim[2], len_prompt[2], prompt_size[2], feature_dim[2], learnable_prompt)
        self.dec_level3 = UpBlock(feature_dim[2], d_state, prompt_dim[2], n_dec_cab[2], bias, dropout, n_history)

        self.prompt_level2 = PromptBlock(prompt_dim[1], len_prompt[1], prompt_size[1], feature_dim[1], learnable_prompt)
        self.dec_level2 = UpBlock(feature_dim[1], d_state, prompt_dim[1], n_dec_cab[1], bias, dropout, n_history)

        self.prompt_level1 = PromptBlock(prompt_dim[0], len_prompt[0], prompt_size[0], feature_dim[0], learnable_prompt)
        self.dec_level1 = UpBlock(feature_dim[0], d_state, prompt_dim[0], n_dec_cab[0], bias, dropout, n_history)

        # OutConv
        self.final_proj = FinalProjection(n_feat0, out_chans)

    def forward(self, x: torch.Tensor, history_feat: Optional[List[torch.Tensor]] = None):
        """
        Forward pass of PromptUnet.

        Args:
            x: (B, C, H, W) tensor after patch embedding via PatchEmbed
            history_feat: list of 3 tensors or None for temporal context, each of shape [(B, D_i, H_i, W_i)]

        Returns:
            tuple:
                - out: (B, C_out, H, W) final projection output from FinalProjection
                - history_feat: updated list of history features for next cascade
        """
        if history_feat is None:
            history_feat = [None, None, None]

        history_feat3, history_feat2 , history_feat1 = history_feat
        current_feat = []
        
        # 0. patch embedding: x_in (B, C_in, H, W) -> x_embed (B, n_feat0, H', W')
        x = self.patch_embed(x)

        # 1. encoder
        # enc_level1: x_embed -> x1_down (B, D1, H'/2, W'/2), enc1 skip (B, D0, H', W')
        x, enc1 = self.enc_level1(x)
        # enc_level2: x1_down -> x2_down (B, D2, H'/4, W'/4), enc2 skip (B, D1, H'/2, W'/2)
        x, enc2 = self.enc_level2(x)
        # enc_level3: x2_down -> x3_down (B, D3, H'/8, W'/8), enc3 skip (B, D2, H'/4, W'/4)
        x, enc3 = self.enc_level3(x)

        # 2. bottleneck: x3_down -> same shape (B, D2, H'/8, W'/8)
        x = self.bottleneck(x)

        # 3. decoder
        current_feat.append(x.clone())
        dec_prompt3 = self.prompt_level3(x)  # (B, prompt_dim[2], H'/8, W'/8)
        x = self.dec_level3(x, dec_prompt3, self.skip_attn3(enc3), history_feat3)
        # x out: (B, D2, H'/4, W'/4)

        current_feat.append(x.clone())
        dec_prompt2 = self.prompt_level2(x)  # (B, prompt_dim[1], H'/4, W'/4)
        x = self.dec_level2(x, dec_prompt2, self.skip_attn2(enc2), history_feat2)
        # x out: (B, D1, H'/2, W'/2)

        current_feat.append(x.clone())
        dec_prompt1 = self.prompt_level1(x)  # (B, prompt_dim[0], H'/2, W'/2)
        x = self.dec_level1(x, dec_prompt1, self.skip_attn1(enc1), history_feat1)
        # x out: (B, n_feat0, H', W')

        # 4. final projection
        if self.n_history > 0:
            # update history_feat shapes accordingly
            # history_feat[i]: (B, D_i * n_history, H_i, W_i)
            for i, history_feat_i in enumerate(history_feat):
                if history_feat_i is None:
                    history_feat[i] = torch.tile(current_feat[i], (1, self.n_history, 1, 1))
                else:
                    dim = self.feature_dim[2-i]
                    history_feat[i] = torch.cat([current_feat[i], history_feat_i[:, :-dim]], dim=1)

        out = self.final_proj(x)

        return out, history_feat


class NormPromptUnet(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        learnable_prompt=False,
        adaptive_input=False,
        n_buffer=0,
        n_history=0,
    ):

        super().__init__()
        self.n_history = n_history
        self.n_buffer = n_buffer
        self.unet = PromptUnet(in_chans=in_chans,
                               out_chans=out_chans,
                               n_feat0=n_feat0,
                               feature_dim=feature_dim,
                               prompt_dim=prompt_dim,
                               len_prompt=len_prompt,
                               prompt_size=prompt_size,
                               n_enc_cab=n_enc_cab,
                               n_dec_cab=n_dec_cab,
                               n_skip_cab=n_skip_cab,
                               n_bottleneck_cab=n_bottleneck_cab,
                               learnable_prompt = learnable_prompt,
                               adaptive_input=adaptive_input,
                               n_buffer = n_buffer,
                               n_history= n_history,
                               )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return rearrange(x, 'b c h w two -> b (two c) h w')

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        return rearrange(x, 'b (two c) h w -> b c h w two', two=2).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        
        # pad to multiple of patch_size * 2**n_downblocks = 4 * 2**3 = 32
        pad_to = 32
        w_mult = ((w - 1) | (pad_to - 1)) + 1
        h_mult = ((h - 1) | (pad_to - 1)) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor,
              h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor,
                history_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                buffer: torch.Tensor = None):
        """
        Forward pass of NormPromptUnet.

        Args:
            x: (B, C_coils, H, W, 2) complex k-space input
            history_feat: tuple of history features or None
            buffer: (B, n_buffer*C, H_small, W_small) adaptive buffer

        Returns:
            tuple:
                - x_out: (B, C_coils, H, W, 2) complex output
                - latent: (B, C_coils, H_small, W_small) optional latent feature
                - history_feat: updated history features
        """
        # Check complex last dim
        if x.shape[-1] != 2:
            raise ValueError("Last dimension must be 2 for complex.")
        cc = x.shape[1]
        # concatenate buffer if provided
        if buffer is not None:
            x = torch.cat([x, buffer], dim=1)

        # flatten complex to channel dim: (B, 2*C, H, W)
        x = self.complex_to_chan_dim(x)
        # normalize, pad, unet, unpad, unnorm back
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x, history_feat = self.unet(x, history_feat)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        # split latent and output
        if buffer is not None:
            x_out, _, latent, _ = torch.split(x, [cc, cc, cc, x.shape[1] - 3*cc], dim=1)
        else:
            x_out = x
            latent = None
        return x_out, latent, history_feat



class PromptMRBlock(nn.Module):

    def __init__(self, model: nn.Module, num_adj_slices=5):

        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        history_feat: Optional[Tuple[torch.Tensor, ...]] = None,
        buffer: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of one PromptMRBlock cascade.

        Args:
            current_kspace: (B, Nc, H, W, 2) current k-space estimate
            ref_kspace: (B, Nc, H, W, 2) reference k-space (masked input)
            mask: (B, 1, H, W, 1) sampling mask
            sens_maps: (B, Nc, H, W, 2) sensitivity maps
            history_feat: optional history features from previous cascades
            buffer: optional buffer image features

        Returns:
            tuple:
                - updated_kspace: (B, Nc, H, W, 2)
                - latent: latent feature or None
                - history_feat: updated history features
        """
        zero = torch.zeros(1,1,1,1,1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        pred, latent, history_feat = self.model(
            sens_reduce(current_kspace, sens_maps, self.num_adj_slices),
            history_feat, buffer)
        model_term = sens_expand(pred, sens_maps, self.num_adj_slices)
        updated = current_kspace - soft_dc - model_term
        return updated, latent, history_feat

class PromptMR(nn.Module):

    def __init__(
        self,
        num_cascades: int,
        num_adj_slices: int,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        sens_n_feat0: int,
        sens_feature_dim: List[int],
        sens_prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        no_use_ca: bool = False,    # left in for compatibility with promptmr pl-module logic
        sens_len_prompt: Optional[List[int]] = None,
        sens_prompt_size: Optional[List[int]] = None,
        sens_n_enc_cab: Optional[List[int]] = None,
        sens_n_dec_cab: Optional[List[int]] = None,
        sens_n_skip_cab: Optional[List[int]] = None,
        sens_n_bottleneck_cab: Optional[List[int]] = None,
        sens_no_use_ca: Optional[bool] = None,  # left in for compatibility with promptmr pl-module logic
        mask_center: bool = True,
        learnable_prompt: bool = False,
        adaptive_input: bool = False,
        n_buffer: int = 4,
        n_history: int = 0,
        use_sens_adj: bool = True,
    ):

        super().__init__()
        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices//2
        self.n_history = n_history
        self.n_buffer = n_buffer
        self.sens_net = SensitivityModel(
            num_adj_slices=num_adj_slices,
            n_feat0=sens_n_feat0,
            feature_dim=sens_feature_dim,
            prompt_dim=sens_prompt_dim,
            len_prompt=sens_len_prompt if sens_len_prompt is not None else len_prompt,
            prompt_size=sens_prompt_size if sens_prompt_size is not None else prompt_size,
            n_enc_cab=sens_n_enc_cab if sens_n_enc_cab is not None else n_enc_cab,
            n_dec_cab=sens_n_dec_cab if sens_n_dec_cab is not None else n_dec_cab,
            n_skip_cab=sens_n_skip_cab if sens_n_skip_cab is not None else n_skip_cab,
            n_bottleneck_cab=sens_n_bottleneck_cab if sens_n_bottleneck_cab is not None else n_bottleneck_cab,
            mask_center=mask_center,
            learnable_prompt = learnable_prompt,
            use_sens_adj = use_sens_adj
        )
        # DC + denoiser in each cascade
        self.cascades = nn.ModuleList([
            PromptMRBlock(
                NormPromptUnet(
                    in_chans=2 * num_adj_slices,
                    out_chans=2 * num_adj_slices,
                    n_feat0=n_feat0,
                    feature_dim=feature_dim,
                    prompt_dim=prompt_dim,
                    len_prompt=len_prompt,
                    prompt_size=prompt_size,
                    n_enc_cab=n_enc_cab,
                    n_dec_cab=n_dec_cab,
                    n_skip_cab=n_skip_cab,
                    n_bottleneck_cab=n_bottleneck_cab,
                    learnable_prompt=learnable_prompt,
                    adaptive_input=adaptive_input,
                    n_buffer = n_buffer,
                    n_history=n_history
                ),
                num_adj_slices=num_adj_slices
            ) for _ in range(num_cascades)
        ])

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: torch.Tensor,
        mask_type: Tuple[str] = ("cartesian",),
        use_checkpoint: bool = True,
        compute_sens_per_coil: bool = False,
    ) -> dict:
        """
        Full PromptMR forward: cascaded DC + U-Net reconstruction.

        Args:
            masked_kspace: (B, Nc, H, W, 2) input under-sampled k-space
            mask: (B, 1, H, W, 1) sampling mask
            num_low_frequencies: (B,) number of ACS lines
            mask_type: tuple of mask type strings
            use_checkpoint: bool flag for gradient checkpointing
            compute_sens_per_coil: bool flag to compute sens maps per coil

        Returns:
            dict with:
             - 'img_pred': (B, 1, H, W) reconstructed image
             - 'img_zf': (B, 1, H, W) zero-filled image
             - 'sens_maps': (B, H, W) complex sensitivity map
        """
        if use_checkpoint:  # and self.training:
            sens_maps = torch.utils.checkpoint.checkpoint(
                 self.sens_net, masked_kspace, mask, num_low_frequencies, mask_type, compute_sens_per_coil,
                use_reentrant=False)
        else:
            sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies, mask_type, compute_sens_per_coil)

        kspace_pred = masked_kspace.clone() # torch.Size([1, 60, 218, 170, 2])
        zero = torch.zeros(1, 1, 1, 1, 1).to(kspace_pred)
        img_zf = sens_reduce(kspace_pred, sens_maps, self.num_adj_slices)
        buffer = torch.cat([img_zf] * self.n_buffer, dim=1) if self.n_buffer > 0 else None
        history_feat = None
        
        for ith,cascade in enumerate(self.cascades):
            is_last = ith == self.num_cascades - 1
            if use_checkpoint and self.training:
                kspace_pred, latent, history_feat  = torch.utils.checkpoint.checkpoint(
                    cascade, kspace_pred, masked_kspace, mask, sens_maps, history_feat, buffer, use_reentrant=False)
            else:
                kspace_pred, latent, history_feat = cascade(kspace_pred, masked_kspace, mask, sens_maps, history_feat,buffer)

            if self.n_buffer>0 and not is_last:
                ffx =  sens_reduce( torch.where(mask, kspace_pred, zero), sens_maps, self.num_adj_slices)
                # adaptive input. buffer: A^H*A*x_i, s_i, x0, A^H*A*x_i-x0
                buffer = torch.cat([ffx, latent, img_zf]+[ffx-img_zf]*(self.n_buffer-3), dim=1)
                
        # get central slice of rss as final output
        kspace_pred = torch.chunk(kspace_pred, self.num_adj_slices, dim=1)[self.center_slice]
        img_pred = rss(complex_abs(ifft2c(kspace_pred)), dim=1)
        
        # prepare for additional output
        img_zf = torch.chunk(masked_kspace, self.num_adj_slices, dim=1)[self.center_slice]
        img_zf = rss(complex_abs(ifft2c(img_zf)), dim=1)
        sens_maps = torch.chunk(sens_maps, self.num_adj_slices, dim=1)[self.center_slice]
        sens_maps = torch.view_as_complex(sens_maps)

        return {
            'img_pred': img_pred,
            'img_zf': img_zf,
            'sens_maps': sens_maps
        }


class SensitivityModel(nn.Module):

    def __init__(
        self,
        num_adj_slices: int = 5,
        n_feat0: int = 24,
        feature_dim: List[int] = [36, 48, 60],
        prompt_dim: List[int] = [12, 24, 36],
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        mask_center: bool = True,
        learnable_prompt = False,
        use_sens_adj: bool = True,
    ):

        super().__init__()
        self.mask_center = mask_center
        self.num_adj_slices = num_adj_slices
        self.use_sens_adj = use_sens_adj
        self.norm_unet = NormPromptUnet(in_chans=2*self.num_adj_slices if use_sens_adj else 2,
                                        out_chans=2*self.num_adj_slices if use_sens_adj else 2,
                                        n_feat0=n_feat0,
                                        feature_dim=feature_dim,
                                        prompt_dim=prompt_dim,
                                        len_prompt=len_prompt,
                                        prompt_size=prompt_size,
                                        n_enc_cab=n_enc_cab,
                                        n_dec_cab=n_dec_cab,
                                        n_skip_cab=n_skip_cab,
                                        n_bottleneck_cab=n_bottleneck_cab,
                                        learnable_prompt = learnable_prompt,
                                        )
        self.kspace_acs_extractor = KspaceACSExtractor(mask_center)
        
            
    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        if self.use_sens_adj:
            x = rearrange(x, 'b (adj coil) h w comp -> (b coil) adj h w comp', adj=self.num_adj_slices)
        else:
            x = rearrange(x, 'b adj_coil h w comp -> (b adj_coil) 1 h w comp')
        return x, b



    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.use_sens_adj:
            x = rearrange(x, '(b coil) adj h w comp -> b (adj coil) h w comp', b=batch_size, adj=self.num_adj_slices)
        else:
            x = rearrange(x, '(b adj_coil) 1 h w comp -> b adj_coil h w comp', b=batch_size)

        return x


    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        
        b, adj_coil, h, w, two = x.shape
        coil = adj_coil//self.num_adj_slices
        x = x.view(b, self.num_adj_slices, coil, h, w, two)
        x = x / rss_complex(x, dim=2).unsqueeze(-1).unsqueeze(2)

        return x.view(b, adj_coil, h, w, two)


    def compute_sens(self, model:nn.Module, images: torch.Tensor, compute_per_coil: bool) -> torch.Tensor:
        bc = images.shape[0] # batch_size * n_coils
        if compute_per_coil:
            output = []
            for i in range(bc):
                output.append(model(images[i].unsqueeze(0))[0])
            output = torch.cat(output, dim=0)
        else:
            output = model(images)[0]
        return output
        
    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[Union[int, torch.Tensor]] = None,
        mask_type: Tuple[str] = ("cartesian",),
        compute_per_coil: bool = False,
    ) -> torch.Tensor:
        """
        Estimate coil sensitivity maps from k-space.

        Args:
            masked_kspace: (B, Nc, H, W, 2) under-sampled k-space
            mask: (B, 1, H, W) sampling mask
            num_low_frequencies: number of ACS lines or tensor
            mask_type: tuple of mask types
            compute_per_coil: bool to compute per-coil adaptively

        Returns:
            sens_maps: (B, Nc, H, W, 2) complex sensitivity maps
        """

        masked_kspace_acs = self.kspace_acs_extractor(masked_kspace, mask, num_low_frequencies, mask_type)
        # convert to image space
        images, batches = self.chans_to_batch_dim(ifft2c(masked_kspace_acs))

        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.compute_sens(self.norm_unet, images, compute_per_coil), batches)
        )



def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    import time
    import torch

    # --- config (match your YAML defaults) ---
    batch_size       = 1
    num_cascades     = 12
    num_adj_slices   = 5
    n_feat0          = 96
    feature_dim      = [192, 384, 768]
    prompt_dim       = [96, 192, 384]
    sens_n_feat0     = 48
    sens_feature_dim = [96, 192, 384]
    sens_prompt_dim  = [48, 96, 192]
    len_prompt       = [5, 5, 5]
    prompt_size      = [64, 32, 16]
    n_enc_cab        = [2, 2, 2]
    n_dec_cab        = [2, 2, 2]
    n_skip_cab       = [1, 1, 1]
    n_bottleneck_cab = 3
    learnable_prompt = False
    adaptive_input   = True
    n_buffer         = 4
    n_history        = 11
    use_sens_adj     = True
    height, width    = 512, 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # instantiate
    model = PromptMR(
        num_cascades=num_cascades,
        num_adj_slices=num_adj_slices,
        n_feat0=n_feat0,
        feature_dim=feature_dim,
        prompt_dim=prompt_dim,
        sens_n_feat0=sens_n_feat0,
        sens_feature_dim=sens_feature_dim,
        sens_prompt_dim=sens_prompt_dim,
        len_prompt=len_prompt,
        prompt_size=prompt_size,
        n_enc_cab=n_enc_cab,
        n_dec_cab=n_dec_cab,
        n_skip_cab=n_skip_cab,
        n_bottleneck_cab=n_bottleneck_cab,
        learnable_prompt=learnable_prompt,
        adaptive_input=adaptive_input,
        n_buffer=n_buffer,
        n_history=n_history,
        use_sens_adj=use_sens_adj
    ).to(device)

    # count params
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,} (~{total_params*4/1024**2:.2f} MB)")

    # dummy inputs
    nc = num_adj_slices * 10  # coil images
    dummy_kspace = torch.randn(batch_size, nc, height, width, 2, device=device)
    dummy_mask   = torch.ones(batch_size, 1, height, width, 1, dtype=torch.bool, device=device)
    dummy_nlf    = torch.tensor([height // 4] * batch_size, device=device)

    # warm-up
    print("Warming up…")
    for _ in range(5):
        _ = model(dummy_kspace, dummy_mask, dummy_nlf)

    # reset & measure
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iters = 20
    torch.cuda.synchronize(device) if device.type=="cuda" else None
    t0 = time.time()
    for _ in range(iters):
        _ = model(dummy_kspace, dummy_mask, dummy_nlf)
    torch.cuda.synchronize(device) if device.type=="cuda" else None
    t1 = time.time()

    print(f"Ran {iters} iters in {t1-t0:.2f}s → {iters/(t1-t0):.1f} it/s")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)/1024**2
        print(f"Peak GPU memory: {peak:.1f} MB")


if __name__ == "__main__":
    main()
