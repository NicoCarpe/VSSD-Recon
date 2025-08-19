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

from data import transforms

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

