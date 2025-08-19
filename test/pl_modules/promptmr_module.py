import torch
from data import transforms
from pl_modules import MriModule
from typing import List
import copy 
from mri_utils import SSIMLoss
import torch.nn.functional as F
import importlib

def get_model_class(module_name, class_name="PromptMR"):
    """
    Dynamically imports the specified module and retrieves the class.

    Args:
        module_name (str): The module to import (e.g., 'model.m1', 'model.m2').
        class_name (str): The class to retrieve from the module (default: 'PromptMR').

    Returns:
        type: The imported class.
    """
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

class PromptMrModule(MriModule):

    def __init__(
        self,
        num_cascades:           int = 12,
        num_adj_slices:         int = 3,
        patch_size:             int = 4,
        n_feat0:                int = 48,
        d_state:                int = 64,
        num_heads:              List[int] = [2,4,8,16],
        feature_dim:            List[int] = [128,256,512],
        prompt_dim:             List[int] = [32,64,128],
        sens_n_feat0:           int = 24,
        sens_feature_dim:       List[int] = [64,128,256],
        sens_prompt_dim:        List[int] = [16,32,64],
        len_prompt:             List[int] = [5,5,5],
        prompt_size:            List[int] = [64,32,16],
        n_enc_cab:              List[int] = [2,3,3],
        n_dec_cab:              List[int] = [2,2,3],
        n_skip_cab:             List[int] = [1,1,1],
        n_bottleneck_cab:       int = 3,
        no_use_ca:              bool = False,
        learnable_prompt:       bool = False,
        adaptive_input:         bool = False,
        n_buffer:               int = 0,
        n_history:              int = 0,
        use_sens_adj:           bool = True,
        dropout:                float = 0.,
        model_version:          str = "vssd_recon",
        lr:                     float = 0.0002,
        lr_step_size:           int = 11,
        lr_gamma:               float = 0.1,
        weight_decay:           float = 0.01,
        use_checkpoint:         bool = False,
        compute_sens_per_coil:  bool = False,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational network.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            d_state: TODO
            num_heads: TODO
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            sens_n_feat0: Number of top-level feature channels for sense map
                estimation PromptUnet in PromptMR.
            sens_feature_dim: feature dim for each level in PromptUnet for
                sensitivity map estimation (SME) network.
            sens_prompt_dim: prompt dim for each level in PromptUnet in
                sensitivity map estimation (SME) network.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in BottleneckBlock.
            no_use_ca: not using channel attention.
            learnable_prompt: whether to set the prompt as learnable parameters.
            adaptive_input: whether to use adaptive input.
            n_buffer: number of buffer in adaptive input.
            n_history: number of historical feature aggregation, should be less than num_cascades.
            dropout: TODO
            use_sens_adj: whether to use adjacent sensitivity map estimation.
            model_version: model version. Default is "promptmr_v2".
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            use_checkpoint: Whether to use checkpointing to trade compute for GPU memory.
            compute_sens_per_coil: (bool) whether to compute sensitivity maps per coil for memory saving
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices

        self.patch_size = patch_size
        self.n_feat0 = n_feat0
        self.d_state = d_state
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim

        self.sens_n_feat0 = sens_n_feat0
        self.sens_feature_dim = sens_feature_dim
        self.sens_prompt_dim = sens_prompt_dim

        self.len_prompt = len_prompt
        self.prompt_size = prompt_size
        self.n_enc_cab = n_enc_cab
        self.n_dec_cab = n_dec_cab
        self.n_skip_cab = n_skip_cab
        self.n_bottleneck_cab = n_bottleneck_cab

        self.no_use_ca = no_use_ca

        self.learnable_prompt = learnable_prompt
        self.adaptive_input = adaptive_input
        self.n_buffer = n_buffer
        self.n_history = n_history
        self.use_sens_adj = use_sens_adj
        self.dropout = dropout
        # two flags for reducing memory usage
        self.use_checkpoint = use_checkpoint
        self.compute_sens_per_coil = compute_sens_per_coil
        
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.model_version = model_version
        PromptMR = get_model_class(f"models.{model_version}")  # Dynamically get the model class
        
        self.promptmr = PromptMR(
            num_cascades=self.num_cascades,
            num_adj_slices=self.num_adj_slices,
            patch_size=self.patch_size,
            n_feat0=self.n_feat0,
            d_state=self.d_state,
            num_heads=self.num_heads,
            feature_dim=self.feature_dim,
            prompt_dim=self.prompt_dim,
            sens_n_feat0=self.sens_n_feat0,
            sens_feature_dim=self.sens_feature_dim,
            sens_prompt_dim=self.sens_prompt_dim,
            len_prompt=self.len_prompt,
            prompt_size=self.prompt_size,
            n_enc_cab=self.n_enc_cab,
            n_dec_cab=self.n_dec_cab,
            n_skip_cab=self.n_skip_cab,
            n_bottleneck_cab=self.n_bottleneck_cab,
            no_use_ca=self.no_use_ca,
            learnable_prompt=self.learnable_prompt,
            n_history=self.n_history,
            n_buffer=self.n_buffer,
            adaptive_input=self.adaptive_input,
            use_sens_adj=self.use_sens_adj,
            dropout=self.dropout,
            **kwargs
        )

        self.loss = SSIMLoss()

    def configure_optimizers(self):

        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # step lr scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )
        return [optim], [scheduler]
    
    # def on_fit_start(self):
    #     if self.global_rank == 0:
    #         try:
    #             flops = self.promptmr.flops()
    #             self.print(f"Estimated GFLOPs: {flops / 1e9:.2f}")
    #         except Exception as e:
    #             self.print(f"FLOP counting failed: {e}")

    def forward(self, masked_kspace, mask, num_low_frequencies, mask_type, use_checkpoint=False, compute_sens_per_coil=False):
        # mask_type = ["uniform", "kt_radial", "kt_gaussian"][mask_type]

        return self.promptmr(masked_kspace, mask, num_low_frequencies, mask_type, use_checkpoint=use_checkpoint, compute_sens_per_coil=compute_sens_per_coil)   


    def training_step(self, batch, batch_idx):
        # mask_type = int(batch.mask_type.item())
        # max_value = float(batch.max_value.item())

        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type, 
                           use_checkpoint=self.use_checkpoint, compute_sens_per_coil=self.compute_sens_per_coil)
        output = output_dict['img_pred']
        target, output = transforms.center_crop_to_smallest(
            batch.target, output)

        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )

        self.log("train_loss", loss.detach().cpu().item(), prog_bar=True)

        ##! raise error if loss is nan
        if torch.isnan(loss):
            raise ValueError(f'nan loss on {batch.fname} of slice {batch.slice_num}')
        return loss


    def validation_step(self, batch, batch_idx):
        # mask_type = int(batch.mask_type.item())
        # max_value     = float(batch.max_value.item())
        # slice_num     = int(batch.slice_num.item())
        # fname         = int(batch.fname.item())
            
        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil)

        output = output_dict['img_pred']
        img_zf = output_dict['img_zf']
        target, output = transforms.center_crop_to_smallest(
            batch.target, output)
        _, img_zf = transforms.center_crop_to_smallest(
            batch.target, img_zf)
        val_loss = self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            )
        cc = batch.masked_kspace.shape[1]
        centered_coil_visual = torch.log(1e-10+torch.view_as_complex(batch.masked_kspace[:,cc//2]).abs())
            
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "img_zf":   img_zf,
            "mask": centered_coil_visual, 
            "sens_maps": output_dict['sens_maps'][:,0].abs(),
            "output": output,
            "target": target,
            "loss": val_loss,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # mask_type   = int(batch.mask_type.item())
        # fname           = int(batch.fname.item())
        # slice_num       = int(batch.slice_num.item())
        # num_slc         = int(batch.num_slc.item())

        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil)
        
        # let the submission script handle this part
        output = output_dict['img_pred']

        crop_size = batch.crop_size 
        crop_size = [crop_size[0][0], crop_size[1][0]] # if batch_size>1
        # detect FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        output = transforms.center_crop(output, crop_size)

        num_slc = batch.num_slc
        return {
            'output': output.cpu(), 
            'slice_num': batch.slice_num, 
            'fname': batch.fname,
            'num_slc':  num_slc
        }
        