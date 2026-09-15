"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import gc
import pathlib
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import lightning as L
import torch
from torchmetrics.metric                import Metric
from torchmetrics.functional.regression import mean_squared_error
from torchmetrics.functional.image      import structural_similarity_index_measure
import matplotlib.pyplot as plt
from io import BytesIO

from mri_utils import utils, save_reconstructions

class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(L.LightningModule):
    """
    Abstract super class for deep learning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 16):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.val_log_indices = None
        # self.training_step_outputs = []
        self.validation_step_outputs = []
        self.erf_accumulator = []  

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()
        

    def on_train_epoch_end(self):
        pass

    def on_validation_batch_end(self, val_logs, batch, batch_idx, dataloader_idx=0):
        # --- basic key checks ---
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            'img_zf',
            "mask",
            'sens_maps',
            "output",
            "target",
            "loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(f"Expected key {k} in dict returned by validation_step.")

        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        if self.val_log_indices is None:
            limit_val_batches = self.trainer.limit_val_batches
            if isinstance(limit_val_batches, float) and limit_val_batches <= 1.0:
                num_val_batches = int(limit_val_batches * len(self.trainer.val_dataloaders))
            else:
                num_val_batches = int(limit_val_batches)
            self.val_log_indices = list(np.random.permutation(num_val_batches)[:self.num_log_images])

        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]

        for i, batch_idx in enumerate(batch_indices):
            if batch_idx not in self.val_log_indices:
                continue

            key = f"val_images_idx_{batch_idx}"
            mask = val_logs["mask"][i].unsqueeze(0)
            target = val_logs["target"][i].unsqueeze(0)
            output = val_logs["output"][i].unsqueeze(0)
            img_zf = val_logs["img_zf"][i].unsqueeze(0)
            sens_maps = val_logs["sens_maps"][i].unsqueeze(0)
            error = torch.abs(target - output)

            # simple normalizations for visualization
            img_zf = img_zf / img_zf.max()
            sens_maps = sens_maps / sens_maps.max()
            output = output / output.max()
            target = target / target.max()
            error = error / error.max()

            cm = plt.get_cmap('mako')
            error_np = error.squeeze().detach().cpu().numpy()
            error_colored = cm(error_np)[:, :, :3]
            error_chw = torch.from_numpy(error_colored).permute(2, 0, 1).float()

            cpu_imgs_chw = [mask, sens_maps, img_zf**0.2, output**0.2, target**0.2, error_chw]
            captions = ['mask', 'sens_maps', 'zf', 'reconstruction', 'target', 'error']

            # --- Convert all to HWC for W&B ---
            cpu_imgs_hwc = []
            for img_chw in cpu_imgs_chw:
                img_chw = img_chw.detach().cpu()
                if img_chw.ndim == 3:
                    if img_chw.shape[0] in (1, 3):
                        img_hwc = img_chw.permute(1, 2, 0).numpy()
                        if img_hwc.shape[-1] == 1:
                            img_hwc = np.repeat(img_hwc, 3, axis=-1)
                    else:
                        img_hwc = img_chw.numpy()
                elif img_chw.ndim == 2:
                    img_hwc = img_chw.numpy()
                else:
                    raise ValueError(f"Unexpected image shape: {img_chw.shape}")
                cpu_imgs_hwc.append(img_hwc)

            self.log_image(key, cpu_imgs_hwc, captions=captions)
                

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                utils.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                utils.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                utils.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_vals[fname] = maxval

        val_step_out_dict = {
            "val_loss": val_logs["loss"],
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "max_vals": max_vals,
        }

        self.validation_step_outputs.append(val_step_out_dict)

        # ─────────── CLEAR LIGHTNING’S INTERNAL BUFFER ───────────
        # (so it doesn’t accumulate every batch’s outputs forever)
        # See Issue: https://github.com/Lightning-AI/pytorch-lightning/issues/19398
        # Should be addressed by this PR: https://github.com/Lightning-AI/pytorch-lightning/pull/20730
        if hasattr(self.trainer, "validation_loop"):
            self.trainer.validation_loop._results = []
        gc.collect()
        # ──────────────────────────────────────────────────────────
         

    def log_image(self, key, images, captions):
        # tensorboard
        # self.logger.experiment.add_image(name, image, global_step=self.global_step)
        # wandb
        self.logger.log_image(key, images, caption=captions, step=self.global_step)
        # logger
        # self.logger.experiment.log(
        #     {key: [wandb.Image(img, caption=caption) for (img,caption) in zip(images,captions)]},
        #     step=self.global_step,
        # )



    def on_validation_epoch_end(self):        
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        for val_log in self.validation_step_outputs:
            losses.append(val_log["val_loss"].view(-1))
            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        # detach & move to CPU, then .item() to get a Python float  
        self.log("validation_loss", (val_loss / tot_slice_examples).detach().cpu().item(), prog_bar=True, sync_dist=True)

        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", (value / tot_examples).detach().cpu().item(), sync_dist=True)
                        
        self.validation_step_outputs.clear()


    def test_epoch_end(self, test_logs):
        outputs = defaultdict(dict)

        # use dicts for aggregation to handle duplicate slices in ddp mode
        for log in test_logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname][int(slice_num.cpu())] = log["output"][i]

        # stack all the slices for each file
        for fname in outputs:
            outputs[fname] = np.stack(
                [out for _, out in sorted(outputs[fname].items())]
            )

        # pull the default_root_dir if we have a trainer, otherwise save to cwd
        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"
        self.print(f"Saving reconstructions to {save_path}")

        save_reconstructions(outputs, save_path)
