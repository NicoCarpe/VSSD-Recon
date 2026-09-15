# Model variants

Same unrolled shape throughout — `PromptMR` (cascades) -> `PromptMRBlock` (DC + denoiser)
-> `NormPromptUnet` -> `SensitivityModel` — differing in which block the U-Net is built
from and which network estimates the sensitivity maps.

Pick one with `model_version` in the model config; `PromptMrModule` imports
`models.<model_version>` by name and takes its `PromptMR` class.

```yaml
model:
  class_path: pl_modules.PromptMrModule
  init_args:
    model_version: vssd_recon_fixed
```

| `model_version`      | Enc / dec / skip block                | Bottleneck  | SME network | Needs `mamba_ssm` |
| -------------------- | ------------------------------------- | ----------- | ----------- | ----------------- |
| `e2e_recon`          | plain conv (`ConvBlock`)              | (in `UNet`) | conv U-Net  | no                |
| `swin_recon`         | Swin transformer, window 8            | Swin        | Swin U-Net  | no                |
| `vss_recon`          | VMamba `SS2D`, 4-direction cross-scan | `SS2D`      | VSS U-Net   | **yes**           |
| `vssd_recon`         | VSSD block, NC-SSD mixer              | MSA         | VSSD U-Net  | no                |
| `vssd_recon_convsme` | VSSD block, NC-SSD mixer              | MSA         | conv U-Net  | no                |
| `vssd_recon_fixed`   | VSSD block, NC-SSD mixer              | MSA         | VSSD U-Net  | no                |

`mamba_ssm` is only needed on the `linear_attn_duality=False` path, which falls back to
`mamba_chunk_scan_combined`, the Triton causal chunk-scan kernel. The NC-SSD path
(`linear_attn_duality=True`, the VSSD-Recon configuration) does not call it.

## Scaffold defects, and `vssd_recon_fixed`

`swin_recon`, `vss_recon`, `vssd_recon` and `vssd_recon_convsme` all construct
`PatchEmbed(k=p, s=p)` and `FinalProjection(n_feat0, out_chans)` with the defaults, so
they share three defects. `e2e_recon` is the only one without that scaffold — no patch
embedding, and it ends in an unnormalised 1x1 conv — so read comparisons against it with
that in mind; comparisons among the four are less affected.

`vssd_recon_fixed` is `vssd_recon` with the three corrected, all outside the VSSD block
(details in the module docstring):

1. **Head normalisation.** `FinalProjection` normalised the *output* channels as its last
   operation, pinning `||out(h, w)||_2` at every pixel, so the head could set the
   direction of its correction but not its magnitude. The LayerNorm now sits on the
   features before the projection (`norm_output=False`).
2. **`patch_size` was inert.** `dim_scale` and the padding multiple were hard-coded for
   `patch_size=4`; other values returned a spatially misaligned field that
   `NormPromptUnet.unpad` cropped without error. Both are now tied to `patch_size`.
3. **`ssd_positve_dA`.** Pinned `False`, which leaves `dA = dt*A` negative
   (`A = -exp(A_log)`) so the aggregated term entered against the `D` skip. Defaults to
   `True` (upstream VSSD); set `ssd_positve_dA: false` to keep it out of a comparison.

The other variants are left as the original work reported them.

Two performance fixes in `VSSDBlock.py` apply to every VSSD variant and are bitwise
neutral: `tTensor` wrapping is off in eager mode (`USE_TTENSOR`; it routed the whole SSM
path through `__torch_function__`, 2.7x the wall clock), and `cpe1` takes the NCHW
tensor it was given instead of a copy recovered from the flattened one.
