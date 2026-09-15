"""Model variants. Select one at runtime with `model_version` (see pl_modules/promptmr_module.py
and models/README.md).

    e2e_recon           CNN U-Net denoiser + CNN U-Net SME         (E2E-VarNet baseline)
    swin_recon          Swin transformer blocks                    (transformer baseline)
    vss_recon           VMamba SS2D 4-direction scan blocks         (multi-scan SSM baseline)
    vssd_recon          VSSD NC-SSD blocks, MSA bottleneck
    vssd_recon_convsme  as vssd_recon, but CNN U-Net SME
    vssd_recon_fixed    vssd_recon with the head / patch-size / dA defects corrected

Importing this package must not pull in any variant: `get_model_class` imports the
selected module by name, and the unselected ones have incompatible extension deps.
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def count_trainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if model is not None
        else 0
    )


def count_untrainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if model is not None
        else 0
    )
