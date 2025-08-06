# from .promptmr_v2 import PromptMR
# from .prompt_mamba import PromptMR
from .vssd_recon import PromptMR

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
