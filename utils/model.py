import argparse
from torch.utils.checkpoint import is_compiled_module
from typing import List
from diffusers import (
    AutoencoderKL, UNet2DConditionModel, DDPMScheduler,
    CLIPTextModel, ControlNetModel
)


def load_models(config: argparse.Namespace,
                model_names: List[str]):
    """
    Selectively load models and noise scheduler from the checkpoint.
    """
    
    vae, unet, text_encoder, controlnet, noise_scheduler = \
        None, None, None, None, None
    
    if 'vae' in model_names:
        vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="vae",
            cache_dir=config.cache_dir,
            revision=config.revision,
            variant=config.variant
        )
    
    if 'unet' in model_names:
        unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="unet",
            cache_dir=config.cache_dir,
            revision=config.revision,
            variant=config.variant
        )

    if 'text_encoder' in model_names:
        text_encoder = CLIPTextModel.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="text_encoder",
            cache_dir=config.cache_dir,
            revision=config.revision,
            variant=config.variant
        )

    if 'controlnet' in model_names:
        controlnet = ControlNetModel.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="controlnet",
            cache_dir=config.cache_dir
        )
    if 'noise_scheduler' in model_names:
        scheduler = DDPMScheduler.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="scheduler",
            cache_dir=config.cache_dir
        )
    
    return vae, unet, text_encoder, controlnet, scheduler


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model
