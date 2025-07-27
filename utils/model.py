import argparse
from typing import List
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, ControlNetModel
from transformers import CLIPTextModel


def load_models(config: argparse.Namespace,
                model_names: List[str] = ["vae", "unet", "text_encoder", "controlnet", "noise_scheduler"]):
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
            variant=config.variant,
            cross_attention_dim=config.cross_attention_dim
        )
        unet.config.cross_attention_dim = config.cross_attention_dim

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
            # TODO: use different pretrained model name
            config.pretrained_model_name_or_path,
            subfolder="controlnet",
            cache_dir=config.cache_dir
        )
    if 'noise_scheduler' in model_names:
        noise_scheduler = DDPMScheduler.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="scheduler",
            cache_dir=config.cache_dir
        )
    
    return vae, unet, text_encoder, controlnet, noise_scheduler
