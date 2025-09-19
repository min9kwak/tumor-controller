import json
import os
import gc
import datetime
import numpy as np
import torch
import contextlib
import matplotlib.pyplot as plt
import yaml
import argparse
from tqdm import tqdm
from easydict import EasyDict as edict

from dataset.brats import BraTSProcessor, BraTSDataset
from dataset.preprocessing.helper import PromptBuilder
from dataset.transforms import create_transforms, create_mask_transforms

from accelerate import Accelerator
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers import StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline


def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)


def setup_argparse():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(description='ControlNet Image Generation')
    
    # Config file
    parser.add_argument('--config', type=str, default='yaml/generate_controlnet.yaml',
                      help='Path to YAML config file')
    
    # Model settings
    parser.add_argument('--controlnet-id', type=str,
                      help='ControlNet checkpoint ID')
    
    # Inference parameters
    parser.add_argument('--num-steps', type=int,
                      help='Number of inference steps')
    parser.add_argument('--strength', type=float,
                      help='Denoising strength')
    parser.add_argument('--guidance-scale', type=float,
                      help='Guidance scale for classifier-free guidance')
    parser.add_argument('--controlnet-conditioning-scale', type=float,
                      help='ControlNet conditioning scale')
    
    return parser


def merge_configs(yaml_config, args):
    """Merge YAML config with command line arguments (args take priority)."""
    config = edict()
    
    # Start with YAML config
    config.update(yaml_config)
    
    # Override with command line arguments (if provided)
    if args.controlnet_id is not None:
        config.model.controlnet_id = args.controlnet_id
    
    if args.num_steps is not None:
        config.inference.num_steps = args.num_steps
    if args.strength is not None:
        config.inference.strength = args.strength
    if args.guidance_scale is not None:
        config.inference.guidance_scale = args.guidance_scale
    if args.controlnet_conditioning_scale is not None:
        config.inference.controlnet_conditioning_scale = args.controlnet_conditioning_scale
    
    return config


def main(args=None):
    """Main function for ControlNet image generation."""
    # Parse arguments
    parser = setup_argparse()
    if args is None:
        args = parser.parse_args()
    
    # Load YAML config
    yaml_config = load_yaml_config(args.config)
    
    # Merge configs (argparse > YAML > default)
    config = merge_configs(yaml_config, args)
    
    # Setup experiment ID (always auto-generate)
    experiment_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # create output directory structure
    output_base_dir = os.path.join("image_generation", experiment_id)
    output_dir_controlnet = os.path.join(output_base_dir, "controlnet")
    output_dir_ldm = os.path.join(output_base_dir, "ldm")
    output_dir_plot = os.path.join(output_base_dir, "plot")

    # create directories
    os.makedirs(output_dir_controlnet, exist_ok=True)
    os.makedirs(output_dir_ldm, exist_ok=True)
    os.makedirs(output_dir_plot, exist_ok=True)

    # controlnet config
    controlnet_config_file = os.path.join('checkpoints/controlnet', config.model.controlnet_id, 'configs.json')
    assert os.path.exists(controlnet_config_file), f"Checkpoint not found: {controlnet_config_file}"
    with open(controlnet_config_file, 'r') as f:
        controlnet_config = edict(json.load(f))

    tokenizer = AutoTokenizer.from_pretrained(controlnet_config.pretrained_model_name_or_path, subfolder='tokenizer',
                                              cache_dir=controlnet_config.cache_dir, use_fast=False)
    prompt_builder = PromptBuilder()
    processor = BraTSProcessor(config=vars(controlnet_config), tokenizer=tokenizer, prompt_builder=prompt_builder)

    # set accelerator if needed
    accelerator = Accelerator()
    with accelerator.main_process_first():
        data_info = processor.process()

    return_keys = ['image', 'edge', 'dilate', 'input_ids', 'label', 'prompt', 'idx']
    image_transform_val, edge_transform_val = create_transforms(resolution=controlnet_config.resolution, train=False)
    dilate_transform, _ = create_mask_transforms(
        resolution=controlnet_config.resolution, dilation_size=controlnet_config.dilation_size, sigma=controlnet_config.sigma
    )

    test_info = data_info['test']['tumor'] + data_info['test']['healthy']
    test_set = BraTSDataset(data_info=test_info, image_transform=image_transform_val,
                            edge_transform=edge_transform_val,
                            dilate_transform=dilate_transform,
                            blur_transform=None,
                            return_keys=return_keys,
                            edge_blur=controlnet_config.edge_blur,
                            seed=controlnet_config.seed)
    tumor_indices = [i for i, d in enumerate(test_set.data_info) if int(d['label']) == 1]

    # load models
    vae = AutoencoderKL.from_pretrained(
        controlnet_config.pretrained_model_name_or_path,
        subfolder="vae",
        cache_dir=controlnet_config.cache_dir,
        revision=controlnet_config.revision,
        variant=controlnet_config.variant
    )
    text_encoder = CLIPTextModel.from_pretrained(
        controlnet_config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        cache_dir=controlnet_config.cache_dir,
        revision=controlnet_config.revision,
        variant=controlnet_config.variant
    )

    unet_dir = os.path.join("checkpoints/finetune_ldm_inpaint", controlnet_config.ldm_inpaint_hash, "unet")
    assert os.path.exists(unet_dir), f"UNet directory not found: {unet_dir}"
    unet = UNet2DConditionModel.from_pretrained(unet_dir)

    ldm_config_path = os.path.join(os.path.dirname(unet_dir), 'configs.json')
    with open(ldm_config_path, 'r') as f:
        ldm_config = edict(json.load(f))

    # add config file paths to our experiment config
    config.paths = edict()
    config.paths.controlnet_config = controlnet_config_file
    config.paths.ldm_config = ldm_config_path
    config.paths.yaml_config = args.config

    # save experiment config
    experiment_config_path = os.path.join(output_base_dir, 'configs.json')
    with open(experiment_config_path, 'w') as f:
        json.dump(dict(config), f, indent=2)

    controlnet_dir = os.path.join(controlnet_config.output_dir, "controlnet")
    assert os.path.exists(controlnet_dir), f"ControlNet directory not found: {controlnet_dir}"
    controlnet = ControlNetModel.from_pretrained(controlnet_dir)

    vae.eval()
    vae.requires_grad_(False)

    text_encoder.eval()
    text_encoder.requires_grad_(False)

    unet.eval()
    unet.requires_grad_(False)

    controlnet.eval()
    controlnet.requires_grad_(False)

    # controlnet pipeline
    pipeline_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        controlnet_config.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=processor.tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float32, # 16 or 32
    )
    pipeline_controlnet.enable_model_cpu_offload()
    pipeline_controlnet.set_progress_bar_config(disable=True)

    # ldm inpaint
    pipeline_ldm = StableDiffusionInpaintPipeline.from_pretrained(
        ldm_config.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=processor.tokenizer,
        unet=unet,
        safety_checker=None,
        revision=ldm_config.revision,
        variant=ldm_config.variant,
        torch_dtype=torch.float32,
    )
    pipeline_ldm.enable_model_cpu_offload()
    pipeline_ldm.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(controlnet_config.seed)
    inference_ctx = contextlib.nullcontext()

    # Setup progress bar
    pbar = tqdm(enumerate(tumor_indices), desc="Processing tumor samples", unit="sample", total=len(tumor_indices))
    
    for sample_idx, test_idx in pbar:
        data = test_set[test_idx]
        raw_data = test_set.data_info[test_idx]
        
        # Extract metadata for filename
        patient_id = raw_data['patient_id']
        modality = raw_data['modality']
        slice_num = raw_data['slice']
        
        # Update progress bar with current sample info
        pbar.set_description(f"Processing {patient_id}-{modality}-{slice_num}")
        
        src = data['image']
        edge = data['edge']
        mask = data['dilate']
        prompt = data['prompt']
        
        # reverse prompt
        try:
            prompt = processor.prompt_builder.reverse_prompt(prompt)
        except Exception as e:
            tqdm.write(f"⚠️ Warning: Failed to reverse prompt '{prompt}': {e}")
            continue
        
        edge = torch.flip(edge, dims=[1])
        
        with torch.no_grad():
            with inference_ctx:
                out_controlnet = pipeline_controlnet(
                    prompt,
                    image=src,
                    mask_image=mask * 255.0,
                    control_image=edge.unsqueeze(0),
                    output_type='pt',
                    num_inference_steps=config.inference.num_steps,
                    strength=config.inference.strength,
                    guidance_scale=config.inference.guidance_scale,
                    controlnet_conditioning_scale=config.inference.controlnet_conditioning_scale,
                    generator=generator
                ).images[0]
        
        with torch.no_grad():
            with inference_ctx:
                out_ldm = pipeline_ldm(
                    prompt,
                    image=src,
                    mask_image=mask * 255.0,
                    output_type='pt',
                    num_inference_steps=config.inference.num_steps,
                    guidance_scale=config.inference.guidance_scale,
                    strength=config.inference.strength,
                    generator=generator
                ).images[0] # (3, 512, 512)

        # Save generated images as numpy arrays
        controlnet_filename = f"{patient_id}-controlnet-{modality}-{slice_num}.npy"
        ldm_filename = f"{patient_id}-ldm-{modality}-{slice_num}.npy"
        
        controlnet_save_path = os.path.join(output_dir_controlnet, controlnet_filename)
        ldm_save_path = os.path.join(output_dir_ldm, ldm_filename)
        
        np.save(controlnet_save_path, out_controlnet.detach().cpu().numpy())
        np.save(ldm_save_path, out_ldm.detach().cpu().numpy())

        # Prepare common data for visualization
        src_gray = src.mean(0).detach().cpu().numpy()        # [H, W]
        mask_np = mask.squeeze(0).detach().cpu().numpy()       # [H, W], binary 0 or 1

        # Red overlay (semi-transparent)
        overlay_rgb = np.stack([src_gray]*3, axis=-1)        # [H, W, 3]
        alpha = 0.2
        red_layer = np.zeros_like(overlay_rgb)
        red_layer[..., 0] = 1.0  # pure red
        overlay_rgb[mask_np > 0.5] = (
            (1 - alpha) * overlay_rgb[mask_np > 0.5] +
            alpha * red_layer[mask_np > 0.5]
        )

        # ControlNet results
        out_gray_controlnet = out_controlnet.mean(0).detach().cpu().numpy()       # [H, W]
        diff_controlnet = np.abs(src_gray - out_gray_controlnet)
        repaint_controlnet = mask_np * out_gray_controlnet + (1 - mask_np) * src_gray
        diff_repaint_controlnet = np.abs(repaint_controlnet - src_gray)

        # LDM results
        out_gray_ldm = out_ldm.mean(0).detach().cpu().numpy()       # [H, W]
        diff_ldm = np.abs(src_gray - out_gray_ldm)
        repaint_ldm = mask_np * out_gray_ldm + (1 - mask_np) * src_gray
        diff_repaint_ldm = np.abs(repaint_ldm - src_gray)

        # Comparison between ControlNet and LDM
        diff_translated = np.abs(out_gray_controlnet - out_gray_ldm)
        diff_repaint_comparison = np.abs(repaint_controlnet - repaint_ldm)

        # Plot 4x4 grid
        fig, axs = plt.subplots(4, 4, figsize=(16, 16))

        # First row: inputs
        axs[0, 0].imshow(src_gray, cmap='gray', vmin=0, vmax=1)
        axs[0, 0].set_title("image")

        axs[0, 1].imshow(mask_np, cmap='gray')
        axs[0, 1].set_title("mask")

        axs[0, 2].imshow(overlay_rgb)
        axs[0, 2].set_title("overlay")

        axs[0, 3].imshow(edge[0], cmap='gray')
        axs[0, 3].set_title("edge")

        # Second row: ControlNet results
        axs[1, 0].imshow(out_gray_controlnet, cmap='gray', vmin=0, vmax=1)
        axs[1, 0].set_title("ControlNet translated")

        axs[1, 1].imshow(diff_controlnet, cmap='hot')
        axs[1, 1].set_title("ControlNet trans-diff")

        axs[1, 2].imshow(repaint_controlnet, cmap='gray', vmin=0, vmax=1)
        axs[1, 2].set_title("ControlNet repaint")

        axs[1, 3].imshow(diff_repaint_controlnet, cmap='hot')
        axs[1, 3].set_title("ControlNet repaint-diff")

        # Third row: LDM results
        axs[2, 0].imshow(out_gray_ldm, cmap='gray', vmin=0, vmax=1)
        axs[2, 0].set_title("LDM translated")

        axs[2, 1].imshow(diff_ldm, cmap='hot')
        axs[2, 1].set_title("LDM trans-diff")

        axs[2, 2].imshow(repaint_ldm, cmap='gray', vmin=0, vmax=1)
        axs[2, 2].set_title("LDM repaint")

        axs[2, 3].imshow(diff_repaint_ldm, cmap='hot')
        axs[2, 3].set_title("LDM repaint-diff")

        # Fourth row: Comparison between ControlNet and LDM
        axs[3, 0].axis('off')  # empty

        axs[3, 1].imshow(diff_translated, cmap='hot')
        axs[3, 1].set_title("Diff: ControlNet vs LDM\n(translated)")

        axs[3, 2].axis('off')  # empty

        axs[3, 3].imshow(diff_repaint_comparison, cmap='hot')
        axs[3, 3].set_title("Diff: ControlNet vs LDM\n(repaint)")

        # Remove axis from all subplots
        for i in range(4):
            for j in range(4):
                axs[i, j].axis('off')

        plt.tight_layout()

        # Save overview plot
        overview_filename = f"{patient_id}-overview-{modality}-{slice_num}.png"
        overview_save_path = os.path.join(output_dir_plot, overview_filename)
        plt.savefig(overview_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Memory cleanup (del variables every iteration, cache cleanup periodically)
        del fig, axs
        del out_controlnet, out_ldm
        del src_gray, mask_np, overlay_rgb
        del out_gray_controlnet, out_gray_ldm
        del diff_controlnet, diff_ldm, diff_translated, diff_repaint_comparison
        del repaint_controlnet, repaint_ldm, diff_repaint_controlnet, diff_repaint_ldm
        
        # Periodic cache cleanup
        if sample_idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # break  # Uncomment this line to process only one sample for testing
    
    # Print completion summary
    pbar.close()
    print(f"\n✅ Processing completed!")
    print(f"📂 Results saved to: {output_base_dir}")
    print(f"📊 Processed {len(tumor_indices)} tumor samples")
    
    # Return summary for telegram message
    return {
        'output_dir': output_base_dir,
        'num_samples': len(tumor_indices),
        'experiment_id': experiment_id
    }


if __name__ == "__main__":
    from utils.msg import send_telegram
    
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    try:
        # Send telegram message when generation starts
        send_telegram(dotenv_path='.env', message=f'🚀 Starting ControlNet image generation.')
        result = main(args)
        # Completed successfully
        completion_msg = (
            f'✅ Successfully completed ControlNet image generation!\n'
            f'📊 Processed {result["num_samples"]} tumor samples\n'
            f'📂 Results saved to: {result["output_dir"]}\n'
            f'🆔 Experiment ID: {result["experiment_id"]}'
        )
        send_telegram(dotenv_path='.env', message=completion_msg)
    except KeyboardInterrupt:
        # Interrupted by user
        send_telegram(dotenv_path='.env', message=f'⚠️ Image generation interrupted by user.')
        raise
    except Exception as e:
        # Other errors
        send_telegram(dotenv_path='.env', message=f'❌ Image generation failed with error: {str(e)}')
        raise
