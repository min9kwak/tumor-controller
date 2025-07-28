import argparse
import os
import gc
import wandb
import contextlib
import matplotlib.pyplot as plt
import math
import logging
import shutil
import numpy as np

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader

import transformers
import bitsandbytes as bnb

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import StableDiffusionInpaintPipeline
from diffusers.optimization import get_scheduler

from dataset.brats import BraTSDataset, BraTSProcessor
from dataset.transforms import create_transforms, create_mask_transforms, brats_collate_fn
from utils.model import load_models


class LDMInpaintFineTuner:
    def __init__(self,
                 args: argparse.Namespace,
                 model_names: List[str],
                 trainable_model_names: List[str],
                 processor: BraTSProcessor):
        
        self.args = args
        self.model_names = model_names
        self.trainable_model_names = trainable_model_names
        self.processor = processor
        
        self._configure_root_logger()
        self._set_accelerator()
        self._set_logging_verbosity()

        if self.args.seed is not None:
            set_seed(self.args.seed)
        
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)
        
        # prepare datasets
        self._prepare_datasets()
        
        # load models
        self._load_models(self.model_names)
        self._set_optimizer()
        self._set_lr_scheduler()
        self._prepare_with_accelerator()
        
    def _prepare_datasets(self):
        """prepare datasets and dataloaders"""
        # process data
        with self.accelerator.main_process_first():
            data_info = self.processor.process()
        
        # set return_keys
        return_keys = ['image', 'dilate', 'input_ids', 'label', 'prompt', 'idx']
        
        # prepare train dataset
        train_data_info = data_info['train']['tumor'] + data_info['train']['healthy']
        image_transform, _ = create_transforms(resolution=self.args.resolution, train=True)
        dilate_transform, _ = create_mask_transforms(resolution=self.args.resolution, dilation_size=self.args.dilation_size, sigma=self.args.sigma)
        
        train_set = BraTSDataset(data_info=train_data_info, image_transform=image_transform,
                                 dilate_transform=dilate_transform, return_keys=return_keys)

        self.train_loader = DataLoader(dataset=train_set, batch_size=self.args.train_batch_size, shuffle=True,  collate_fn=brats_collate_fn,
                                       num_workers=self.args.dataloader_num_workers if hasattr(self.args, 'dataloader_num_workers') else 0)
        
        # TODO: prepare test dataset
        
        # prepare validation dataset (select some samples from test dataset)
        image_transform_val, _ = create_transforms(resolution=self.args.resolution, train=False)
            
        # create full test dataset first to avoid healthy samples being filtered out
        test_info = data_info['test']['tumor'] + data_info['test']['healthy']
        full_test_set = BraTSDataset(data_info=test_info, image_transform=image_transform_val,
                                     dilate_transform=dilate_transform, return_keys=return_keys)
        
        # get indices for tumor and healthy samples from the full test set
        tumor_indices = [i for i, d in enumerate(full_test_set.data_info) if int(d['label']) == 1]
        healthy_indices = [i for i, d in enumerate(full_test_set.data_info) if int(d['label']) == 0]
        
        # select validation indices from each class
        np.random.seed(self.args.seed)
        samples_per_class = self.args.num_validation_samples // 2
        
        selected_tumor_idx = np.random.choice(len(tumor_indices),
                                              size=min(samples_per_class, len(tumor_indices)),
                                              replace=False)
        selected_healthy_idx = np.random.choice(len(healthy_indices),
                                                size=min(samples_per_class, len(healthy_indices)),
                                                replace=False)
        
        # get actual dataset indices
        self.validation_indices = ([tumor_indices[i] for i in selected_tumor_idx] + [healthy_indices[i] for i in selected_healthy_idx])
        self.validation_set = full_test_set

    def train(self):
        
        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            
            # Check if lr_scheduler has the expected number of training steps
            # Note: After accelerator.prepare(), lr_scheduler becomes AcceleratedScheduler
            original_scheduler = getattr(self.lr_scheduler, 'scheduler', self.lr_scheduler)
            if hasattr(original_scheduler, 'num_training_steps'):
                expected_steps = self.args.max_train_steps * self.accelerator.num_processes
                if original_scheduler.num_training_steps != expected_steps:
                    self.logger.warning(
                        f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(self.train_loader)}) does not match. "
                        "This inconsistency may result in the learning rate scheduler not functioning properly."
                    )
                
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.args))
            
            # Use task as project name and experiment_id as run name
            if self.args.report_to == "wandb":
                # Set wandb run name via environment variable before accelerator init
                os.environ["WANDB_RUN_NAME"] = self.args.experiment_id
                
                # Use only accelerator for wandb initialization (avoid double init)
                self.accelerator.init_trackers(self.args.task, config=tracker_config)
            else:
                self.accelerator.init_trackers(self.args.report_to, config=tracker_config)

        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        # self.logger.info(f"  Num examples = {len(self.train_set)}")
        self.logger.info(f"  Num batches each epoch = {len(self.train_loader)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        
        # Load checkpoint
        initial_global_step, first_epoch = self._load_checkpoint(num_update_steps_per_epoch)
        
        # Create the progress bar. Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        # image_logs = None
        global_step = initial_global_step

        # Run baseline validation before training starts
        if self.accelerator.is_main_process and self.validation_set is not None:
            self.logger.info("Running baseline validation before training...")
            self.log_validation(is_final_validation=False, global_step=0, validation_mode="baseline")
        
        # For accumulated logging
        accumulated_loss = 0.0
        log_count = 0
        
        for epoch in range(first_epoch, self.args.num_train_epochs):
            for _, batch in enumerate(self.train_loader):
                
                # Train Step
                loss = self.train_step(batch)
                
                # Accumulate loss for logging (every step)
                accumulated_loss = accumulated_loss + loss.detach().item()
                log_count = log_count + 1

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if self.accelerator.is_main_process:
                        if global_step % self.args.checkpointing_steps == 0:
                            # save checkpoint
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            self._save_checkpoint(global_step)

                        # logging
                        if self.validation_set is not None and global_step % self.args.validation_steps == 0:
                            self.log_validation(is_final_validation=False, global_step=global_step)
                        
                        # Log accumulated metrics every logging_steps
                        if global_step % self.args.logging_steps == 0 and log_count > 0:
                            avg_loss = accumulated_loss / log_count
                            logs = {
                                "train/loss": avg_loss,
                                "train/lr": self.lr_scheduler.get_last_lr()[0],
                                "train/epoch": epoch
                            }
                            self.accelerator.log(logs, step=global_step)
                            
                            # Reset accumulation
                            accumulated_loss = 0.0
                            log_count = 0

                # Update progress bar with current loss
                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                if global_step >= self.args.max_train_steps:
                    break
        
        # create the pipeline using the trained modules and save it
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # save trainable models
            for name in self.trainable_model_names:
                model = getattr(self, name)
                model = self.accelerator.unwrap_model(model)
                setattr(self, name, model)
                model.save_pretrained(os.path.join(self.args.output_dir, name))

            # run a final round of validation
            if self.validation_set is not None:
                self.log_validation(is_final_validation=True, global_step=global_step)
        
        # finish
        self.accelerator.end_training()

    def train_step(self, batch):
        # TODO: self.unet -> select from trainable_model_names
        with self.accelerator.accumulate(self.unet):
            
            # create mask: [B, 1, H, W]; binary
            mask = batch['dilate']
            image = batch['image'].to(dtype=self.weight_dtype)

            # apply mask to image: due to -1 ~ +1 normalization, -1 is black (masked)
            image_masked = image.clone()
            # Expand mask to match image channels: [B, 1, H, W] -> [B, C, H, W]
            mask_expanded = mask.expand(-1, image.shape[1], -1, -1)
            image_masked[mask_expanded == 0] = -1.0

            # image to latent
            latent_target = self.vae.encode(image).latent_dist.sample()
            latent_target = latent_target * self.vae.config.scaling_factor

            latent_masked = self.vae.encode(image_masked).latent_dist.sample()
            latent_masked = latent_masked * self.vae.config.scaling_factor

            # sample and add noise to the latents
            if self.args.offset_noise:
                noise = torch.randn_like(latent_masked) + 0.1 * torch.randn(
                    latent_masked.shape[0], latent_masked.shape[1], 1, 1, device=latent_masked.device
                )
            else:
                noise = torch.randn_like(latent_masked)

            bsz, channels, height, width = latent_masked.shape

            # sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latent_masked.device
            )
            timesteps = timesteps.long()

            noisy_latent = self.noise_scheduler.add_noise(latent_masked.float(), noise.float(), timesteps)
            noisy_latent = noisy_latent.to(dtype=self.weight_dtype)

            if self.accelerator.unwrap_model(self.unet).config.in_channels == channels * 2:
                noisy_latent = torch.cat([noisy_latent, noisy_latent], dim=1)

            # Get the text embedding for conditioning
            encoder_hidden_state = self.text_encoder(batch["input_ids"], return_dict=False)[0]

            # Unet forward
            pred = self.unet(noisy_latent, timesteps, encoder_hidden_state, return_dict=False)[0]

            # ground truth noise
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latent_target, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            
            # resize mask to latent size
            mask_latent = F.interpolate(mask.float(), size=target.shape[-2:], mode="nearest")

            # loss: only over masked (inpaint) region
            loss = F.mse_loss(pred * mask_latent, target * mask_latent, reduction="sum") / (mask_latent.sum() + 1e-8)

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = []
                for name in self.trainable_model_names:
                    model = getattr(self, name, None)
                    if model is not None:
                        params_to_clip.extend(model.parameters())
                self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
            
                # average scalar loss from all GPUs
                loss_avg = self.accelerator.reduce(loss.detach(), reduction="mean")
            else:
                # fallback for accumulation steps
                loss_avg = loss

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=self.args.set_grads_to_none)

            return loss_avg
    
    def _set_accelerator(self):

        assert self.args.report_to == 'wandb', "Only wandb is supported for now"

        logging_dir = Path(self.args.output_dir, self.args.logging_dir)

        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir,
            logging_dir=logging_dir
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=project_config,
        )

        # Set AMP (except for MPS)
        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False

        # Log accelerator state early
        self.logger.info(str(self.accelerator.state))

    def _set_logging_verbosity(self):
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    def _configure_root_logger(self) -> logging.Logger:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger = get_logger(__name__)

    def _load_models(self, model_names):
        # Selectively load models
        vae, unet, text_encoder, controlnet, noise_scheduler = load_models(
            self.args, model_names
        )

        model_dict = {
            'vae': vae,
            'unet': unet,
            'text_encoder': text_encoder,
            'controlnet': controlnet
        }

        # Selectively load trainable models
        for name, model in model_dict.items():
            if name in self.trainable_model_names and model is not None:
                model.train()
                model.requires_grad_(True)
            elif model is not None:
                model.eval()
                model.requires_grad_(False)
        if noise_scheduler is not None:
            self.noise_scheduler = noise_scheduler
        
        if self.args.gradient_checkpointing:
            for name, model in model_dict.items():
                if name in self.trainable_model_names and model is not None:
                    model.enable_gradient_checkpointing()

        for name, model in model_dict.items():
            if name in self.trainable_model_names:
                if self.accelerator.unwrap_model(model).dtype != torch.float32:
                    raise ValueError(f"{name} loaded as datatype {self.accelerator.unwrap_model(model).dtype}")
            setattr(self, name, model)
        
        self.noise_scheduler = noise_scheduler
    
    def _set_optimizer(self):

        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate * self.args.gradient_accumulation_steps * self.args.train_batch_size * self.accelerator.num_processes
            )

        if self.args.use_8bit_adam:
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = bnb.optim.AdamW
        
        trainable_params = []
        for name in self.trainable_model_names:
            model = getattr(self, name, None)
            if model is not None:
                trainable_params.extend(model.parameters())

        self.optimizer = optimizer_cls(
            trainable_params,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
    
    def _set_lr_scheduler(self):
        num_warmup_steps_for_scheduler = self.args.lr_warmup_steps * self.accelerator.num_processes
        if self.args.max_train_steps is None:
            len_train_dataloader_after_sharding = math.ceil(len(self.train_loader) / self.accelerator.num_processes)
            num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / self.args.gradient_accumulation_steps)
            num_training_steps_for_scheduler = (
                self.args.num_train_epochs * num_update_steps_per_epoch * self.accelerator.num_processes
            )
        else:
            num_training_steps_for_scheduler = self.args.max_train_steps * self.accelerator.num_processes
        
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )
    
    def _prepare_with_accelerator(self):
        # Prepare only trainable models
        models = [getattr(self, name) for name in self.trainable_model_names]
        prepared = self.accelerator.prepare(*models, self.optimizer, self.train_loader, self.lr_scheduler)

        for name, model in zip(self.trainable_model_names, prepared[:len(models)]):
            setattr(self, name, model)
        self.optimizer, self.train_loader, self.lr_scheduler = prepared[len(models):]

        # Determine weight dtype for inference models
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        # Move inference-only models (eval + no grad) to device with proper dtype
        for name in ['vae', 'unet', 'text_encoder', 'controlnet']:
            if name not in self.trainable_model_names:
                model = getattr(self, name, None)
                if model is not None:
                    model.to(self.accelerator.device, dtype=weight_dtype)
        self.weight_dtype = weight_dtype

    def _load_checkpoint(self, num_update_steps_per_epoch):
        
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0

        return initial_global_step, first_epoch

    def _save_checkpoint(self, global_step):
        if self.args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= self.args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                self.logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                self.logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(self.args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
        self.accelerator.save_state(save_path)
        self.logger.info(f"Saved state to {save_path}")

    @torch.no_grad()
    def log_validation(self, is_final_validation=False, global_step=0, validation_mode=None):
        
        self.logger.info("Running validation... ")

        # Step 1: load models
        # Load models based on whether they were trained or not
        if not is_final_validation:
            # During training validation - unwrap only trainable models
            if 'vae' in self.trainable_model_names:
                vae = self.accelerator.unwrap_model(self.vae)
            else:
                vae = self.vae  # Use original model (not prepared by accelerator)
            
            if 'unet' in self.trainable_model_names:
                unet = self.accelerator.unwrap_model(self.unet)
            else:
                unet = self.unet  # Use original model (not prepared by accelerator)
        else:
            # Final validation - load from saved checkpoints for trainable models, pretrained for others
            if 'vae' in self.trainable_model_names:
                vae = AutoencoderKL.from_pretrained(
                    os.path.join(self.args.output_dir, "vae"),
                    torch_dtype=self.weight_dtype
                )
            else:
                # Load original pretrained VAE since it wasn't trained
                vae = AutoencoderKL.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    subfolder="vae",
                    revision=self.args.revision,
                    variant=self.args.variant,
                    torch_dtype=self.weight_dtype
                )
            
            if 'unet' in self.trainable_model_names:
                unet = UNet2DConditionModel.from_pretrained(
                    os.path.join(self.args.output_dir, "unet"),
                    torch_dtype=self.weight_dtype,
                )
            else:
                # Load original pretrained UNet since it wasn't trained
                unet = UNet2DConditionModel.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    subfolder="unet",
                    revision=self.args.revision,
                    variant=self.args.variant,
                    torch_dtype=self.weight_dtype
                )
        
        vae.eval()
        unet.eval()
        
        # Step 2: define pipeline
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=self.text_encoder,
            tokenizer=self.processor.tokenizer,
            unet=unet,
            safety_checker=None,
            revision=self.args.revision,
            variant=self.args.variant,
            torch_dtype=self.weight_dtype,
        )
        # using UniPCMultistepScheduler provides different results but reuqires less time. use default
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if self.args.seed is not None:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(self.args.seed)
        else:
            generator = None

        inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")

        # Step 3: define tracker
        if validation_mode:
            tracker_key = validation_mode
        elif is_final_validation:
            tracker_key = "test"
        else:
            tracker_key = "validation"
            
        wandb_tracker = next(
            (t for t in self.accelerator.trackers if t.name == "wandb"), None
        )
        if wandb_tracker is None:
            self.logger.warning(f"Wandb tracker does not exist, skipping image logging.")
            # return []
        
        # Step 4: inference
        for mode in ['original', 'reversed']:
            logs = []
            for cnt, val_idx in enumerate(self.validation_indices):
                
                data = self.validation_set[val_idx]
                
                src = data['image']         # 0 ~ 1, (3, 512, 512)
                mask = data['dilate']       # 0 ~ 1, (1, 512, 512)
                prompt = data['prompt']     # string

                if mode == 'original':
                    prompt = prompt
                else:
                    try:
                        prompt = self.processor.prompt_builder.reverse_prompt(prompt)
                    except Exception as e:
                        self.logger.warning(f"Reverse prompt failed at idx={cnt} due to: {e}")
                        continue
                
                with inference_ctx:
                    out = pipeline(
                        prompt,
                        image=src,
                        mask_image=mask * 255.0,
                        output_type='pt',
                        num_inference_steps=self.args.num_inference_steps,
                        guidance_scale=self.args.guidance_scale,
                        generator=generator
                    ).images[0] # (3, 512, 512)
                
                # Convert grayscale
                src_gray = src.mean(0).detach().cpu().numpy()        # [H, W]
                out_gray = out.mean(0).detach().cpu().numpy()       # [H, W]
                mask_np = mask.squeeze(0).detach().cpu().numpy()       # [H, W], binary 0 or 1

                # Diff: abs(original - translated)
                diff = np.abs(src_gray - out_gray)

                # Repaint: output only in masked region, original elsewhere
                repaint = mask_np * out_gray + (1 - mask_np) * src_gray
                diff_repaint = np.abs(repaint - src_gray)

                # Red overlay (semi-transparent)
                overlay_rgb = np.stack([src_gray]*3, axis=-1)        # [H, W, 3]
                alpha = 0.2
                red_layer = np.zeros_like(overlay_rgb)
                red_layer[..., 0] = 1.0  # pure red
                overlay_rgb[mask_np > 0.5] = (
                    (1 - alpha) * overlay_rgb[mask_np > 0.5] +
                    alpha * red_layer[mask_np > 0.5]
                )

                # Plot
                fig, axs = plt.subplots(2, 3, figsize=(9, 6))
                axs = axs.ravel()
                axs[0].imshow(src_gray, cmap='gray', vmin=0, vmax=1)
                axs[0].set_title("input")

                axs[1].imshow(mask_np, cmap='gray')
                axs[1].set_title("mask")

                axs[2].imshow(overlay_rgb)
                axs[2].set_title("overlay")

                axs[3].imshow(out_gray, cmap='gray', vmin=0, vmax=1)
                axs[3].set_title("translated")

                axs[4].imshow(diff, cmap='hot')
                axs[4].set_title("abs-diff")

                axs[5].imshow(diff_repaint, cmap='hot')
                axs[5].set_title("abs-diff (repaint)")

                for ax in axs:
                    ax.axis('off')

                plt.tight_layout()
                
                # use wandb caption
                logs.append(wandb.Image(fig, caption=f"[{cnt}] [{mode}] {prompt}"))
                plt.close(fig)
                del fig
            
            # Step 5: log
            if self.accelerator.is_main_process:
                wandb_tracker.log({f"{tracker_key}_{mode}": logs}, step=global_step)
        
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        # return logs
