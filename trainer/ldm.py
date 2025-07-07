import argparse
import os
import gc
import wandb
import contextlib
import matplotlib.pyplot as plt
import math
import logging
import shutil

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
from diffusers import AutoPipelineForImage2Image
from diffusers.optimization import get_scheduler

from dataset.brats import BraTSDataset
from utils.model import load_models, unwrap_model


class LDMFineTuner:
    def __init__(self,
                 args: argparse.Namespace,
                 model_names: List[str],
                 trainable_model_names: List[str],
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 validation_set: BraTSDataset):
        
        self.args = args
        self.model_names = model_names
        self.trainable_model_names = trainable_model_names
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_set = validation_set

        self._configure_root_logger()
        self._set_accelerator()
        self._set_logging_verbosity()

        if self.args.seed is not None:
            set_seed(self.args.seed)
        
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

        self._load_models(self.model_names)
        self._set_optimizer()
        self._set_lr_scheduler()
        self._prepare_with_accelerator()
        
    def train(self):
        
        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            if self.lr_scheduler.num_training_steps != self.args.max_train_steps * self.accelerator.num_processes:
                self.logger.warning(
                    f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(self.train_loader)}) does not match. "
                    "This inconsistency may result in the learning rate scheduler not functioning properly."
                )
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.args))
            
            # tensorboard cannot handle list types for config
            # TODO: change this arguments later according to the config
            tracker_config.pop("validation_prompt")
            tracker_config.pop("validation_image")

            self.accelerator.init_trackers(self.args.report_to, tracker_config)

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

        image_logs = None
        for epoch in range(first_epoch, self.args.num_train_epochs):
            for step, batch in enumerate(self.train_loader):
                
                # Train Step
                loss = self.train_step(batch)

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if self.accelerator.is_main_process:
                        # save checkpoint
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        self._save_checkpoint(global_step)

                        # logging
                        if self.args.validation_set is not None and global_step % self.args.validation_steps == 0:
                            image_logs = self.log_validation(is_final_validation=False)

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)
                if global_step >= self.args.max_train_steps:
                    break
        
        # create the pipeline using the trained modules and save it
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            
            self.vae = unwrap_model(self.vae)
            self.vae.save_pretrained(self.args.output_dir)
            self.unet = unwrap_model(self.unet)
            self.unet.save_pretrained(self.args.output_dir)

            # run a final round of validation
            if self.args.validation_set is not None:
                image_logs = self.log_validation(is_final_validation=True)
        
        # finish
        self.accelerator.end_training()

    def train_step(self, batch):
        with self.accelerator.accumulate(self.unet):
            # image to latent
            latent = self.vae.encode(batch["image"].to(dtype=self.weight_dtype)).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor

            # sample and add noise to the latents
            # TODO: add noise_offset (default True) to the config file
            if self.args.offset_noise:
                noise = torch.randn_like(latent) + 0.1 * torch.randn(
                    latent.shape[0], latent.shape[1], 1, 1, device=latent.device
                )
            else:
                noise = torch.randn_like(latent)

            bsz, channels, height, width = latent.shape

            # sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latent.device
            )
            timesteps = timesteps.long()

            noisy_latent = self.noise_scheduler.add_noise(
                latent.float(), noise.float(), timesteps
            )
            noisy_latent = noisy_latent.to(dtype=self.weight_dtype)

            # Get the text embedding for conditioning
            input_id = self.text_encoder(batch["input_id"], return_dict=False)[0]

            # optional step
            if unwrap_model(self.unet).config.in_channels == channels * 2:
                noisy_latent = torch.cat([noisy_latent, noisy_latent], dim=1)

            pred = self.unet(
                noisy_latent,
                timesteps,
                input_id,
                return_dict=False,
            )

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latent, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = []
                for name in self.trainable_model_names:
                    model = getattr(self, name, None)
                    if model is not None:
                        params_to_clip.extend(model.parameters())
                self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
            
                # 모든 GPU의 스칼라 loss 값을 평균
                loss_avg = self.accelerator.reduce(loss.detach(), reduction="mean")
                if self.accelerator.is_main_process:     # ▸ rank-0만 Tracker 호출
                    self.accelerator.log(
                        {"train/loss": loss_avg.item()},
                        step=self.global_step             # or self.accelerator.step
                    )

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=self.args.set_grads_to_none)

            

    
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

    def _load_models(self):
        # Selectively load models
        vae, unet, text_encoder, controlnet, noise_scheduler = load_models(
            self.args, self.model_names
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
                if unwrap_model(model, self.accelerator).dtype != torch.float32:
                    raise ValueError(f"{name} loaded as datatype {unwrap_model(model, self.accelerator).dtype}")
            setattr(self, name, model)
        
        self.noise_scheduler = noise_scheduler
    
    def _set_optimizer(self):
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

    def log_validation(self, is_final_validation=False):
        self.logger.info("Running validation... ")

        # Step 1: load models
        # trainable models
        if not is_final_validation:
            vae = self.accelerator.unwrap_model(self.vae)
            unet = self.accelerator.unwrap_model(self.unet)
        else:
            vae = AutoencoderKL.from_pretrained(
                self.args.output_dir,
                torch_dtype=self.weight_dtype
            )
            unet = UNet2DConditionModel.from_pretrained(
                self.args.output_dir,
                torch_dtype=self.weight_dtype,
            )
        
        # Step 2: define pipeline
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            self.args.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=unet,
            # controlnet=controlnet,
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
        tracker_key = "test" if is_final_validation else "validation"
        wandb_tracker = next(
            (t for t in self.accelerator.trackers if t.name == "wandb"), None
        )
        if wandb_tracker is None:
            self.logger.warning(f"Wandb tracker does not exist, skipping image logging.")
            return []
        
        # Step 4: inference
        logs = []
        for idx, data in enumerate(self.validation_set):
            src = data['image']
            prompt = data['prompt']

            with inference_ctx, torch.no_grad():
                out = pipeline(
                    prompt,
                    image=src,
                    strength=self.args.strength,
                    guidance_scale=self.args.guidance_scale,
                    output_type='pt',
                    # negative_prompt='3d render, illustration, doll, cartoon, colored, bokeh',
                    num_inference_steps=self.args.num_inference_steps,
                    generator=generator
                ).images[0] # (3, 512, 512)
            
            # (512, 512)
            src_gray = src.mean(0)
            out_gray = out.mean(0)
            diff = (src_gray - out_gray).abs()

            # matplotlib
            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            for ax, img, title in zip(axs, [src_gray, out_gray, diff], ["input", "translated", "abs-diff"]):
                ax.imshow(img.cpu().numpy(), cmap='gray',
                          vmin=0.0, vmax=1.0)
                ax.set_title(title)
                ax.axis('off')
            fig.suptitle(f"[{idx}] {prompt}", fontsize=10)
            fig.tight_layout()

            logs.append(wandb.Image(fig))
            plt.close(fig)
        
        # Step 5: log
        if self.accelerator.is_main_process:
            wandb_tracker.log({tracker_key: logs}, step=self.global_step)
        
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        return logs
