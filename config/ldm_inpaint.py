# -*- coding: utf-8 -*-

import argparse
from config.base import ConfigBase, str2bool, str2tuple, handle_none


class ConfigLDMInpaint(ConfigBase):
    def __init__(self, args: argparse.Namespace = None, **kwargs):
        super(ConfigLDMInpaint, self).__init__(args, **kwargs)

    @staticmethod
    def ddp_parser() -> argparse.ArgumentParser:
        parser = ConfigBase.ddp_parser()

        # additional arguments
        # parser.add_argument('--argument', type=type, default=default, help='')

        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        parser = ConfigBase.data_parser()

        # additional arguments
        # parser.add_argument('--argument', type=type, default=default, help='')

        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        parser = ConfigBase.train_parser()

        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        parser = ConfigBase.model_parser()

        parser.add_argument('--pretrained_model_name_or_path', type=str, default='runwayml/stable-diffusion-v1-5',
                            help="Path to pretrained model or model identifier from huggingface.co/models.")        
        parser.add_argument('--revision', type=str, default=None,
                            help="Revision of pretrained model identifier from huggingface.co/models.")
        parser.add_argument('--variant', type=str, default=None,
                            help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
        
        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        parser = ConfigBase.logging_parser()

        parser.add_argument('--tracker_project_name', type=str, default="finetune_ldm",
                            help="The `project_name` argument passed to Accelerator.init_trackers. "
                                 "For more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator")

        return parser
    
    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        
        parser = argparse.ArgumentParser("LDM", add_help=False)
        
        # LDM specific arguments
        parser.add_argument('--offset_noise', type=str2bool, default=True,
                            help="Whether to add offset noise during training.")
        parser.add_argument('--guidance_scale', type=float, default=7.5, help="Guidance scale for inference.")
        parser.add_argument('--strength', type=float, default=1.0, help="Strength for inference.")
        parser.add_argument('--num_inference_steps', type=int, default=30, help="Number of inference steps.")
        parser.add_argument('--tumor_size', type=str, default="medium-sized", help="Tumor size for validation prompts.")
        parser.add_argument('--dilation_size', type=int, default=5, help="Dilation size for masking/inpainting.")
        parser.add_argument('--sigma', type=float, default=5.0, help="Sigma for Gaussian blur.")
        # TODO: tumor_size: small, mild, medium-sized, moderate, large -> reverse_prompt
        
        return parser
