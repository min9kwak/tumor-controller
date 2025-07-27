# -*- coding: utf-8 -*-

import argparse
from config.base import ConfigBase, str2bool, str2tuple, handle_none


class ConfigLDMInpaint(ConfigBase):
    def __init__(self, args: argparse.Namespace = None, **kwargs):
        super(ConfigLDM, self).__init__(args, **kwargs)

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

        # additional arguments
        # parser.add_argument('--argument', type=type, default=default, help='')

        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        parser = ConfigBase.logging_parser()

        # additional arguments
        # parser.add_argument('--argument', type=type, default=default, help='')

        return parser
    
    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        
        parser = argparse.ArgumentParser("LDM", add_help=False)
        
        # LDM specific arguments
        parser.add_argument('--offset_noise', type=str2bool, default=True,
                            help="Whether to add offset noise during training.")
        parser.add_argument('--strength', type=float, default=0.8,
                            help="Strength for image-to-image translation.")
        parser.add_argument('--guidance_scale', type=float, default=7.5,
                            help="Guidance scale for inference.")
        parser.add_argument('--num_inference_steps', type=int, default=20,
                            help="Number of inference steps.")
        parser.add_argument('--tumor_size', type=str, default="medium-sized",
                            help="Tumor size for validation prompts.")
        parser.add_argument('--dilation_size', type=int, default=5, help="Dilation size for masking/inpainting.")
        parser.add_argument('--sigma', type=float, default=5.0, help="Sigma for Gaussian blur.")
        
        return parser
