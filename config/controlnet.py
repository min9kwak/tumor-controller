# -*- coding: utf-8 -*-

import os
import copy
import json
import argparse
import datetime
from config.base import ConfigBase, str2bool, str2tuple, handle_none


class ConfigControlNet(ConfigBase):
    def __init__(self, args: argparse.Namespace = None, **kwargs):
        super(ConfigControlNet, self).__init__(args, **kwargs)

    @staticmethod
    def ddp_parser() -> argparse.ArgumentParser:
        parser = ConfigBase.ddp_parser()

        # additional arguments
        # parser.add_argument('--argument', type=type, default=default, help='')

        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        parser = ConfigBase.data_parser()

        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        parser = ConfigBase.train_parser()

        # additional arguments
        # parser.add_argument('--argument', type=type, default=default, help='')

        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        parser = ConfigBase.model_parser()

        parser.add_argument('--ldm_inpaint_hash', type=str, default='', help="Hash of LDM-Inpaint model.")
        parser.add_argument('--checkpoint_num', type=int, default=None, help="Checkpoint number.")
        
        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        parser = ConfigBase.logging_parser()

        parser.add_argument('--tracker_project_name', type=str, default="controlnet",
                            help="The `project_name` argument passed to Accelerator.init_trackers. "
                                 "For more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator")

        return parser
    
    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("LDM", add_help=False)

        # additional arguments
        # parser.add_argument('--argument', type=type, default=default, help='')
        parser.add_argument('--guidance_scale', type=float, default=7.5, help="Guidance scale for inference.")
        parser.add_argument('--num_inference_steps', type=int, default=30, help="Number of inference steps.")
        parser.add_argument('--strength', type=float, default=1.0, help="Strength for inference.")
        parser.add_argument('--dilation_size', type=int, default=5, help="Dilation size for masking/inpainting.")
        parser.add_argument('--sigma', type=float, default=5.0, help="Sigma for Gaussian blur.")
        parser.add_argument('--tumor_size', type=str, default="medium-sized", help="Tumor size for validation prompts.")
        
        # parser.add_argument('--controlnet_mask', type=str2bool, default=True, help="Whether to use controlnet mask.")
        # parser.add_argument('--controlnet_edge', type=str2bool, default=True, help="Whether to use controlnet edge.")

        parser.add_argument('--blur_edge', type=str2bool, default=False, help="Whether to blur edge.")
        parser.add_argument('--controlnet_conditioning_scale', type=float, default=0.8, help="ControlNet conditioning scale.")

        return parser
