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

        parser.add_argument('--controlnet_model_name_or_path', type=str, default='lllyasviel/sd-controlnet-canny',
                            help="Path to pretrained controlnet model or model identifier from huggingface.co/models.")
        
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

        # additional arguments
        # parser.add_argument('--argument', type=type, default=default, help='')
        parser.add_argument('--guidance_scale', type=float, default=7.5, help="Guidance scale for inference.")
        parser.add_argument('--num_inference_steps', type=int, default=30, help="Number of inference steps.")
        parser.add_argument('--strength', type=float, default=0.8, help="Strength for inference.")
        parser.add_argument('--dilation_size', type=int, default=5, help="Dilation size for masking/inpainting.")
        parser.add_argument('--sigma', type=float, default=5.0, help="Sigma for Gaussian blur.")

        return parser
