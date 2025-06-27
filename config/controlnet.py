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

        parser.add_argument('--proportion_empty_prompts', type=float, default=0,
                            help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")        
        
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

        # additional arguments
        # parser.add_argument('--argument', type=type, default=default, help='')

        return parser
