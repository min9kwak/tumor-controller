# -*- coding: utf-8 -*-

import os
import copy
import json
import argparse
import datetime


def str2bool(string: str):
    """
    string to boolean
    ex) parser.add_argument('--argument', type=str2bool, default=True)
    """
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2tuple(string: str):
    return tuple(int(item) for item in string.split(','))


def handle_none(string: str):
    if string.lower() == "none":
        return None
    else:
        return string


class ConfigBase(object):
    def __init__(self, args: argparse.Namespace = None, **kwargs):

        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()

        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)

        if not hasattr(self, 'experiment_id'):
            self.experiment_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self._task = None

    @classmethod
    def parse_arguments(cls, args=None):
        """Create a configuration object from command line arguments."""
        parents = [
            cls.ddp_parser(),
            cls.data_parser(),
            cls.model_parser(),
            cls.train_parser(),
            cls.logging_parser(),
            cls.task_specific_parser(),
        ]

        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args

        config = cls()
        parser.parse_args(args=args, namespace=config)  # sets parsed arguments as attributes of namespace

        return config

    @classmethod
    def from_json(cls, json_path: str):
        """Create a configuration object from a .json file."""
        with open(json_path, 'r') as f:
            configs = json.load(f)

        return cls(args=configs)

    def save(self, path: str = None):
        """Save configurations to a .json file."""
        if path is None:
            save_path = os.path.join(self.output_dir, 'configs.json')
        else:
            save_path = os.path.join(self.output_dir, f'configs_{path}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs['task'] = self.task
        attrs['output_dir'] = self.output_dir

        with open(save_path, 'w') as f:
            json.dump(attrs, f, indent=2)

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        self._task = value

    @property
    def output_dir(self) -> str:
        """The output directory where the model predictions and checkpoints will be written."""
        out = os.path.join('checkpoints', self.task, self.experiment_id)
        os.makedirs(out, exist_ok=True)
        return out
    
    @property
    def logging_dir(self) -> str:
        """[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."""
        log = os.path.join(self.output_dir, 'logs')
        os.makedirs(log, exist_ok=True)
        return log

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    @staticmethod
    def ddp_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Data Distributed Training", add_help=False)
        parser.add_argument('--server', type=str, default='psc', choices=('local', 'psc'))
        parser.add_argument('--pin_memory', type=str2bool, default=False)
        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--modality', type=str, default='t1ce+t2', choices=('t1ce', 't2', 't1ce+t2'),
                            help="The modality of the dataset to train on.")
        parser.add_argument('--n_splits', type=int, default=10,
                            help="The number of splits for the dataset.")
        parser.add_argument('--fold_index', type=int, default=0,
                            help="The index of the fold for the dataset.")
        parser.add_argument('--max_train_samples', type=int, default=None,
                            help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
        parser.add_argument('--resolution', type=int, default=512,
                            help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
        parser.add_argument('--seed', type=int, default=2025, help="Random seed that will be set at the beginning of training.")
        parser.add_argument('--proportion_empty_prompts', type=float, default=0,
                            help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")
        
        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument('--train_batch_size', type=int, default=4, help="Batch size (per device) for the training dataloader.")
        parser.add_argument('--num_train_epochs', type=int, default=1, help="Number of training epochs.")
        parser.add_argument('--max_train_steps', type=int, default=None,
                            help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
        parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                            help="Whether to resume from the checkpoint directory. "
                                 "Use a path saved by `--checkpointing_steps`, or `'latest'` to automatically select the last available checkpoint.")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--gradient_checkpointing', type=str2bool, default=False,
                            help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
        parser.add_argument('--learning_rate', type=float, default=5e-6, help="Initial learning rate (after the warmup period) to use.")
        parser.add_argument('--scale_lr', type=str2bool, default=False,
                            help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
        parser.add_argument('--lr_scheduler', type=str, default="constant",
                            choices=("linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"),
                            help="The scheduler type to use.")
        parser.add_argument('--lr_warmup_steps', type=int, default=500,
                            help="Number of steps for the warmup.")
        parser.add_argument('--lr_num_cycles', type=int, default=1,
                            help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
        parser.add_argument('--lr_power', type=float, default=1.0,
                            help="Power factor of the polynomial scheduler.")
        parser.add_argument('--use_8bit_adam', type=str2bool, default=False,
                            help="Whether or not to use 8-bit Adam from bitsandbytes.")
        parser.add_argument('--dataloader_num_workers', type=int, default=0,
                            help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
        parser.add_argument('--adam_beta1', type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        parser.add_argument('--adam_beta2', type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
        parser.add_argument('--adam_weight_decay', type=float, default=1e-2, help="Weight decay to use.")
        parser.add_argument('--adam_epsilon', type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
        parser.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm.")
        parser.add_argument('--allow_tf32', type=str2bool, default=False,
                            help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information,"
                                  " see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"))
        parser.add_argument('--report_to', type=str, default="wandb", choices=("tensorboard", "wandb", "all"),
                            help="The integration to report the results and logs to. Supported platforms are `tensorboard`, `wandb`, and `all`.")
        parser.add_argument('--mixed_precision', type=str, default=None, choices=("no", "fp16", "bf16"),
                            help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                                  " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                                  " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."))
        parser.add_argument('--enable_xformers_memory_efficient_attention', type=str2bool, default=False,
                            help="Whether or not to use xformers.")
        parser.add_argument('--set_grads_to_none', type=str2bool, default=False,
                            help=("Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
                                  " behaviors, so disable this argument if it causes any problems. More info:"
                                  " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"))
        parser.add_argument('--validation_steps', type=int, default=100, help="Run validation every X steps.")
        parser.add_argument('--logging_steps', type=int, default=20, help="Log training loss and metrics every X steps.")
        parser.add_argument('--num_validation_samples', type=int, default=4,
                            help="Number of validation samples to use.")
        
        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("Model Configuration", add_help=False)

        parser.add_argument('--pretrained_model_name_or_path', type=str, default='runwayml/stable-diffusion-v1-5',
                            help="Path to pretrained model or model identifier from huggingface.co/models.")        
        parser.add_argument('--revision', type=str, default=None,
                            help="Revision of pretrained model identifier from huggingface.co/models.")
        parser.add_argument('--variant', type=str, default=None,
                            help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
        parser.add_argument('--cross_attention_dim', type=int, default=768,
                            help="Cross-attention dimension for the UNet model.")
        parser.add_argument('--tokenizer_name', type=str, default=None,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument('--cache_dir', type=str, default="/ocean/projects/med230010p/mkwak/models",
                            help="Path to the directory where the downloaded models and datasets will be stored.")
        
        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        
        # move to task-specific parser
        parser.add_argument('--checkpointing_steps', type=int, default=500,
                            help="Save a checkpoint of the training state every X updates.")
        parser.add_argument('--checkpoints_total_limit', type=int, default=None,
                            help="Max number of checkpoints to store.")
        parser.add_argument('--tracker_project_name', type=str, default="finetune_ldm",
                            help="The `project_name` argument passed to Accelerator.init_trackers. "
                                 "For more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator")
        
        return parser
