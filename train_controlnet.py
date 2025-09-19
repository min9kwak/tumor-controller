import numpy as np
import json
import yaml
import argparse
import os
from transformers import AutoTokenizer

from config.controlnet import ConfigControlNet
from dataset.brats import BraTSProcessor
from dataset.preprocessing.helper import PromptBuilder
from trainer.controlnet import ControlNetTrainer
from utils.util import set_env


def load_yaml_config(yaml_path):
    """Load configuration from YAML file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config



def parse_arguments_with_yaml():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default='yaml/controlnet.yaml', help='Path to YAML config file')
    config_args, remaining_args = parser.parse_known_args()

    # load YAML config
    yaml_config = load_yaml_config(config_args.config)
    
    # Convert YAML config to command line format
    yaml_args = []
    for key, value in yaml_config.items():
        if value is not None:  # Skip None values
            arg_name = f"--{key}"
            yaml_args.extend([arg_name, str(value)])
    
    # Combine YAML args first, then CLI args (CLI takes precedence)
    combined_args = yaml_args + remaining_args
    
    # create ConfigControlNet object with combined arguments
    args = ConfigControlNet.parse_arguments(combined_args)

    # save config path
    args.config = config_args.config

    return args


def main(args: ConfigControlNet):
    
    # 1. models
    model_names = ['vae', 'text_encoder', 'noise_scheduler']
    trainable_model_names = ['controlnet']
    
    setattr(args, 'model_names', model_names)
    setattr(args, 'trainable_model_names', trainable_model_names)

    # inherit args from ldm_inpaint
    inherit_args = [
        # ddp_parser
        'server', 'pin_memory',
        # data_parser
        'modality', 'n_splits', 'fold_index', 'max_train_samples', 'resolution', 'seed', 'proportion_empty_prompts',
        # train_parser
        'enable_xformers_memory_efficient_attention',
        # model_parser
        'cross_attention_dim', 'tokenizer_name', 'cache_dir',
        'pretrained_model_name_or_path', 'revision', 'variant',
        # task_specific_parser
        'offset_noise', 'dilation_size', 'sigma',
    ]

    # Check if LDM Inpaint checkpoint exists
    args_ldm_path = f"checkpoints/finetune_ldm_inpaint/{args.ldm_inpaint_hash}/configs.json"
    if not os.path.exists(args_ldm_path):
        print(f"❌ Error: LDM Inpaint checkpoint not found!")
        print(f"   Expected path: {args_ldm_path}")
        print(f"   Please check if:")
        print(f"   1. ldm_inpaint_hash is correct: '{args.ldm_inpaint_hash}'")
        print(f"   2. LDM Inpaint training was completed successfully")
        print(f"   3. The checkpoint directory exists")
        raise FileNotFoundError(f"LDM Inpaint config not found: {args_ldm_path}")
    
    with open(args_ldm_path, 'r') as f:
        args_ldm = json.load(f)

    for arg_name in inherit_args:
        if arg_name in args_ldm:
            setattr(args, arg_name, args_ldm[arg_name])
    setattr(args, 'task_previous', args_ldm['task'])
    
    # Print configuration AFTER inheritance
    print(f"🔧 Loading configuration from: {args.config}")
    print(f"📊 Training parameters (after inheritance from LDM Inpaint):")
    print(f"  - Batch size: {args.train_batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Modality: {args.modality}")
    print(f"  - Resolution: {args.resolution}")
    print(f"  - Epochs: {args.num_train_epochs}")
    print(f"  - Mixed precision: {args.mixed_precision}")
    print(f"  - Previous task: {args.task_previous}")
    print()
    
    # 2. load tokenizer and prompt builder
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer',
                                              cache_dir=args.cache_dir, use_fast=False)
    prompt_builder = PromptBuilder()

    # 3. prepare data processor
    processor = BraTSProcessor(config=vars(args), tokenizer=tokenizer, prompt_builder=prompt_builder)
    
    # Add preprocessing parameters to args
    setattr(args, 'low_thresh', processor.low_thresh)
    setattr(args, 'high_thresh', processor.high_thresh)
    
    # 4. create Trainer
    finetuner = ControlNetTrainer(
        args=args,
        model_names=model_names,
        trainable_model_names=trainable_model_names,
        processor=processor
    )

    # 5. save config
    args.save()
    
    # 6. train
    finetuner.train()


if __name__ == "__main__":
    
    from utils.msg import send_telegram
    
    # Parse arguments with YAML config support
    args = parse_arguments_with_yaml()
    args.task = 'controlnet'
    args = set_env(args)

    np.random.seed(args.seed)
    
    # send telegram message when training starts
    try:
        # send telegram message when training starts
        send_telegram(dotenv_path='.env', message=f'Starting {args.task} - {args.experiment_id}.')
        main(args)
        # Completed successfully
        send_telegram(dotenv_path='.env', message=f'✅ Successfully finished {args.task} - {args.experiment_id}.')
    except KeyboardInterrupt:
        # Interrupted by user
        send_telegram(dotenv_path='.env', message=f'⚠️ Training interrupted by user - {args.task} - {args.experiment_id}.')
        raise
    except Exception as e:
        # Other errors
        send_telegram(dotenv_path='.env', message=f'❌ Training failed with error - {args.task} - {args.experiment_id}: {str(e)}')
        raise
