import numpy as np
import yaml
import argparse
import os
from transformers import AutoTokenizer

from config.ldm import ConfigLDM
from dataset.brats import BraTSProcessor
from trainer.ldm import LDMFineTuner
from utils.util import set_env

# TODO: trainable model -> unet only

# TODO: move to config/base.py
def load_yaml_config(yaml_path):
    """Load configuration from YAML file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config



def parse_arguments_with_yaml():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default='yaml/ldm.yaml',
                        help='Path to YAML config file')
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
    
    # create ConfigLDM object with combined arguments
    args = ConfigLDM.parse_arguments(combined_args)

    # save config path
    args.config = config_args.config

    return args


def main(args: ConfigLDM):
    
    print(f"🔧 Loading configuration from: {args.config}")
    print(f"📊 Training parameters:")
    print(f"  - Batch size: {args.train_batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Modality: {args.modality}")
    print(f"  - Resolution: {args.resolution}")
    print(f"  - Epochs: {args.num_train_epochs}")
    print(f"  - Mixed precision: {args.mixed_precision}")
    print()
    
    # 2. models
    model_names = ['vae', 'unet', 'text_encoder', 'noise_scheduler']
    # trainable_model_names = ['vae', 'unet']  # Train both for medical domain adaptation
    trainable_model_names = ['unet']  # Train both for medical domain adaptation
    
    setattr(args, 'model_names', model_names)
    setattr(args, 'trainable_model_names', trainable_model_names)
    
    # 0. save config
    args.save()
    
    # 1. load tokenizer for preparing datasets
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder='tokenizer',
        cache_dir=args.cache_dir,
        use_fast=False
    )
    
    # 2. prepare data processor
    processor = BraTSProcessor(config=vars(args), tokenizer=tokenizer)
    
    # 3. create Trainer
    finetuner = LDMFineTuner(
        args=args,
        model_names=model_names,
        trainable_model_names=trainable_model_names,
        processor=processor
    )
    
    # 4. train
    finetuner.train()


if __name__ == "__main__":
    
    from utils.msg import send_telegram
    
    # Parse arguments with YAML config support
    args = parse_arguments_with_yaml()
    args.task = 'finetune_ldm'
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
