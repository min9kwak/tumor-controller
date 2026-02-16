import random
import re


class PromptBuilder:
    def __init__(self):
        self.templates_healthy = [
            "{modality} of a {age_desc} healthy individual.",
            "A {age_desc} healthy person undergoing {modality}.",
            "{modality} image of a healthy brain (age: {age_desc}).",
        ]
        self.templates_tumor = [
            "{modality} of a {age_desc} patient with a {size_desc} tumor.",
            "A {age_desc} patient's {modality} scan showing a {size_desc} tumor.",
            "{modality} image showing a {size_desc} brain tumor in a {age_desc} patient.",
        ]

    def classify_tumor_size(self, pixels: int) -> str:
        if pixels <= 1400:
            return "small"
        elif pixels <= 1700:
            return "mild"
        elif pixels <= 2000:
            return "medium-sized"
        elif pixels <= 2300:
            return "moderate"
        else:
            return "large"

    def generate_prompt(self, modality: str, age: int | None, tumor_size: int, force_healthy: bool = False) -> str:
        modality_str = {
            "t1ce": "contrast-enhanced T1-weighted brain MRI",
            "t2": "T2-weighted brain MRI"
        }.get(modality.lower(), "brain MRI")
        age_desc = f"{age}-year-old" if age is not None else "unknown-age"

        if force_healthy or tumor_size == 0:
            template = random.choice(self.templates_healthy)
            return template.format(modality=modality_str, age_desc=age_desc)
        else:
            size_desc = self.classify_tumor_size(tumor_size)
            template = random.choice(self.templates_tumor)
            return template.format(modality=modality_str, age_desc=age_desc, size_desc=size_desc)

    def reverse_prompt(self, prompt: str) -> str:
        original_prompt = prompt
        
        # healthy → tumor
        if "healthy" in prompt:
            # Pattern 1: "A {age} healthy person undergoing {modality}"
            result = re.sub(r"A ([^.]+) healthy person undergoing ([^.]+)", 
                          r"A \1 patient's \2 scan showing a medium-sized tumor", prompt)
            if result != prompt:
                return result
            
            # Pattern 2: "{modality} of a {age} healthy individual"
            result = re.sub(r"([^.]+) of a ([^.]+) healthy individual", 
                          r"\1 of a \2 patient with a medium-sized tumor", prompt)
            if result != prompt:
                return result
            
            # Pattern 3: "{modality} image of a healthy brain (age: {age})"
            result = re.sub(r"([^.]+) image of a healthy brain \(age: ([^)]+)\)", 
                          r"\1 image showing a medium-sized brain tumor in a \2 patient", prompt)
            if result != prompt:
                return result
            
            # General fallback 1: "healthy individual/person"
            result = re.sub(r"healthy (individual|person)", "patient with a medium-sized tumor", prompt)
            if result != prompt:
                return result
            
            # General fallback 2: "healthy brain"
            result = re.sub(r"healthy brain", "brain with a medium-sized tumor", prompt)
            if result != prompt:
                return result

        # tumor → healthy  
        elif "tumor" in prompt or "patient" in prompt:
            # Pattern 1: "{modality} image showing a {size} brain tumor in a {age} patient"
            result = re.sub(r"([^.]+) image showing a (small|mild|medium-sized|moderate|large) brain tumor in a ([^.]+) patient", 
                          r"\1 image of a healthy brain (age: \3)", prompt)
            if result != prompt:
                return result
            
            # Pattern 2: "A {age} patient's {modality} scan showing a {size} tumor"
            result = re.sub(r"A ([^.]+) patient's ([^.]+) scan showing a (small|mild|medium-sized|moderate|large) tumor", 
                          r"A \1 healthy person undergoing \2", prompt)
            if result != prompt:
                return result
            
            # Pattern 3: "{modality} of a {age} patient with a {size} tumor"
            result = re.sub(r"([^.]+) of a ([^.]+) patient with a (small|mild|medium-sized|moderate|large) tumor", 
                          r"\1 of a \2 healthy individual", prompt)
            if result != prompt:
                return result
            
            # General fallback 1: "patient with a {size} tumor"
            result = re.sub(r"patient with a (small|mild|medium-sized|moderate|large) tumor", 
                          "healthy individual", prompt)
            if result != prompt:
                return result
            
            # General fallback 2: "{image/scan} showing a {size} tumor"
            result = re.sub(r"(image|scan) showing a (small|mild|medium-sized|moderate|large) (brain )?tumor", 
                          r"\1 of a healthy brain", prompt)
            if result != prompt:
                return result
            
            # General fallback 3: "brain with a {size} tumor"
            result = re.sub(r"brain with a (small|mild|medium-sized|moderate|large) tumor", 
                          "healthy brain", prompt)
            if result != prompt:
                return result
            
            # General fallback 4: remaining "patient"
            result = re.sub(r"patient", "healthy individual", prompt)
            if result != prompt:
                return result
        
        print(f"[Warning] Reverse failed: {original_prompt}")
        return original_prompt

if __name__ == '__main__':
    
    from accelerate import Accelerator
    from dataset.transforms import create_transforms, create_mask_transforms
    from dataset.brats import BraTSProcessor, BraTSDataset
    from dataset.preprocessing.helper import PromptBuilder
    from transformers import AutoTokenizer
    from utils.util import set_env
    from easydict import EasyDict as edict
    
    
    # 0. set environment
    config = {'proportion_empty_prompts': 0.0,
              'modality': 't2',
              'n_splits': 10,
              'fold_index': 0,
              'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
              'resolution': 512,
              'seed': 2025}
    
    config = set_env(config)
    config = edict(config)
    
    # 1. prepare dataset
    tokenizer = AutoTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5',
                                              subfolder='tokenizer',
                                              cache_dir=config['cache_dir'],
                                              use_fast=True)
    prompt_builder = PromptBuilder()
    accelerator = Accelerator()
    
    processor = BraTSProcessor(config=config, tokenizer=tokenizer, prompt_builder=prompt_builder)
    
    with accelerator.main_process_first():
        data_info = processor.process()
    
    tumor_info = data_info['train']['tumor']
    healthy_info = data_info['train']['healthy']
    
    # 2. create dataset
    image_transform_val, _ = create_transforms(resolution=512, train=False)
    dilate_transform, _ = create_mask_transforms(resolution=512, dilation_size=5, sigma=5.0)
    
    # Create single dataset with all samples
    train_set = BraTSDataset(tumor_info + healthy_info, image_transform=image_transform_val, dilate_transform=dilate_transform, return_keys=['image', 'dilate', 'prompt'])
    
    # Get indices for tumor and healthy samples
    tumor_indices = [i for i, d in enumerate(train_set.data_info) if int(d['label']) == 1]
    healthy_indices = [i for i, d in enumerate(train_set.data_info) if int(d['label']) == 0]
    
    # check reverse prompt
    print("=== Tumor samples ===")
    for i in range(min(20, len(tumor_indices))):
        idx = tumor_indices[i]
        print(f"Original: {train_set[idx]['prompt']}")
        print(f"Reversed: {processor.prompt_builder.reverse_prompt(train_set[idx]['prompt'])}")
        print("-" * 30)

    print("=== Healthy samples ===")
    for i in range(min(20, len(healthy_indices))):
        idx = healthy_indices[i]
        print(f"Original: {train_set[idx]['prompt']}")
        print(f"Reversed: {processor.prompt_builder.reverse_prompt(train_set[idx]['prompt'])}")
        print("-" * 30)
