import os
import json
import tqdm
import pandas as pd


class PromptBuilder:
    
    def __init__(self):
        pass

    def classify_tumor_size(self, pixels: int) -> str:
        if pixels <= 1400:
            return "small tumor"
        elif pixels <= 1700:
            return "mild tumor"
        elif pixels <= 2000:
            return "medium-sized tumor"
        elif pixels <= 2300:
            return "moderate tumor"
        else:
            return "large tumor"

    def generate_prompt(self, modality: str, age: int | None, tumor_pixels: int, force_healthy: bool = False) -> str:
        modality_str = {
            "t1ce": "A contrast-enhanced T1-weighted brain MRI",
            "t2": "A T2-weighted brain MRI"
        }.get(modality.lower(), "A brain MRI")

        age_desc = f"{age}-year-old" if age is not None else "unknown age"

        if force_healthy or tumor_pixels == 0:
            subject_desc = f"a {age_desc} healthy individual"
        else:
            size_desc = self.classify_tumor_size(tumor_pixels)
            subject_desc = f"a {age_desc} patient with a {size_desc}"

        return f"{modality_str} of {subject_desc}."

    def reverse_prompt(self, prompt: str, tumor_size: str = "medium-sized") -> str:
        if "healthy individual" in prompt:
            return prompt.replace("healthy individual", f"patient with a {tumor_size} tumor")
        
        elif "patient with a" in prompt:
            return prompt.replace("patient with a small tumor", "healthy individual") \
                         .replace("patient with a mild tumor", "healthy individual") \
                         .replace("patient with a medium-sized tumor", "healthy individual") \
                         .replace("patient with a moderate tumor", "healthy individual") \
                         .replace("patient with a large tumor", "healthy individual")
        
        else:
            print(f"Failed to reverse prompt: {prompt}")
            return prompt
