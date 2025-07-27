import os
import json
import tqdm
import pandas as pd
import random


class PromptBuilder:
    def __init__(self):
        # define various templates (healthy/tumor)
        self.templates_healthy = [
            "{modality} of a {age_desc} healthy individual.",
            "A {age_desc} healthy person undergoing {modality}.",
            "{modality} image of a healthy brain (age: {age_desc}).",
        ]
        self.templates_tumor = [
            "{modality} of a {age_desc} patient with a {size_desc} tumor.",
            "A {age_desc} patient’s {modality} scan showing a {size_desc} tumor.",
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
        # modality description
        modality_str = {
            "t1ce": "contrast-enhanced T1-weighted brain MRI",
            "t2": "T2-weighted brain MRI"
        }.get(modality.lower(), "brain MRI")

        # age description
        age_desc = f"{age}-year-old" if age is not None else "unknown-age"

        # healthy vs tumor template
        if force_healthy or tumor_size == 0:
            template = random.choice(self.templates_healthy)
            return template.format(modality=modality_str, age_desc=age_desc)
        else:
            size_desc = self.classify_tumor_size(tumor_size)
            template = random.choice(self.templates_tumor)
            return template.format(modality=modality_str, age_desc=age_desc, size_desc=size_desc)

    def reverse_prompt(self, prompt: str) -> str:
        # if "healthy" is included, change to tumor (change to medium-sized tumor)
        if "healthy" in prompt:
            return prompt.replace("healthy individual", "patient with a medium-sized tumor") \
                         .replace("healthy person", "patient with a medium-sized tumor") \
                         .replace("healthy brain", "brain with a medium-sized tumor")
        # if "tumor" is included, change to healthy
        elif "tumor" in prompt or "patient" in prompt:
            return prompt.replace("patient with a small tumor", "healthy individual") \
                         .replace("patient with a mild tumor", "healthy individual") \
                         .replace("patient with a medium-sized tumor", "healthy individual") \
                         .replace("patient with a moderate tumor", "healthy individual") \
                         .replace("patient with a large tumor", "healthy individual") \
                         .replace("brain with a", "healthy brain")
        else:
            print(f"[Warning] Reverse failed: {prompt}")
            return prompt
