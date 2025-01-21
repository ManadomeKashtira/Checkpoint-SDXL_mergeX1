import os
import json
import torch
import random
from datetime import datetime
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import gc
import numpy as np

class ModelMerger:
    MERGE_METHODS = {
        "weighted_sum": lambda tensors, ratios: sum(t * r for t, r in zip(tensors, ratios)),
        "sigmoid": lambda tensors, ratios: torch.sigmoid(sum(t * r for t, r in zip(tensors[1:], ratios[1:]))) * tensors[0],
        "geometric": lambda tensors, ratios: torch.prod(torch.stack([torch.pow(t, r) for t, r in zip(tensors, ratios)]), dim=0),
        "max": lambda tensors, ratios: torch.max(torch.stack([t * r for t, r in zip(tensors, ratios)]), dim=0)[0],
        "add_difference": lambda tensors, ratios: tensors[0] + sum((t - tensors[0]) * r for t, r in zip(tensors[1:], ratios[1:])),
        "smooth_add_difference": lambda tensors, ratios: tensors[0] + torch.tanh(sum((t - tensors[0]) * r for t, r in zip(tensors[1:], ratios[1:]))),
        "multiply_difference": lambda tensors, ratios: tensors[0] * (1 + sum((t/tensors[0] - 1) * r for t, r in zip(tensors[1:], ratios[1:]))),
        "similarity": lambda tensors, ratios: (sum(r * torch.cosine_similarity(tensors[0], t, dim=0).unsqueeze(-1) * t for t, r in zip(tensors, ratios))) / sum(ratios),
        "train_difference": lambda tensors, ratios: tensors[0] + sum(torch.sign(t - tensors[0]) * r for t, r in zip(tensors[1:], ratios[1:])),
        "triple_sum": lambda tensors, ratios: sum(t * r for t, r in zip(tensors, ratios)) / torch.norm(tensors[0]),
        "tensor_sum": lambda tensors, ratios: torch.stack([t * r for t, r in zip(tensors, ratios)]).sum(0),
        "sum_twice": lambda tensors, ratios: sum(tensors[0] + t * r for t, r in zip(tensors[1:], ratios[1:])) / len(tensors[1:])
    }

    def __init__(self, config):
        self.config = config
        self.merge_method = self.MERGE_METHODS[config["merge_method"]]
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        random.seed(config["merge_seed"])
        
        self.num_models = len(config["model_paths"])
        if self.num_models == 2:
            self.ratios = [1 - config["alpha"], config["alpha"]]
        else:
            self.ratios = [1 - config["alpha"] - config["beta"], config["alpha"], config["beta"]]

    def get_component_type(self, key):
        if key.startswith('model.diffusion_model'):
            return 'UNET'
        elif key.startswith('first_stage_model'):
            return 'VAE'
        elif key.startswith('transformer_'):
            return 'TEXT_ENCODER'
        else:
            return 'OTHER'

    def normalize_tensor_weights(self, tensor, reference_tensor):
        """Normalize tensor weights to match reference tensor statistics"""
        with torch.no_grad():
            ref_mean = reference_tensor.mean()
            ref_std = reference_tensor.std()
            current_mean = tensor.mean()
            current_std = tensor.std()
            
            # Normalize and scale
            normalized = (tensor - current_mean) / (current_std + 1e-8)
            scaled = normalized * ref_std + ref_mean
            
            return scaled

    def merge_tensors(self, tensors, component_type, key):
        """Merge tensors with proper normalization and range preservation"""
        if component_type == 'VAE':
            vae_idx = {"first": 0, "second": 1, "last": -1}[self.config["vae_source"]]
            return tensors[vae_idx].clone()

        try:
            # Move tensors to device
            device_tensors = [t.to(self.device) for t in tensors]
            reference_stats = device_tensors[0].clone()
            
            # Perform merge
            merged = self.merge_method(device_tensors, self.ratios)
            
            # Normalize to match reference statistics
            merged = self.normalize_tensor_weights(merged, reference_stats)
            
            # Additional handling for specific layers
            if 'weight' in key or 'bias' in key:
                if merged.dim() > 1:  # For weight matrices
                    orig_norm = torch.norm(reference_stats, dim=1, keepdim=True)
                    merged_norm = torch.norm(merged, dim=1, keepdim=True)
                    scale_factor = orig_norm / (merged_norm + 1e-8)
                    merged = merged * scale_factor

            return merged.cpu()

        except Exception as e:
            print(f"Error merging tensor {key}: {str(e)}")
            return tensors[0].clone()

    def merge(self):
        try:
            # Load models
            models = [load_file(path) for path in self.config["model_paths"]]
            all_keys = set(models[0].keys())
            merged_model = {}
            
            print(f"\nMerging {len(self.config['model_paths'])} models...")
            print(f"Merge method: {self.config['merge_method']}")
            print(f"Ratios: {self.ratios}")
            
            # Process tensors
            for key in tqdm(all_keys, desc="Merging tensors"):
                if key not in models[0]:
                    continue

                component_type = self.get_component_type(key)
                tensors = [m.get(key, torch.zeros_like(models[0][key])) for m in models]
                
                merged_tensor = self.merge_tensors(tensors, component_type, key)
                merged_model[key] = merged_tensor

                # Periodic saving to manage memory
                if len(merged_model) >= 500:
                    if os.path.exists(self.config["output_path"]):
                        existing = load_file(self.config["output_path"])
                        existing.update(merged_model)
                        save_file(existing, self.config["output_path"])
                    else:
                        save_file(merged_model, self.config["output_path"])
                    merged_model.clear()
                    gc.collect()
                    torch.cuda.empty_cache()

            # Save remaining tensors
            if merged_model:
                if os.path.exists(self.config["output_path"]):
                    existing = load_file(self.config["output_path"])
                    existing.update(merged_model)
                    save_file(existing, self.config["output_path"])
                else:
                    save_file(merged_model, self.config["output_path"])

            print(f"\nMerge completed! Model saved at: {self.config['output_path']}")
            return True
            
        except Exception as e:
            print(f"Error during merge: {str(e)}")
            return False
