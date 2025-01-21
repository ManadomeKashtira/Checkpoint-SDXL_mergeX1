import os
import json
import torch
import random
from datetime import datetime
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import gc
import numpy as np
from typing import List, Dict, Any

class ModelMerger:
    """Memory-optimized SDXL Model Merger"""
    
    MERGE_METHODS = {
        "weighted_sum": lambda tensors, ratios: sum(t * r for t, r in zip(tensors, ratios)),
        "sigmoid": lambda tensors, ratios: torch.sigmoid(sum(t * r for t, r in zip(tensors[1:], ratios[1:]))) * tensors[0],
        "add_difference": lambda tensors, ratios: tensors[0] + sum((t - tensors[0]) * r for t, r in zip(tensors[1:], ratios[1:])),
        "sum_twice": lambda tensors, ratios: sum(tensors[0] + t * r for t, r in zip(tensors[1:], ratios[1:])) / len(tensors[1:]),
        "multiply_difference": lambda tensors, ratios: tensors[0] * (1 + sum((t/tensors[0] - 1) * r for t, r in zip(tensors[1:], ratios[1:]))),
        "smooth_add_difference": lambda tensors, ratios: tensors[0] + torch.tanh(sum((t - tensors[0]) * r for t, r in zip(tensors[1:], ratios[1:]))),
        "geometric": lambda tensors, ratios: torch.prod(torch.stack([torch.pow(t, r) for t, r in zip(tensors, ratios)]), dim=0),
        "tensor_sum": lambda tensors, ratios: torch.stack([t * r for t, r in zip(tensors, ratios)]).sum(0),
        "triple_sum": lambda tensors, ratios: sum(t * r for t, r in zip(tensors, ratios)) / torch.norm(tensors[0])
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.merge_method = self.MERGE_METHODS[config["merge_method"]]
        self.precision = config["precision"]
        self.chunk_size = config["chunk_size"]
        self.device = self._setup_device(config["device"])
        
        # Set random seed
        torch.manual_seed(config["merge_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config["merge_seed"])
        
        # Calculate ratios
        self.num_models = len(config["model_paths"])
        if self.num_models == 2:
            self.ratios = [1 - config["alpha"], config["alpha"]]
        else:
            self.ratios = [1 - config["alpha"] - config["beta"], config["alpha"], config["beta"]]

    def _setup_device(self, device_name: str) -> torch.device:
        """Setup computation device"""
        if device_name == 'tpu':
            try:
                import torch_xla.core.xla_model as xm
                return xm.xla_device()
            except ImportError:
                print("TPU support not available. Falling back to CPU.")
                return torch.device('cpu')
        return torch.device(device_name)

    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype based on precision setting"""
        if self.precision == "prune":
            return torch.float32
        return {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }[self.precision]

    def merge_tensors(self, tensors: List[torch.Tensor], component_type: str, key: str) -> torch.Tensor:
        """Memory-efficient tensor merging"""
        try:
            # Handle VAE
            if component_type == 'VAE':
                vae_idx = {"first": 0, "second": 1, "last": -1}[self.config["vae_source"]]
                return tensors[vae_idx].clone()

            # Convert to target precision
            dtype = self._get_dtype()
            device_tensors = [t.to(self.device, dtype=dtype) for t in tensors]

            # Merge
            merged = self.merge_method(device_tensors, self.ratios)

            # Handle pruning if specified
            if self.precision == "prune":
                threshold = self.config["prune_threshold"]
                mask = torch.abs(merged) > threshold
                merged = merged * mask

            # Move back to CPU and convert to target precision
            merged = merged.cpu().to(dtype)

            return merged

        except Exception as e:
            print(f"Error merging {key}: {str(e)}")
            return tensors[0].clone()

    def process_chunk(self, chunk_keys: List[str], models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Process a chunk of tensors"""
        chunk_result = {}
        for key in chunk_keys:
            if key not in models[0]:
                continue

            component_type = self.get_component_type(key)
            tensors = [m.get(key, torch.zeros_like(models[0][key])) for m in models]
            merged_tensor = self.merge_tensors(tensors, component_type, key)
            chunk_result[key] = merged_tensor

        return chunk_result

    def merge(self) -> bool:
        """Perform memory-efficient model merge"""
        try:
            # Load model keys first
            print("\nLoading model keys...")
            models = []
            for path in self.config["model_paths"]:
                models.append(load_file(path))
                print(f"Loaded: {path}")

            # Get all keys
            all_keys = list(models[0].keys())
            total_tensors = len(all_keys)
            
            # Process in chunks
            print(f"\nMerging {total_tensors} tensors in chunks of {self.chunk_size}...")
            for i in tqdm(range(0, total_tensors, self.chunk_size)):
                chunk_keys = all_keys[i:i + self.chunk_size]
                chunk_result = self.process_chunk(chunk_keys, models)
                
                # Save chunk
                self._save_checkpoint(chunk_result)
                
                # Clear memory
                del chunk_result
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return True

        except Exception as e:
            print(f"Error during merge: {str(e)}")
            return False

    def _save_checkpoint(self, merged_tensors: Dict[str, torch.Tensor]) -> None:
        """Save merged tensors"""
        try:
            if os.path.exists(self.config["output_path"]):
                existing = load_file(self.config["output_path"])
                existing.update(merged_tensors)
                save_file(existing, self.config["output_path"])
            else:
                save_file(merged_tensors, self.config["output_path"])
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            raise

    def get_component_type(self, key: str) -> str:
        """Identify SDXL component type"""
        if 'model.diffusion_model' in key:
            return 'UNET'
        elif 'first_stage_model' in key:
            return 'VAE'
        elif any(x in key for x in ['text_model', 'transformer', 'encoder']):
            return 'TEXT_ENCODER'
        return 'OTHER'
