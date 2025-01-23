import os
import json
import torch
import random
import hashlib
import uuid
import logging
from datetime import datetime
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import gc
import numpy as np
from typing import List, Dict, Any, Optional

class ModelMerger:
    """
    Advanced SDXL Model Merger with Comprehensive Error Tracking and Diagnostics
    
    Features:
    - Memory-optimized tensor merging
    - Multiple merging methods
    - Detailed error logging
    - Comprehensive merge statistics
    """
    
    # Existing merge methods from previous implementation
    MERGE_METHODS = {
        "weighted_sum": lambda tensors, ratios: sum(t * r for t, r in zip(tensors, ratios)),
        "sigmoid": lambda tensors, ratios: torch.sigmoid(sum(t * r for t, r in zip(tensors[1:], ratios[1:]))) * tensors[0],
        "add_difference": lambda tensors, ratios: tensors[0] + sum((t - tensors[0]) * r for t, r in zip(tensors[1:], ratios[1:])),
        "sum_twice": lambda tensors, ratios: sum(tensors[0] + t * r for t, r in zip(tensors[1:], ratios[1:])) / len(tensors[1:]),
        "multiply_difference": lambda tensors, ratios: tensors[0] * (1 + sum((t/tensors[0] - 1) * r for t, r in zip(tensors[1:], ratios[1:]))),
        "smooth_add_difference": lambda tensors, ratios: tensors[0] + torch.tanh(sum((t - tensors[0]) * r for t, r in zip(tensors[1:], ratios[1:]))),
        "geometric": lambda tensors, ratios: torch.prod(torch.stack([torch.pow(t, r) for t, r in zip(tensors, ratios)]), dim=0),
        "tensor_sum": lambda tensors, ratios: torch.stack([t * r for t, r in zip(tensors, ratios)]).sum(0),
        "triple_sum": lambda tensors, ratios: sum(t * r for t, r in zip(tensors, ratios)) / torch.norm(tensors[0]),
        "quadratic": lambda tensors, ratios: tensors[0] + sum((t - tensors[0]) * r * r for t, r in zip(tensors[1:], ratios[1:])),
        "cubic": lambda tensors, ratios: tensors[0] + sum((t - tensors[0]) * r * r * r for t, r in zip(tensors[1:], ratios[1:])),
        "exponential": lambda tensors, ratios: tensors[0] + sum((t - tensors[0]) * torch.exp(torch.tensor(r)) for t, r in zip(tensors[1:], ratios[1:])),
        "logarithmic": lambda tensors, ratios: tensors[0] + sum((t - tensors[0]) * torch.log1p(torch.tensor(r)) for t, r in zip(tensors[1:], ratios[1:])),
        "cosine": lambda tensors, ratios: tensors[0] + sum((t - tensors[0]) * torch.cos(torch.tensor(r * np.pi/2)) for t, r in zip(tensors[1:], ratios[1:])),
        "inverse": lambda tensors, ratios: sum(t * (1/r if r > 0 else 1) for t, r in zip(tensors, ratios)) / sum(1/r if r > 0 else 1 for r in ratios),
        "softmax": lambda tensors, ratios: sum(t * torch.softmax(torch.tensor(ratios), dim=0)[i] for i, t in enumerate(tensors)),
        "harmonic": lambda tensors, ratios: len(tensors) / sum(1/(t * r + 1e-8) for t, r in zip(tensors, ratios)),
        "elastic": lambda tensors, ratios: tensors[0] + torch.sinh(sum((t - tensors[0]) * r for t, r in zip(tensors[1:], ratios[1:]))),
        "squared_geometric": lambda tensors, ratios: torch.prod(torch.stack([torch.pow(t, r*r) for t, r in zip(tensors, ratios)]), dim=0),
        "adaptive": lambda tensors, ratios: sum(t * (r / (torch.norm(t) + 1e-8)) for t, r in zip(tensors, ratios)),
        "crossfade": lambda tensors, ratios: sum(t * torch.sigmoid(torch.tensor(r * 10 - 5)) for t, r in zip(tensors, ratios)),
        "gaussian": lambda tensors, ratios: sum(t * torch.exp(-torch.tensor((1-r)**2 * 4)) for t, r in zip(tensors, ratios)),
        "magnitude_weighted": lambda tensors, ratios: sum(t * r * torch.norm(t) for t, r in zip(tensors, ratios)) / sum(torch.norm(t) for t in tensors)
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelMerger with advanced configuration and logging
        
        Args:
            config (Dict[str, Any]): Configuration dictionary for model merging
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler('model_merge.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Configuration and initialization
        self.config = config
        self.merge_method = self.MERGE_METHODS[config["merge_method"]]
        self.precision = config["precision"]
        self.chunk_size = config.get("chunk_size", 100)
        self.device = self._setup_device(config.get("device", "cuda"))
        
        # Enhanced metadata initialization
        self.merge_metadata = {
            "merge_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "source_models": config["model_paths"],
            "merge_method": config["merge_method"],
            "merge_parameters": {
                "alpha": config.get("alpha", 0.5),
                "beta": config.get("beta", 0),
                "precision": config["precision"]
            },
            "model_checksums": {},
            "merge_stats": {
                "total_tensors": 0,
                "merged_tensors": 0,
                "error_tensors": 0,
                "skipped_tensors": 0
            },
            "error_details": []
        }
        
        # Random seed management
        self._set_random_seeds(config.get("merge_seed", 42))
        
        # Calculate merge ratios
        self.num_models = len(config["model_paths"])
        self.ratios = self._calculate_merge_ratios(config)

    def _set_random_seeds(self, seed: int):
        """Set consistent random seeds for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _calculate_merge_ratios(self, config: Dict[str, Any]) -> List[float]:
        """
        Calculate merge ratios based on number of models
        
        Args:
            config (Dict[str, Any]): Merge configuration
        
        Returns:
            List[float]: Calculated merge ratios
        """
        if self.num_models == 2:
            return [1 - config["alpha"], config["alpha"]]
        else:
            return [1 - config["alpha"] - config["beta"], config["alpha"], config["beta"]]
   
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of the model file"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
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

    def merge_tensors(
        self, 
        tensors: List[torch.Tensor], 
        component_type: str, 
        key: str
    ) -> Optional[torch.Tensor]:
        """
        Memory-efficient and robust tensor merging with comprehensive error handling
        
        Args:
            tensors (List[torch.Tensor]): Tensors to merge
            component_type (str): Type of model component
            key (str): Tensor key for identification
        
        Returns:
            Optional[torch.Tensor]: Merged tensor or None if merging fails
        """
        try:
            # VAE special handling
            if component_type == 'VAE':
                vae_source = self.config.get("vae_source", "first")
                vae_index_map = {"first": 0, "second": 1, "last": -1}
                return tensors[vae_index_map[vae_source]].clone()

            # Precision and device conversion
            dtype = self._get_dtype()
            device_tensors = [t.to(self.device, dtype=dtype) for t in tensors]

            # Merge tensors
            merged = self.merge_method(device_tensors, self.ratios)

            # Optional pruning
            if self.precision == "prune":
                threshold = self.config.get("prune_threshold", 1e-4)
                mask = torch.abs(merged) > threshold
                merged = merged * mask

            # Post-processing
            merged_result = merged.cpu().to(dtype)
            
            # Update merge statistics
            self.merge_metadata["merge_stats"]["merged_tensors"] += 1
            
            return merged_result

        except Exception as e:
            # Comprehensive error tracking
            error_info = {
                "key": key,
                "component_type": component_type,
                "error_message": str(e),
                "tensor_shapes": [t.shape for t in tensors]
            }
            
            self.logger.error(f"Tensor Merge Error: {error_info}")
            self.merge_metadata["merge_stats"]["error_tensors"] += 1
            self.merge_metadata["error_details"].append(error_info)
            
            return None

    def merge(self) -> bool:
      try:
          # Model checksums and initial diagnostics
          self._calculate_model_checksums()
        
          # Load models
          models = self._load_models()
        
          # Tensor key management
          all_keys = list(models[0].keys())
          total_tensors = len(all_keys)
          self.merge_metadata["merge_stats"]["total_tensors"] = total_tensors
        
          self.logger.info(f"Starting merge of {total_tensors} tensors")
        
          # Chunk-based processing
          for i in tqdm(range(0, total_tensors, self.chunk_size), desc="Merging"):
              chunk_keys = all_keys[i:i + self.chunk_size]
              chunk_result = self._process_tensor_chunk(chunk_keys, models)
            
            # Save chunk immediately and clear from memory
              self._save_merged_model(chunk_result)
              del chunk_result
            
               # Memory management
              gc.collect()
              if torch.cuda.is_available():
                   torch.cuda.empty_cache()
        
            # Save detailed metadata
          self._save_merge_metadata()
          self._log_merge_summary()
        
          return True

      except Exception as e:
           self.logger.error(f"Comprehensive merge failure: {e}")
           return False

    def _calculate_model_checksums(self):
        """Calculate and store SHA256 checksums for source models"""
        for path in self.config["model_paths"]:
            self.merge_metadata["model_checksums"][path] = self._calculate_model_hash(path)

    def _load_models(self) -> List[Dict[str, torch.Tensor]]:
        """
        Load source models with diagnostic logging
        
        Returns:
            List[Dict[str, torch.Tensor]]: Loaded model tensors
        """
        models = []
        for path in self.config["model_paths"]:
            model = load_file(path)
            models.append(model)
            self.logger.info(f"Loaded model: {path} with {len(model)} tensors")
        return models

    def _process_tensor_chunk(
        self, 
        chunk_keys: List[str], 
        models: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a chunk of tensors with robust error handling
        
        Args:
            chunk_keys (List[str]): Keys to process in this chunk
            models (List[Dict[str, torch.Tensor]]): Source models
        
        Returns:
            Dict[str, torch.Tensor]: Successfully merged tensors
        """
        chunk_result = {}
        for key in chunk_keys:
            if key not in models[0]:
                self.merge_metadata["merge_stats"]["skipped_tensors"] += 1
                continue

            component_type = self.get_component_type(key)
            tensors = [m.get(key, torch.zeros_like(models[0][key])) for m in models]
            
            merged_tensor = self.merge_tensors(tensors, component_type, key)
            if merged_tensor is not None:
                chunk_result[key] = merged_tensor

        return chunk_result

    def _save_merged_model(self, merged_result: Dict[str, torch.Tensor]):
        """Save merged tensors using checkpoint-based method"""
        try:
            if os.path.exists(self.config["output_path"]):
                existing = load_file(self.config["output_path"])
                existing.update(merged_result)
                save_file(existing, self.config["output_path"])
            else:
                 save_file(merged_result, self.config["output_path"])
        
            self.logger.info(f"Merged model saved to: {self.config['output_path']}")
        except Exception as e:
            self.logger.error(f"Error saving merged model: {e}") 

    def _save_merge_metadata(self):
        """Save comprehensive merge metadata"""
        metadata_path = f"{os.path.splitext(self.config['output_path'])[0]}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.merge_metadata, f, indent=2)
        self.logger.info(f"Metadata saved to: {metadata_path}")

    def _log_merge_summary(self):
        """Log a comprehensive summary of the merge operation"""
        stats = self.merge_metadata["merge_stats"]
        self.logger.info("Merge Operation Summary:")
        self.logger.info(f"Total Tensors: {stats['total_tensors']}")
        self.logger.info(f"Merged Tensors: {stats['merged_tensors']}")
        self.logger.info(f"Error Tensors: {stats['error_tensors']}")
        self.logger.info(f"Skipped Tensors: {stats['skipped_tensors']}")
  
    def get_component_type(self, key: str) -> str:
        """Identify SDXL component type"""
        if 'model.diffusion_model' in key:
            return 'UNET'
        elif 'first_stage_model' in key:
            return 'VAE'
        elif any(x in key for x in ['text_model', 'transformer', 'encoder']):
            return 'TEXT_ENCODER'
        return 'OTHER'
