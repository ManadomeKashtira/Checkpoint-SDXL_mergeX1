import os
import sys
import argparse
import torch
from datetime import datetime
from model_merger import ModelMerger

MERGE_METHODS = {
    "WS": "weighted_sum",
    "SIG": "sigmoid",
    "ADD": "add_difference",
    "SUM": "sum_twice",
    "MUL": "multiply_difference",
    "SMOOTH": "smooth_add_difference",
    "GEOMEAN": "geometric",
    "TENSOR": "tensor_sum",
    "TRIPLE": "triple_sum",
    "SG": "squared_geometric",
    "COS": "cosine",
    "QUAD": "quadratic",
    "CB": "cubic",
    "EXP": "exponential",
    "LOG": "logarithmic",
    "INV": "inverse",
    "MW": "magnitude_weighted",
    "GAU": "gaussian",
    "CF": "crossfade",
    "ADV": "adaptive",
    "SGEO": "squared_geometric",
    "EL": "elastic",
    "HAR": "harmonic",
    "SMX" "softmax"
}

PRECISION_TYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "prune": "prune"  # Special handling for pruned models
}

def create_parser():
    parser = argparse.ArgumentParser(description="Memory-Optimized SDXL Checkpoint Merger")
    
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='2 or 3 model paths to merge (safetensors format)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output path for merged model'
    )
    
    parser.add_argument(
        '--method',
        default='WS',
        choices=MERGE_METHODS.keys(),
        help='Merge method (WS=weighted sum, SIG=sigmoid, etc.)'
    )
    
    parser.add_argument(
        '--precision',
        default='fp16',
        choices=PRECISION_TYPES.keys(),
        help='Model precision (fp16, bf16, fp32, or prune)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='Alpha ratio (0-1)'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=0.0,
        help='Beta ratio for third model (0-1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (default: timestamp-based)'
    )
    
    parser.add_argument(
        '--vae-source',
        default='first',
        choices=['first', 'second', 'last'],
        help='Which model to take VAE from'
    )
    
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu', 'tpu'],
        help='Device to use for merging'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=4,
        help='Number of tensors to process at once (lower = less RAM usage)'
    )
    
    parser.add_argument(
        '--prune-threshold',
        type=float,
        default=0.1,
        help='Threshold for pruning (only used with --precision prune)'
    )

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate device
    if args.device == 'tpu':
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        except ImportError:
            print("TPU support requires torch_xla. Falling back to CPU.")
            args.device = 'cpu'
    
    # Setup configuration
    config = {
        "model_paths": args.models,
        "output_path": args.output,
        "merge_method": MERGE_METHODS[args.method],
        "precision": args.precision,
        "alpha": args.alpha,
        "beta": args.beta,
        "merge_seed": args.seed if args.seed is not None else int(datetime.now().timestamp()),
        "vae_source": args.vae_source,
        "device": args.device,
        "chunk_size": args.chunk_size,
        "prune_threshold": args.prune_threshold
    }
    
    print("\nMerge Configuration:")
    print(f"Method: {args.method}")
    print(f"Precision: {args.precision}")
    print(f"Alpha: {args.alpha}")
    print(f"Beta: {args.beta}")
    print(f"Chunk Size: {args.chunk_size}")
    if args.precision == 'prune':
        print(f"Prune Threshold: {args.prune_threshold}")
    
    merger = ModelMerger(config)
    success = merger.merge()
    
    if success:
        print(f"\nMerge completed successfully!")
        print(f"Output saved to: {args.output}")
    else:
        print("\nMerge failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
