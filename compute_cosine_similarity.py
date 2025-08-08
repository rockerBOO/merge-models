#!/usr/bin/env python3
"""
Compute cosine similarity between weights of two SafeTensors models
"""

import argparse
import torch
import safetensors.torch
from tqdm import tqdm

def compute_cosine_similarity(model1_path: str, model2_path: str, verbose: bool = True):
    """
    Compute cosine similarity between weights of two models
    
    Args:
    - model1_path: Path to first SafeTensors model
    - model2_path: Path to second SafeTensors model
    - verbose: Enable detailed output
    
    Returns:
    - Dictionary of layer-wise cosine similarities
    """
    # Open models safely without loading all tensors
    with safetensors.safe_open(model1_path, framework="pt") as f1, \
         safetensors.safe_open(model2_path, framework="pt") as f2:
        
        # Get common keys between models
        common_keys = list(set(f1.keys()) & set(f2.keys()))
        
        # Compute similarities
        similarities = {}
        overall_similarities = []
        
        # Use tqdm for progress tracking
        for key in tqdm(common_keys, desc="Computing Cosine Similarities"):
            try:
                # Load tensors only when needed
                w1 = f1.get_tensor(key)
                w2 = f2.get_tensor(key)
                
                # Flatten weights
                w1 = w1.flatten()
                w2 = w2.flatten()
                
                # Ensure same length
                min_len = min(len(w1), len(w2))
                w1, w2 = w1[:min_len], w2[:min_len]
                
                # Compute cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    w1.unsqueeze(0), w2.unsqueeze(0), dim=1
                ).item()
                
                similarities[key] = cos_sim
                overall_similarities.append(cos_sim)
                
                # Explicitly delete tensors to free memory
                del w1, w2
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Error processing key {key}: {e}")
                continue
    
    # Compute overall statistics
    result = {
        'layer_similarities': similarities,
        'mean_similarity': torch.mean(torch.tensor(overall_similarities)).item(),
        'median_similarity': torch.median(torch.tensor(overall_similarities)).item(),
        'min_similarity': min(overall_similarities),
        'max_similarity': max(overall_similarities)
    }
    
    # Verbose output
    if verbose:
        print("\nCosine Similarity Results:")
        print(f"Total layers analyzed: {len(similarities)}")
        print(f"Mean similarity: {result['mean_similarity']:.4f}")
        print(f"Median similarity: {result['median_similarity']:.4f}")
        print(f"Min similarity: {result['min_similarity']:.4f}")
        print(f"Max similarity: {result['max_similarity']:.4f}")
        
        # Top and bottom 5 layers
        sorted_layers = sorted(similarities.items(), key=lambda x: x[1])
        print("\nTop 5 most similar layers:")
        for key, sim in sorted_layers[-5:]:
            print(f"{key}: {sim:.4f}")
        
        print("\nBottom 5 least similar layers:")
        for key, sim in sorted_layers[:5]:
            print(f"{key}: {sim:.4f}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Compute cosine similarity between model weights")
    parser.add_argument("model1", help="Path to first SafeTensors model")
    parser.add_argument("model2", help="Path to second SafeTensors model")
    parser.add_argument("--output", "-o", help="Path to output JSON result")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Compute similarities
    similarities = compute_cosine_similarity(
        args.model1, 
        args.model2, 
        verbose=not args.quiet
    )
    
    # Optional output to JSON
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(similarities, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()
