#!/usr/bin/env python3
"""
Embeddings Generator using Jina Embeddings v3 or other Hugging Face models.

This script generates embeddings for text using Hugging Face models.
It's designed to be called from Node.js as a subprocess.
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Union

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import numpy as np
except ImportError:
    print("Required packages not found. Please install them with:")
    print("pip install sentence-transformers torch numpy")
    sys.exit(1)


def get_embedding(
    text: str,
    model_name: str = "jinaai/jina-embeddings-v3"
) -> List[float]:
    """
    Generate embeddings for text using the specified model.
    
    Args:
        text: The text to embed
        model_name: The Hugging Face model name to use
        
    Returns:
        A list of floats representing the embedding vector
    """
    try:
        # Load model
        model = SentenceTransformer(model_name)
        
        # Generate embedding
        embedding = model.encode(text, normalize_embeddings=True)
        
        # Convert to Python list
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().detach().numpy()
        
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}", file=sys.stderr)
        raise


def main():
    """Main function to handle CLI arguments and generate embeddings."""
    parser = argparse.ArgumentParser(description="Generate embeddings for text")
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--model", 
        default="jinaai/jina-embeddings-v3", 
        help="Hugging Face model name to use"
    )
    args = parser.parse_args()
    
    try:
        # Read input text
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Generate embedding
        embedding = get_embedding(text, args.model)
        
        # Save to output file
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"embedding": embedding}, f)
        
        print(f"Embedding saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())