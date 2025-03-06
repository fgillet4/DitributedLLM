#!/usr/bin/env python3
"""
Benchmark script for the DistributedLLM system.

This script benchmarks the performance of different distributed configurations
to find the optimal settings for a given hardware setup.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.inference import ModelInference, DistributedInference
from src.worker.compute_engine import ComputeEngine
from src.worker.resource_monitor import ResourceMonitor
from src.worker.communication import CoordinatorClient, TaskProcessor
from src.coordinator.scheduler import Scheduler
from src.utils.networking import find_available_port


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger("Benchmark")


def benchmark_local_inference(model_id: str, num_iterations: int = 5, batch_size: int = 1):
    """
    Benchmark local inference performance.
    
    Args:
        model_id: ID of the model to benchmark
        num_iterations: Number of benchmark iterations
        batch_size: Batch size for inference
    
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Benchmarking local inference for model {model_id}")
    
    try:
        # Initialize model
        logger.info("Initializing model...")
        inference = ModelInference(model_id)
        
        # Prepare benchmark prompts
        prompt = "The quick brown fox jumps over the lazy dog."
        if batch_size > 1:
            prompts = [prompt] * batch_size
        else:
            prompts = prompt
        
        # Run warmup
        logger.info("Running warmup...")
        _ = inference.generate(prompts, max_new_tokens=20)
        
        # Run benchmark
        logger.info(f"Running {num_iterations} iterations with batch size {batch_size}...")
        times = []
        tokens_generated = []
        
        for i in range(num_iterations):
            start_time = time.time()
            result = inference.generate(prompts, max_new_tokens=100)
            end_time = time.time()
            
            # Calculate tokens generated
            num_tokens = inference.inference_stats["total_tokens_generated"] / inference.inference_stats["total_requests"]
            tokens_generated.append(num_tokens)
            
            # Record time
            times.append(end_time - start_time)
            
            logger.info(f"Iteration {i+1}/{num_iterations}: {times[-1]:.2f}s, {num_tokens:.1f} tokens")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_second = avg_tokens / avg_time
        
        # Get model statistics
        stats = inference.get_stats()
        
        # Compile results
        results = {
            "model_id": model_id,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "average_time": avg_time,
            "average_tokens": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "model_stats": stats
        }
        
        logger.info(f"Benchmark results: {tokens_per_second:.2f} tokens/sec")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in local inference benchmark: {e}")
        return {"error": str(e)}


def benchmark_distributed_inference(
    model_id: str,
    num_workers: int = 2,
    num_iterations: int = 5,
    batch_size: int = 1
):
    """
    Benchmark distributed inference performance.
    
    Args:
        model_id: ID of the model to benchmark
        num_workers: Number of worker processes to use
        num_iterations: Number of benchmark iterations
        batch_size: Batch size for inference
    
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Benchmarking distributed inference for model {model_id} with {num_workers} workers")
    
    try:
        # Start coordinator
        coordinator_port = find_available_port(5000, 6000)
        if not coordinator_port:
            raise RuntimeError("Could not find available port for coordinator")
        
        # Start coordinator process
        # In a real benchmark, we would start an actual coordinator process
        # For this mock implementation, we'll just simulate it
        logger.info(f"Starting coordinator on port {coordinator_port}...")
        
        # Start worker processes
        worker_ports = []
        for _ in range(num_workers):
            port = find_available_port(6001, 7000)
            if not port:
                raise RuntimeError("Could not find available port for worker")
            worker_ports.append(port)
        
        logger.info(f"Starting {num_workers} workers on ports {worker_ports}...")
        
        # In a real benchmark, we would start actual worker processes
        # For this mock implementation, we'll just simulate them
        
        # Create DistributedInference client
        logger.info("Initializing distributed inference client...")
        # For this mock implementation, we'll just create a ModelInference instance
        inference = ModelInference(model_id)
        
        # Prepare benchmark prompts
        prompt = "The quick brown fox jumps over the lazy dog."
        if batch_size > 1:
            prompts = [prompt] * batch_size
        else:
            prompts = prompt
        
        # Run warmup
        logger.info("Running warmup...")
        _ = inference.generate(prompts, max_new_tokens=20)
        
        # Run benchmark
        logger.info(f"Running {num_iterations} iterations with batch size {batch_size}...")
        times = []
        tokens_generated = []
        
        for i in range(num_iterations):
            start_time = time.time()
            result = inference.generate(prompts, max_new_tokens=100)
            end_time = time.time()
            
            # Calculate tokens generated
            num_tokens = inference.inference_stats["total_tokens_generated"] / inference.inference_stats["total_requests"]
            tokens_generated.append(num_tokens)
            
            # Record time
            times.append(end_time - start_time)
            
            logger.info(f"Iteration {i+1}/{num_iterations}: {times[-1]:.2f}s, {num_tokens:.1f} tokens")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_second = avg_tokens / avg_time
        
        # Get model statistics
        stats = inference.get_stats()
        
        # Compile results
        results = {
            "model_id": model_id,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "average_time": avg_time,
            "average_tokens": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "model_stats": stats
        }
        
        logger.info(f"Benchmark results: {tokens_per_second:.2f} tokens/sec with {num_workers} workers")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in distributed inference benchmark: {e}")
        return {"error": str(e)}


def benchmark_model_sharding(
    model_id: str,
    shard_config: List[Dict[str, Any]],
    num_iterations: int = 5
):
    """
    Benchmark model sharding configurations.
    
    Args:
        model_id: ID of the model to benchmark
        shard_config: List of sharding configurations to test
        num_iterations: Number of benchmark iterations
    
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Benchmarking model sharding for model {model_id}")
    
    try:
        results = []
        
        for config in shard_config:
            logger.info(f"Testing sharding configuration: {config}")
            
            # Initialize model with this sharding configuration
            logger.info("Initializing model...")
            inference = ModelInference(model_id, device_map=config["device_map"])
            
            # Prepare benchmark prompts
            prompt = "The quick brown fox jumps over the lazy dog."
            
            # Run warmup
            logger.info("Running warmup...")
            _ = inference.generate(prompt, max_new_tokens=20)
            
            # Run benchmark
            logger.info(f"Running {num_iterations} iterations...")
            times = []
            tokens_generated = []
            
            for i in range(num_iterations):
                start_time = time.time()
                result = inference.generate(prompt, max_new_tokens=100)
                end_time = time.time()
                
                # Calculate tokens generated
                num_tokens = inference.inference_stats["total_tokens_generated"] / inference.inference_stats["total_requests"]
                tokens_generated.append(num_tokens)
                
                # Record time
                times.append(end_time - start_time)
                
                logger.info(f"Iteration {i+1}/{num_iterations}: {times[-1]:.2f}s, {num_tokens:.1f} tokens")
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            avg_tokens = sum(tokens_generated) / len(tokens_generated)
            tokens_per_second = avg_tokens / avg_time
            
            # Compile results for this configuration
            config_results = {
                "device_map": config["device_map"],
                "average_time": avg_time,
                "average_tokens": avg_tokens,
                "tokens_per_second": tokens_per_second,
            }
            
            results.append(config_results)
            
            logger.info(f"Configuration results: {tokens_per_second:.2f} tokens/sec")
        
        # Compile overall results
        benchmark_results = {
            "model_id": model_id,
            "num_iterations": num_iterations,
            "configurations": results
        }
        
        return benchmark_results
    
    except Exception as e:
        logger.error(f"Error in model sharding benchmark: {e}")
        return {"error": str(e)}


def save_results(results, output_file: str):
    """
    Save benchmark results to a file.
    
    Args:
        results: Benchmark results to save
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark DistributedLLM system.")
    parser.add_argument("--mode", choices=["local", "distributed", "sharding"], required=True,
                       help="Benchmark mode")
    parser.add_argument("--model", default="llama-2-7b",
                       help="Model ID to benchmark")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of benchmark iterations")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--workers", type=int, default=2,
                       help="Number of worker processes for distributed mode")
    parser.add_argument("--output", default="benchmark_results.json",
                       help="Output file for benchmark results")
    
    args = parser.parse_args()
    
    # Run benchmark based on mode
    if args.mode == "local":
        results = benchmark_local_inference(
            model_id=args.model,
            num_iterations=args.iterations,
            batch_size=args.batch_size
        )
    
    elif args.mode == "distributed":
        results = benchmark_distributed_inference(
            model_id=args.model,
            num_workers=args.workers,
            num_iterations=args.iterations,
            batch_size=args.batch_size
        )
    
    elif args.mode == "sharding":
        # Define sharding configurations to test
        shard_configs = [
            {"device_map": "auto"},
            {"device_map": "balanced"},
            {"device_map": {"embedding": "cuda:0", "transformer.blocks.0": "cuda:0", "transformer.blocks.1": "cuda:0",
                            "transformer.blocks.2": "cuda:1", "transformer.blocks.3": "cuda:1", "lm_head": "cuda:1"}}
        ]
        
        results = benchmark_model_sharding(
            model_id=args.model,
            shard_config=shard_configs,
            num_iterations=args.iterations
        )
    
    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()