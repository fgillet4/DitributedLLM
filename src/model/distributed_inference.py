"""
Optimized Distributed Inference for DistributedLLM.

This module implements efficient distributed inference capabilities,
combining the tensor parallelism from the original DistributedLLM with
performance optimizations inspired by DeepSpeed.

Features:
1. Distribution of computation across multiple devices and machines
2. Efficient communication patterns between workers
3. KV-cache management for autoregressive generation
4. Support for both local and distributed inference modes
"""

import logging
import time
import threading
import queue
import torch
from typing import Dict, List, Optional, Union, Any, Tuple

from src.model.layers import ShardedModel
from src.worker.communication import CoordinatorClient
from src.model.optimized_inference import OptimizedInference
from src.model.deepspeed_integration import (
    is_deepspeed_available,
    DeepSpeedConfig,
    optimize_with_deepspeed
)

logger = logging.getLogger(__name__)


class DistributedInference:
    """
    Enhanced distributed inference manager for DistributedLLM.
    
    Coordinates token generation across multiple worker machines,
    each potentially running a shard of the model with optimized kernels.
    """
    
    def __init__(
        self,
        coordinator_client: CoordinatorClient,
        model_id: str,
        tokenizer = None,
        max_sequence_length: int = 2048,
        use_deepspeed: bool = True,
        tensor_parallel_size: int = 1,
        dtype: torch.dtype = torch.float16,
        use_cuda_graph: bool = False,
        use_quantization: bool = False,
        quantization_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the distributed inference manager.
        
        Args:
            coordinator_client: Client for communicating with coordinator
            model_id: ID of the model to use
            tokenizer: Optional tokenizer to use (will be loaded if not provided)
            max_sequence_length: Maximum sequence length for generation
            use_deepspeed: Whether to use DeepSpeed optimizations if available
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type for model weights
            use_cuda_graph: Whether to use CUDA graphs for inference
            use_quantization: Whether to use quantization
            quantization_config: Quantization configuration
        """
        self.coordinator_client = coordinator_client
        self.model_id = model_id
        self.max_sequence_length = max_sequence_length
        
        # Set up tokenizer
        self.tokenizer = tokenizer
        
        # DeepSpeed configuration
        self.use_deepspeed = use_deepspeed and is_deepspeed_available()
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.use_cuda_graph = use_cuda_graph
        self.use_quantization = use_quantization
        self.quantization_config = quantization_config
        
        # Request queues
        self.request_queue = queue.Queue()
        self.result_queue = {}  # Map from request_id to result
        
        # Statistics and monitoring
        self.inference_stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_inference_time": 0,
            "tokens_per_second": 0,
            "average_latency": 0,
        }
        
        # Start worker threads
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.worker_thread.start()
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text completions using distributed inference.
        
        Args:
            prompt: Input prompt text or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for repetitive tokens
            do_sample: Whether to use sampling vs greedy decoding
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text completion(s)
        """
        # Create a unique ID for this request
        request_id = f"req_{time.time()}_{hash(str(prompt))}"
        
        # Create result placeholder
        self.result_queue[request_id] = None
        
        # Prepare generation parameters
        params = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            **kwargs
        }
        
        # Add request to queue
        self.request_queue.put((request_id, params))
        
        # Wait for result
        while self.result_queue[request_id] is None:
            time.sleep(0.1)
            
            # Check if processing failed
            if request_id not in self.result_queue:
                raise RuntimeError("Request processing failed")
        
        # Get result
        result = self.result_queue[request_id]
        del self.result_queue[request_id]
        
        # Return single result if input was a single string
        if isinstance(prompt, str) and isinstance(result, list) and len(result) == 1:
            return result[0]
        
        return result
    
    def _process_requests(self):
        """Process requests from the queue and send to coordinator."""
        while self.running:
            try:
                # Get request from queue
                try:
                    request_id, params = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the request
                start_time = time.time()
                
                try:
                    # Tokenize input
                    prompt = params["prompt"]
                    is_batch = isinstance(prompt, list)
                    
                    if is_batch:
                        prompts = prompt
                    else:
                        prompts = [prompt]
                    
                    # Create generation tasks
                    tasks = []
                    for i, prompt_text in enumerate(prompts):
                        task = {
                            "type": "token_generation",
                            "prompt": prompt_text,
                            "params": {
                                k: v for k, v in params.items() 
                                if k not in ["prompt"]
                            },
                            "sequence_index": i
                        }
                        tasks.append(task)
                    
                    # Submit tasks to coordinator
                    task_ids = self.coordinator_client.submit_tasks(tasks)
                    
                    # Wait for tasks to complete
                    results = self.coordinator_client.wait_for_tasks(task_ids)
                    
                    # Process results
                    generated_texts = []
                    for result in sorted(results, key=lambda r: r["sequence_index"]):
                        generated_texts.append(result["text"])
                    
                    # Set result
                    if is_batch:
                        self.result_queue[request_id] = generated_texts
                    else:
                        self.result_queue[request_id] = generated_texts[0] if generated_texts else ""
                    
                    # Update statistics
                    generation_time = time.time() - start_time
                    
                    self.inference_stats["total_requests"] += len(prompts)
                    self.inference_stats["total_inference_time"] += generation_time
                    self.inference_stats["average_latency"] = (
                        (self.inference_stats["average_latency"] * (self.inference_stats["total_requests"] - len(prompts)) +
                         generation_time) / self.inference_stats["total_requests"]
                    )
                    
                    # Estimate tokens generated
                    if "total_tokens_generated" in result:
                        total_new_tokens = sum(r.get("tokens_generated", 0) for r in results)
                    else:
                        # Estimate if not provided
                        total_new_tokens = len(prompts) * params["max_new_tokens"]
                    
                    self.inference_stats["total_tokens_generated"] += total_new_tokens
                    
                    # Calculate tokens per second
                    if generation_time > 0:
                        tokens_per_second = total_new_tokens / generation_time
                        # Exponential moving average for tokens_per_second
                        alpha = 0.1  # Smoothing factor
                        self.inference_stats["tokens_per_second"] = (
                            alpha * tokens_per_second +
                            (1 - alpha) * self.inference_stats["tokens_per_second"]
                        )
                    
                    logger.info(f"Generated response in {generation_time:.2f}s "
                               f"({total_new_tokens / max(generation_time, 0.001):.2f} tokens/sec)")
                
                except Exception as e:
                    logger.error(f"Error processing request {request_id}: {e}")
                    # Set error result
                    if is_batch:
                        self.result_queue[request_id] = [f"Error: {str(e)}"] * len(prompts)
                    else:
                        self.result_queue[request_id] = f"Error: {str(e)}"
                
                finally:
                    # Mark request as processed
                    self.request_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in request processing thread: {e}")
                time.sleep(1)  # Avoid tight loop on persistent errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return self.inference_stats
    
    def stop(self):
        """Stop the inference manager."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)


class LocalDistributedInference:
    """
    Local implementation of DistributedInference for testing and development.
    
    Simulates distributed inference on a single machine using local tensor parallelism,
    while maintaining the same API as the fully distributed version.
    """
    
    def __init__(
        self,
        model: ShardedModel,
        tokenizer,
        tensor_parallel_size: int = 1,
        dtype: torch.dtype = torch.float16,
        use_deepspeed: bool = True,
        use_quantization: bool = False,
        quantization_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize local inference.
        
        Args:
            model: Model to use for inference
            tokenizer: Tokenizer for text processing
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type for model weights
            use_deepspeed: Whether to use DeepSpeed optimizations
            use_quantization: Whether to use quantization
            quantization_config: Quantization configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        
        # Track whether we're using DeepSpeed
        self.using_deepspeed = False
        
        # Apply optimizations
        if use_deepspeed and is_deepspeed_available():
            try:
                # Use DeepSpeed optimizations
                logger.info("Using DeepSpeed optimizations for local inference")
                self.inference_engine = optimize_with_deepspeed(
                    model=model,
                    tokenizer=tokenizer,
                    tensor_parallel_size=tensor_parallel_size,
                    dtype=dtype,
                    use_quantization=use_quantization,
                    quantization_config=quantization_config
                )
                self.using_deepspeed = True
            except Exception as e:
                logger.warning(f"Failed to initialize DeepSpeed: {e}")
                logger.info("Falling back to native optimizations")
                
                # Use native optimizations
                self.inference_engine = OptimizedInference(
                    model=model,
                    tokenizer=tokenizer,
                    dtype=dtype,
                    quantization_config=quantization_config if use_quantization else None
                )
        else:
            # Use native optimizations
            logger.info("Using native optimizations for local inference")
            self.inference_engine = OptimizedInference(
                model=model,
                tokenizer=tokenizer,
                dtype=dtype,
                quantization_config=quantization_config if use_quantization else None
            )
        
        # Statistics
        self.inference_stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_inference_time": 0,
            "tokens_per_second": 0,
        }
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text completions using local optimized inference.
        
        Args:
            prompt: Input prompt text or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for repetitive tokens
            do_sample: Whether to use sampling vs greedy decoding
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text completion(s)
        """
        # Track stats
        start_time = time.time()
        
        # Generate text
        generated_text = self.inference_engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            **kwargs
        )
        
        # Update statistics
        generation_time = time.time() - start_time
        num_prompts = 1 if isinstance(prompt, str) else len(prompt)
        
        self.inference_stats["total_requests"] += num_prompts
        self.inference_stats["total_inference_time"] += generation_time
        
        # Get detailed stats from engine if available
        if hasattr(self.inference_engine, "inference_stats"):
            engine_stats = self.inference_engine.inference_stats
            
            if "total_tokens_generated" in engine_stats:
                # Use stats from engine
                self.inference_stats["total_tokens_generated"] += engine_stats["total_tokens_generated"]
                self.inference_stats["tokens_per_second"] = engine_stats["tokens_per_second"]
            else:
                # Estimate tokens generated
                estimated_tokens = num_prompts * max_new_tokens
                self.inference_stats["total_tokens_generated"] += estimated_tokens
                
                if generation_time > 0:
                    tokens_per_second = estimated_tokens / generation_time
                    # Exponential moving average
                    alpha = 0.1
                    self.inference_stats["tokens_per_second"] = (
                        alpha * tokens_per_second +
                        (1 - alpha) * self.inference_stats["tokens_per_second"]
                    )
        
        return generated_text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = self.inference_stats.copy()
        
        # Add information about DeepSpeed usage
        stats["using_deepspeed"] = self.using_deepspeed
        stats["tensor_parallel_size"] = self.tensor_parallel_size
        
        # Add engine-specific stats if available
        if hasattr(self.inference_engine, "get_stats"):
            engine_stats = self.inference_engine.get_stats()
            stats["engine_stats"] = engine_stats
        
        return stats


class HybridInference:
    """
    Hybrid inference mode that can switch between local and distributed inference.
    
    This class provides automatic fallback to local inference if the coordinator
    is unavailable, and can dynamically switch between modes based on load.
    """
    
    def __init__(
        self,
        model: Optional[ShardedModel] = None,
        tokenizer = None,
        coordinator_client: Optional[CoordinatorClient] = None,
        model_id: str = "",
        prefer_local: bool = False,
        tensor_parallel_size: int = 1,
        dtype: torch.dtype = torch.float16,
        use_deepspeed: bool = True,
        use_quantization: bool = False,
        quantization_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hybrid inference.
        
        Args:
            model: Local model (optional)
            tokenizer: Tokenizer for text processing
            coordinator_client: Client for coordinator communication (optional)
            model_id: ID of the model to use remotely
            prefer_local: Whether to prefer local inference when possible
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type for model weights
            use_deepspeed: Whether to use DeepSpeed optimizations
            use_quantization: Whether to use quantization
            quantization_config: Quantization configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.coordinator_client = coordinator_client
        self.model_id = model_id
        self.prefer_local = prefer_local
        
        # Initialize inference engines
        self.local_inference = None
        self.distributed_inference = None
        
        # Set up local inference if model is provided
        if model is not None:
            self.local_inference = LocalDistributedInference(
                model=model,
                tokenizer=tokenizer,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                use_deepspeed=use_deepspeed,
                use_quantization=use_quantization,
                quantization_config=quantization_config
            )
        
        # Set up distributed inference if coordinator client is provided
        if coordinator_client is not None:
            self.distributed_inference = DistributedInference(
                coordinator_client=coordinator_client,
                model_id=model_id,
                tokenizer=tokenizer,
                use_deepspeed=use_deepspeed,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                use_quantization=use_quantization,
                quantization_config=quantization_config
            )
        
        # Validate that at least one inference mode is available
        if self.local_inference is None and self.distributed_inference is None:
            raise ValueError("Must provide either a model or a coordinator client")
        
        # Performance tracking
        self.performance_history = {
            "local": [],
            "distributed": []
        }
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
        use_local: Optional[bool] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text completions using either local or distributed inference.
        
        Args:
            prompt: Input prompt text or list of prompts
            max_new_tokens: Maximum tokens to generate
            use_local: Force use of local or distributed inference (None for auto)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text completion(s)
        """
        # Determine which inference mode to use
        if use_local is None:
            # Auto mode - decide based on availability and performance
            use_local = self._should_use_local_inference(prompt, max_new_tokens)
        
        # Use specified inference mode
        if use_local and self.local_inference is not None:
            logger.info("Using local inference")
            start_time = time.time()
            
            result = self.local_inference.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            
            # Track performance
            generation_time = time.time() - start_time
            self._update_performance_history("local", generation_time, prompt, max_new_tokens)
            
            return result
        
        elif not use_local and self.distributed_inference is not None:
            logger.info("Using distributed inference")
            start_time = time.time()
            
            result = self.distributed_inference.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            
            # Track performance
            generation_time = time.time() - start_time
            self._update_performance_history("distributed", generation_time, prompt, max_new_tokens)
            
            return result
        
        else:
            # Fallback to whatever is available
            if self.local_inference is not None:
                logger.warning("Falling back to local inference")
                return self.local_inference.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    **kwargs
                )
            else:
                logger.warning("Falling back to distributed inference")
                return self.distributed_inference.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    **kwargs
                )
    
    def _should_use_local_inference(self, prompt, max_new_tokens) -> bool:
        """
        Determine whether to use local or distributed inference.
        
        Makes decision based on:
        1. Availability of inference modes
        2. User preference
        3. Historical performance
        4. Current load
        """
        # Check availability
        if self.local_inference is None:
            return False
        if self.distributed_inference is None:
            return True
        
        # Check user preference
        if self.prefer_local:
            return True
        
        # Check performance history
        if len(self.performance_history["local"]) >= 5 and len(self.performance_history["distributed"]) >= 5:
            # Calculate average performance
            local_avg = sum(self.performance_history["local"][-5:]) / 5
            distributed_avg = sum(self.performance_history["distributed"][-5:]) / 5
            
            # Use the faster mode with a 10% margin to avoid oscillation
            if local_avg < distributed_avg * 0.9:
                return True
            elif distributed_avg < local_avg * 0.9:
                return False
        
        # Default based on input size
        num_prompts = 1 if isinstance(prompt, str) else len(prompt)
        total_tokens = num_prompts * max_new_tokens
        
        # For small generations, prefer local
        return total_tokens < 1000
    
    def _update_performance_history(self, mode, time_taken, prompt, max_new_tokens):
        """Update performance history for a specific inference mode."""
        num_prompts = 1 if isinstance(prompt, str) else len(prompt)
        total_tokens = num_prompts * max_new_tokens
        
        # Normalize by tokens generated for fair comparison
        normalized_time = time_taken / total_tokens
        
        # Add to history (keep last 10 entries)
        self.performance_history[mode].append(normalized_time)
        if len(self.performance_history[mode]) > 10:
            self.performance_history[mode].pop(0)
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get inference statistics for both local and distributed modes."""
        stats = {
            "local": self.local_inference.get_stats() if self.local_inference else None,
            "distributed": self.distributed_inference.get_stats() if self.distributed_inference else None,
            "performance_history": self.performance_history
        }
        return stats