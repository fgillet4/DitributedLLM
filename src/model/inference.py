"""
Inference utilities for DistributedLLM.

Provides functionality for running inference on language models
in a distributed manner across multiple devices and machines.
"""

import logging
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
import numpy as np

from src.model.layers import ShardedModel, ShardedModelLoader
from src.model.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class ModelInference:
    """
    Manages distributed inference for large language models.
    
    This class handles:
    1. Token generation across distributed model components
    2. Batching and sequencing of requests
    3. Memory management and caching
    4. Coordination of multi-device inference
    """
    
    def __init__(
        self,
        model_id: str,
        device_map: Optional[Union[str, Dict[str, str]]] = "auto",
        cache_dir: Optional[str] = None,
        max_batch_size: int = 8,
        max_sequence_length: int = 2048,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the inference manager.
        
        Args:
            model_id: ID of the model to load
            device_map: How to distribute model components across devices
            cache_dir: Directory for caching model weights
            max_batch_size: Maximum batch size for inference
            max_sequence_length: Maximum sequence length for generation
            dtype: Data type to use for model weights and activation
        """
        self.model_id = model_id
        self.device_map = device_map
        self.cache_dir = cache_dir
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        
        # Load model
        logger.info(f"Loading model {model_id} for inference...")
        self.model = self._load_model()
        
        # Load tokenizer
        self.tokenizer = Tokenizer(model_id, cache_dir=cache_dir)
        
        # Inference settings
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "max_new_tokens": 256,
        }
        
        # KV-cache for efficient generation
        self.kv_cache = {}
        
        # Statistics and monitoring
        self.inference_stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_inference_time": 0,
            "tokens_per_second": 0,
        }
    
    def _load_model(self) -> ShardedModel:
        """Load the model with the specified device mapping."""
        # Get model configuration and structure
        model_info = ShardedModelLoader.load_model(self.model_id, self.device_map)
        
        # Create the actual ShardedModel instance
        model = ShardedModel(model_info["config"], model_info["device_map"])
        
        # Set model to eval mode
        model.eval()
        
        return model
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate text completions based on the input prompt.
        
        Args:
            prompt: Input prompt text or list of prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for token repetition
            do_sample: Whether to use sampling or greedy decoding
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated text completions
        """
        # Update generation config with any provided parameters
        generation_config = self.generation_config.copy()
        if max_new_tokens is not None:
            generation_config["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            generation_config["temperature"] = temperature
        if top_p is not None:
            generation_config["top_p"] = top_p
        if top_k is not None:
            generation_config["top_k"] = top_k
        if repetition_penalty is not None:
            generation_config["repetition_penalty"] = repetition_penalty
        if do_sample is not None:
            generation_config["do_sample"] = do_sample
        
        # Convert single prompt to list
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        
        # Tokenize prompts
        input_tokens = self.tokenizer.encode_batch(prompts)
        
        # Run generation
        start_time = time.time()
        
        # Generate token IDs
        generated_ids = self._generate_tokens(input_tokens, generation_config)
        
        # Decode generated tokens
        generated_texts = self.tokenizer.decode_batch(generated_ids)
        
        # Update statistics
        generation_time = time.time() - start_time
        num_tokens_generated = sum(len(ids) - len(input_ids) for ids, input_ids in zip(generated_ids, input_tokens))
        
        self.inference_stats["total_requests"] += len(prompts)
        self.inference_stats["total_tokens_generated"] += num_tokens_generated
        self.inference_stats["total_inference_time"] += generation_time
        self.inference_stats["tokens_per_second"] = num_tokens_generated / generation_time
        
        logger.info(f"Generated {num_tokens_generated} tokens in {generation_time:.2f}s ({num_tokens_generated / generation_time:.2f} tokens/sec)")
        
        return generated_texts
    
    def _generate_tokens(
        self,
        input_tokens: List[List[int]],
        generation_config: Dict[str, Any]
    ) -> List[List[int]]:
        """
        Generate token IDs based on input tokens.
        
        Args:
            input_tokens: Batch of input token IDs
            generation_config: Generation parameters
        
        Returns:
            List of generated token ID sequences
        """
        # Convert to tensors
        max_len = max(len(tokens) for tokens in input_tokens)
        padded_tokens = [tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens)) for tokens in input_tokens]
        input_ids = torch.tensor(padded_tokens, dtype=torch.long)
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        for i, tokens in enumerate(input_tokens):
            attention_mask[i, len(tokens):] = 0
        
        # Move to the first device in the model
        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)
        attention_mask = attention_mask.to(first_device)
        
        # Handle batching if needed
        batch_size = input_ids.shape[0]
        if batch_size > self.max_batch_size:
            logger.warning(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}. Splitting into multiple batches.")
            
            # Process batches and concatenate results
            all_generated_ids = []
            for i in range(0, batch_size, self.max_batch_size):
                batch_input_ids = input_ids[i:i+self.max_batch_size]
                batch_attention_mask = attention_mask[i:i+self.max_batch_size]
                batch_generated_ids = self._generate_tokens_single_batch(
                    batch_input_ids, batch_attention_mask, generation_config
                )
                all_generated_ids.extend(batch_generated_ids)
            return all_generated_ids
        else:
            # Process single batch
            return self._generate_tokens_single_batch(input_ids, attention_mask, generation_config)
    
    def _generate_tokens_single_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: Dict[str, Any]
    ) -> List[List[int]]:
        """
        Generate tokens for a single batch that fits in memory.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            generation_config: Generation parameters
        
        Returns:
            List of token ID sequences
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        max_length = seq_len + generation_config["max_new_tokens"]
        
        # Initialize storage for generated sequences
        generated_ids = input_ids.clone()
        
        # Initialize KV cache if enabled
        use_kv_cache = True  # Could be a config parameter
        
        # Generate tokens autoregressively
        for _ in range(generation_config["max_new_tokens"]):
            # Get predictions for next token
            with torch.no_grad():
                if use_kv_cache and generated_ids.shape[1] > 1:
                    # Use cached values for previous tokens
                    # In a real implementation, this would use a specific forward method
                    # that takes advantage of KV caching
                    logits = self.model(generated_ids[:, -1:], None)[:, -1, :]
                else:
                    # No cache or first token
                    logits = self.model(generated_ids, attention_mask)[:, -1, :]
            
            # Apply temperature
            if generation_config["temperature"] > 0:
                logits = logits / generation_config["temperature"]
            
            # Apply repetition penalty
            if generation_config["repetition_penalty"] > 1.0:
                # Create score tensor from logits
                scores = logits.clone()
                
                # Apply penalty to previously generated tokens
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        # If score > 0, reduce it; if score < 0, increase it
                        scores[i, token_id] /= generation_config["repetition_penalty"]
                
                logits = scores
            
            # Apply top-k filtering
            if generation_config["top_k"] > 0:
                indices_to_remove = logits < torch.topk(logits, generation_config["top_k"])[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if generation_config["top_p"] < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > generation_config["top_p"]
                
                # Shift indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted indices back to original order
                for i in range(batch_size):
                    indices_to_remove = sorted_indices_to_remove[i].scatter(
                        0, sorted_indices[i], sorted_indices_to_remove[i]
                    )
                    logits[i][indices_to_remove] = float('-inf')
            
            # Get next token
            if generation_config["do_sample"]:
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append new tokens
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
            ], dim=1)
            
            # Check for end of sequence tokens
            if all((next_tokens == self.tokenizer.eos_token_id).view(-1)):
                break
        
        # Convert to list of token IDs
        return [ids.tolist() for ids in generated_ids]
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the input text.
        
        Args:
            text: Input text or list of texts
        
        Returns:
            Embeddings as numpy array [batch_size, embedding_dim]
        """
        # Convert single text to list
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Tokenize text
        input_tokens = self.tokenizer.encode_batch(texts)
        
        # Convert to tensors
        max_len = max(len(tokens) for tokens in input_tokens)
        padded_tokens = [tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens)) for tokens in input_tokens]
        input_ids = torch.tensor(padded_tokens, dtype=torch.long)
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        for i, tokens in enumerate(input_tokens):
            attention_mask[i, len(tokens):] = 0
        
        # Move to the first device in the model
        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)
        attention_mask = attention_mask.to(first_device)
        
        # Forward pass through the model to get embeddings
        with torch.no_grad():
            # In a real implementation, we'd use a specific embedding method
            # Here we're just using the output of the final layer
            outputs = self.model(input_ids, attention_mask)
            
            # Get embeddings from the last token of each sequence
            embeddings = []
            for i, length in enumerate(map(len, input_tokens)):
                embeddings.append(outputs[i, length-1])
            
            # Stack embeddings
            embeddings = torch.stack(embeddings)
        
        # Convert to numpy
        return embeddings.cpu().numpy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return self.inference_stats
    
    def clear_cache(self):
        """Clear KV-cache and other temporary storage."""
        self.kv_cache = {}
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Cleared inference caches")
    
    def unload(self):
        """Unload the model from memory."""
        del self.model
        self.model = None
        
        # Clear caches
        self.clear_cache()
        
        # Run garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Unloaded model {self.model_id}")


class DistributedInference:
    """
    Manages inference across multiple worker nodes.
    
    This class coordinates token generation across multiple machines,
    each potentially running a shard of the model.
    """
    
    def __init__(
        self,
        coordinator_client,
        model_id: str,
        tokenizer = None,
        max_sequence_length: int = 2048,
    ):
        """
        Initialize the distributed inference manager.
        
        Args:
            coordinator_client: Client for communicating with coordinator
            model_id: ID of the model to use
            tokenizer: Optional tokenizer to use (will be loaded if not provided)
            max_sequence_length: Maximum sequence length for generation
        """
        self.coordinator_client = coordinator_client
        self.model_id = model_id
        self.max_sequence_length = max_sequence_length
        
        # Load tokenizer if not provided
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer(model_id)
        
        # Request queues
        self.request_queue = queue.Queue()
        self.result_queue = {}  # Map from request_id to result
        
        # Statistics and monitoring
        self.inference_stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_inference_time": 0,
            "tokens_per_second": 0,
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
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text completions using distributed inference.
        
        Args:
            prompt: Input prompt text or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            top_k: Top-k sampling threshold
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
                    
                    # Tokenize prompts
                    input_tokens = self.tokenizer.encode_batch(prompts)
                    
                    # Create generation tasks
                    tasks = []
                    for i, tokens in enumerate(input_tokens):
                        task = {
                            "type": "token_generation",
                            "input_ids": tokens,
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
                    output_ids = []
                    for result in sorted(results, key=lambda r: r["sequence_index"]):
                        output_ids.append(result["output_ids"])
                    
                    # Decode output tokens
                    generated_texts = self.tokenizer.decode_batch(output_ids)
                    
                    # Set result
                    if is_batch:
                        self.result_queue[request_id] = generated_texts
                    else:
                        self.result_queue[request_id] = generated_texts[0]
                    
                    # Update statistics
                    generation_time = time.time() - start_time
                    num_new_tokens = sum(len(out) - len(inp) for out, inp in zip(output_ids, input_tokens))
                    
                    self.inference_stats["total_requests"] += len(prompts)
                    self.inference_stats["total_tokens_generated"] += num_new_tokens
                    self.inference_stats["total_inference_time"] += generation_time
                    self.inference_stats["tokens_per_second"] = (
                        self.inference_stats["total_tokens_generated"] / 
                        self.inference_stats["total_inference_time"]
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing request {request_id}: {e}")
                    # Set error result
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