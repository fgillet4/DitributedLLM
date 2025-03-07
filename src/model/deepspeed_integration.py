"""
DeepSpeed Integration for DistributedLLM.

This module provides compatibility with DeepSpeed's inference capabilities,
allowing users to leverage DeepSpeed optimizations within DistributedLLM.

Features:
1. DeepSpeed checkpoint loading
2. Kernel injection for high-performance inference
3. Efficient tensor parallelism
4. Integration with DeepSpeed-Inference API
"""

import logging
import os
import json
import torch
import importlib.util
from typing import Dict, List, Optional, Union, Any, Tuple

from src.model.layers import ShardedModel
from src.model.optimized_inference import OptimizedInference
from src.model.quantization import MoQConfig, apply_moq

logger = logging.getLogger(__name__)

# Check if DeepSpeed is available
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    logger.warning("DeepSpeed not available. Install with 'pip install deepspeed'")


def is_deepspeed_available():
    """Check if DeepSpeed is available"""
    return DEEPSPEED_AVAILABLE


class DeepSpeedConfig:
    """
    Configuration for DeepSpeed integration.
    
    Handles parsing and validation of DeepSpeed configuration options.
    """
    
    def __init__(
        self,
        tensor_parallel_size: int = 1,
        dtype: torch.dtype = torch.float16,
        checkpoint_dict: Optional[Dict[str, Any]] = None,
        replace_with_kernel_inject: bool = True,
        enable_cuda_graph: bool = False,
        max_out_tokens: int = 1024,
        use_quantization: bool = False,
        quantization_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DeepSpeed configuration.
        
        Args:
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type for model weights
            checkpoint_dict: Checkpoint configuration for loading weights
            replace_with_kernel_inject: Whether to inject optimized kernels
            enable_cuda_graph: Whether to use CUDA graphs for inference
            max_out_tokens: Maximum number of output tokens
            use_quantization: Whether to use quantization
            quantization_config: Quantization configuration
        """
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.checkpoint_dict = checkpoint_dict
        self.replace_with_kernel_inject = replace_with_kernel_inject
        self.enable_cuda_graph = enable_cuda_graph
        self.max_out_tokens = max_out_tokens
        self.use_quantization = use_quantization
        
        # Validate and set quantization config
        if use_quantization:
            if quantization_config is None:
                # Default quantization config
                self.quantization_config = {
                    'bits': 8,
                    'group_size': 64,
                    'mlp_extra_grouping': True
                }
            else:
                self.quantization_config = quantization_config
        else:
            self.quantization_config = None


def load_deepspeed_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a DeepSpeed checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
    
    Returns:
        Dictionary containing the model weights
    """
    if not DEEPSPEED_AVAILABLE:
        raise ImportError("DeepSpeed is not available. Please install it with 'pip install deepspeed'")
    
    # Handle different checkpoint formats
    if os.path.isdir(checkpoint_path):
        # Try to find the latest checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_path}")
        
        # Sort by modification time (latest first)
        checkpoints = sorted(
            checkpoints,
            key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)),
            reverse=True
        )
        checkpoint_path = os.path.join(checkpoint_path, checkpoints[0])
    
    logger.info(f"Loading DeepSpeed checkpoint from {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'module' in checkpoint:
        return checkpoint['module']
    else:
        return checkpoint


def create_injection_policy() -> Dict[str, List[str]]:
    """
    Create an injection policy for DeepSpeed kernel optimization.
    
    The policy defines which layers should be replaced with optimized kernels.
    
    Returns:
        Dictionary mapping layer types to lists of weight names
    """
    from src.model.layers import ShardedTransformerBlock
    
    policy = {
        ShardedTransformerBlock: [
            'attention.q_proj', 
            'attention.k_proj', 
            'attention.v_proj', 
            'attention.o_proj',
            'mlp.gate_proj',
            'mlp.up_proj',
            'mlp.down_proj'
        ]
    }
    
    return policy


def init_inference(
    model: ShardedModel,
    config: DeepSpeedConfig,
    tokenizer = None
) -> Union[ShardedModel, 'deepspeed.InferenceEngine']:
    """
    Initialize model for inference with DeepSpeed optimizations.
    
    Args:
        model: Model to optimize
        config: DeepSpeed configuration
        tokenizer: Optional tokenizer
    
    Returns:
        Optimized model or DeepSpeed inference engine
    """
    if not DEEPSPEED_AVAILABLE:
        logger.warning("DeepSpeed not available. Falling back to DistributedLLM optimizations")
        # Use our own optimized inference
        return OptimizedInference(
            model=model,
            tokenizer=tokenizer,
            dtype=config.dtype,
            quantization_config=config.quantization_config if config.use_quantization else None
        )
    
    # If quantization is enabled, apply it first
    if config.use_quantization:
        # Create MoQ config
        moq_config = MoQConfig(
            bits=config.quantization_config.get('bits', 8),
            group_size=config.quantization_config.get('group_size', 64),
            mlp_extra_grouping=config.quantization_config.get('mlp_extra_grouping', True)
        )
        
        # Apply quantization
        logger.info(f"Applying quantization with config: {config.quantization_config}")
        model = apply_moq(model, moq_config)
    
    # Prepare tensor parallel config
    if config.tensor_parallel_size > 1:
        tp_config = {"tp_size": config.tensor_parallel_size}
    else:
        tp_config = None
    
    # Prepare injection policy if needed
    if config.replace_with_kernel_inject:
        injection_policy = create_injection_policy()
    else:
        injection_policy = None
    
    # Get the right dtype
    if config.dtype == torch.float16:
        ds_dtype = torch.half
    elif config.dtype == torch.float32:
        ds_dtype = torch.float
    elif config.dtype == torch.int8:
        ds_dtype = torch.int8
    else:
        logger.warning(f"Unsupported dtype {config.dtype}, falling back to float16")
        ds_dtype = torch.half
    
    logger.info(f"Initializing DeepSpeed inference with tensor parallel size {config.tensor_parallel_size}")
    
    # Initialize DeepSpeed inference
    ds_engine = deepspeed.init_inference(
        model=model,
        tensor_parallel=tp_config,
        dtype=ds_dtype,
        checkpoint=config.checkpoint_dict,
        replace_with_kernel_inject=config.replace_with_kernel_inject,
        injection_policy=injection_policy
    )
    
    return ds_engine


def convert_checkpoint_to_deepspeed(
    checkpoint_path: str,
    output_path: str,
    tensor_parallel_size: int = 1
) -> str:
    """
    Convert a DistributedLLM checkpoint to DeepSpeed format.
    
    Args:
        checkpoint_path: Path to DistributedLLM checkpoint
        output_path: Path to save DeepSpeed checkpoint
        tensor_parallel_size: Tensor parallel size for splitting
    
    Returns:
        Path to converted checkpoint
    """
    if not DEEPSPEED_AVAILABLE:
        raise ImportError("DeepSpeed is not available. Please install it with 'pip install deepspeed'")
    
    logger.info(f"Converting checkpoint from {checkpoint_path} to DeepSpeed format")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create DeepSpeed-compatible checkpoint
    ds_checkpoint = {
        'module': checkpoint['model'],
        'optimizer': None,
        'lr_scheduler': None,
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'deepspeed_version': '0.0',  # Will be updated by DeepSpeed
    }
    
    # Create checkpoint directory
    os.makedirs(output_path, exist_ok=True)
    
    # If tensor parallel size is 1, just save a single file
    if tensor_parallel_size == 1:
        checkpoint_file = os.path.join(output_path, 'ds_inference_checkpoint.pt')
        torch.save(ds_checkpoint, checkpoint_file)
        logger.info(f"Saved DeepSpeed checkpoint to {checkpoint_file}")
        return checkpoint_file
    
    # For model parallel, we need to split the checkpoint
    # This is a simplified version - actual implementation would be more complex
    for tp_rank in range(tensor_parallel_size):
        # Create rank-specific checkpoint
        rank_checkpoint = {
            'module': {},  # Will contain only this rank's parameters
            'optimizer': None,
            'lr_scheduler': None,
            'epoch': ds_checkpoint['epoch'],
            'global_step': ds_checkpoint['global_step'],
            'deepspeed_version': ds_checkpoint['deepspeed_version'],
        }
        
        # Create rank directory
        rank_dir = os.path.join(output_path, f'mp_rank_{tp_rank:02d}')
        os.makedirs(rank_dir, exist_ok=True)
        
        # Save rank checkpoint
        rank_file = os.path.join(rank_dir, 'ds_inference_checkpoint.pt')
        torch.save(rank_checkpoint, rank_file)
    
    # Create checkpoint config
    checkpoint_config = {
        "type": "Megatron",
        "version": 0.0,
        "checkpoints": [
            f"mp_rank_{tp_rank:02d}/ds_inference_checkpoint.pt" for tp_rank in range(tensor_parallel_size)
        ],
    }
    
    # Save checkpoint config
    config_file = os.path.join(output_path, 'ds_inference_config.json')
    with open(config_file, 'w') as f:
        json.dump(checkpoint_config, f, indent=4)
    
    logger.info(f"Saved DeepSpeed checkpoint config to {config_file}")
    return config_file


class DeepSpeedModelWrapper:
    """
    Wrapper for models optimized with DeepSpeed.
    
    Provides a consistent interface for both DeepSpeed and native models.
    """
    
    def __init__(
        self,
        model: Union[ShardedModel, 'deepspeed.InferenceEngine'],
        tokenizer = None,
        config: Optional[DeepSpeedConfig] = None
    ):
        """
        Initialize the wrapped model.
        
        Args:
            model: DeepSpeed-optimized model or engine
            tokenizer: Tokenizer for text processing
            config: DeepSpeed configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DeepSpeedConfig()
        
        # Check if this is a DeepSpeed engine
        self.is_deepspeed_engine = hasattr(model, 'module')
        
        # If it's a DeepSpeed engine, get the module
        if self.is_deepspeed_engine:
            self.module = model.module
        else:
            self.module = model
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text with the model.
        
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
            Generated text or list of generated texts
        """
        # Handle different model types
        if hasattr(self.model, 'generate'):
            # Model has a generate method, use it directly
            return self.model.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                **kwargs
            )
        else:
            # We need to implement token generation manually
            # Convert single prompt to list
            if isinstance(prompt, str):
                prompts = [prompt]
                is_single_prompt = True
            else:
                prompts = prompt
                is_single_prompt = False
            
            # Tokenize prompts
            input_tokens = self.tokenizer.encode_batch(prompts, add_special_tokens=True)
            
            # Convert to tensors
            max_len = max(len(tokens) for tokens in input_tokens)
            input_ids = torch.zeros((len(prompts), max_len), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            
            for i, tokens in enumerate(input_tokens):
                input_ids[i, :len(tokens)] = torch.tensor(tokens)
            
            # Move to the first device in the model
            first_device = next(self.module.parameters()).device
            input_ids = input_ids.to(first_device)
            attention_mask = attention_mask.to(first_device)
            
            # Generate tokens (simplified implementation)
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    next_token_logits = outputs[0][:, -1, :]
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Sample next token
                    if do_sample:
                        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1)
                    else:
                        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Append new tokens
                    input_ids = torch.cat([input_ids, next_tokens], dim=1)
                    
                    # Update attention mask
                    attention_mask = torch.cat([
                        attention_mask, 
                        torch.ones((len(prompts), 1), dtype=attention_mask.dtype, device=first_device)
                    ], dim=1)
            
            # Convert to lists of token IDs
            output_ids = [ids.tolist() for ids in input_ids]
            
            # Decode to text
            generated_texts = self.tokenizer.decode_batch(output_ids)
            
            return generated_texts[0] if is_single_prompt else generated_texts
    
    def __call__(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)


class DeepSpeedFastGenWrapper:
    """
    Wrapper for DeepSpeed-FastGen support.
    
    Integrates with DeepSpeed's FastGen for high-performance generation.
    """
    
    def __init__(
        self,
        model: ShardedModel,
        tokenizer = None,
        config: Optional[DeepSpeedConfig] = None
    ):
        """
        Initialize FastGen wrapper.
        
        Args:
            model: Model to optimize
            tokenizer: Tokenizer for text processing
            config: DeepSpeed configuration
        """
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is not available. Please install it with 'pip install deepspeed'")
        
        # Check if FastGen is available
        try:
            from deepspeed.inference.fastgen import FastGen
            self.fastgen_available = True
        except ImportError:
            logger.warning("DeepSpeed-FastGen not available. Falling back to standard inference.")
            self.fastgen_available = False
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DeepSpeedConfig()
        
        # Initialize FastGen if available
        if self.fastgen_available:
            try:
                from deepspeed.inference.fastgen import FastGen
                
                # Initialize DeepSpeed inference first
                ds_engine = init_inference(model, self.config, tokenizer)
                
                # Now initialize FastGen
                self.generator = FastGen(
                    ds_engine.module if hasattr(ds_engine, 'module') else ds_engine,
                    tokenizer,
                    max_output_len=self.config.max_out_tokens
                )
                
                logger.info("Initialized DeepSpeed-FastGen for high-performance generation")
            except Exception as e:
                logger.warning(f"Failed to initialize DeepSpeed-FastGen: {e}")
                self.fastgen_available = False
                
                # Fall back to standard wrapper
                self.wrapper = DeepSpeedModelWrapper(
                    init_inference(model, self.config, tokenizer),
                    tokenizer,
                    self.config
                )
        else:
            # Fall back to standard wrapper
            self.wrapper = DeepSpeedModelWrapper(
                init_inference(model, self.config, tokenizer),
                tokenizer,
                self.config
            )
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text with FastGen.
        
        Args:
            prompt: Input prompt text or list of prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling threshold
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text or list of generated texts
        """
        if self.fastgen_available:
            # Use FastGen for generation
            try:
                generation_params = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_new_tokens": max_new_tokens,
                    **kwargs
                }
                
                # Handle single vs. batch input
                if isinstance(prompt, str):
                    return self.generator.generate(prompt, **generation_params)
                else:
                    return [self.generator.generate(p, **generation_params) for p in prompt]
            
            except Exception as e:
                logger.error(f"FastGen generation failed: {e}")
                logger.info("Falling back to standard generation")
                return self.wrapper.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    **kwargs
                )
        else:
            # Use standard wrapper
            return self.wrapper.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **kwargs
            )


def optimize_with_deepspeed(
    model: ShardedModel,
    tokenizer = None,
    tensor_parallel_size: int = 1,
    dtype: torch.dtype = torch.float16,
    checkpoint_path: Optional[str] = None,
    enable_fastgen: bool = True,
    use_quantization: bool = False,
    quantization_config: Optional[Dict[str, Any]] = None
) -> Union[DeepSpeedModelWrapper, DeepSpeedFastGenWrapper]:
    """
    Optimize a model with DeepSpeed.
    
    Args:
        model: Model to optimize
        tokenizer: Tokenizer for text processing
        tensor_parallel_size: Number of GPUs for tensor parallelism
        dtype: Data type for model weights
        checkpoint_path: Path to checkpoint file or directory
        enable_fastgen: Whether to use FastGen for generation
        use_quantization: Whether to use quantization
        quantization_config: Quantization configuration
    
    Returns:
        Optimized model wrapper
    """
    # Create checkpoint dict if path is provided
    checkpoint_dict = None
    if checkpoint_path:
        if os.path.isdir(checkpoint_path):
            # Look for config json
            config_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.json')]
            if config_files:
                config_path = os.path.join(checkpoint_path, config_files[0])
                with open(config_path, 'r') as f:
                    checkpoint_dict = json.load(f)
            else:
                # Create a simple config
                checkpoint_dict = {
                    "type": "ds_model",
                    "version": 0.0,
                    "checkpoints": checkpoint_path
                }
        else:
            # Single file checkpoint
            checkpoint_dict = {
                "type": "ds_model",
                "version": 0.0,
                "checkpoints": checkpoint_path
            }
    
    # Create DeepSpeed config
    config = DeepSpeedConfig(
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        checkpoint_dict=checkpoint_dict,
        replace_with_kernel_inject=True,
        enable_cuda_graph=True,
        max_out_tokens=1024,
        use_quantization=use_quantization,
        quantization_config=quantization_config
    )
    
    # Use FastGen if requested and available
    if enable_fastgen and DEEPSPEED_AVAILABLE:
        try:
            from deepspeed.inference.fastgen import FastGen
            return DeepSpeedFastGenWrapper(model, tokenizer, config)
        except ImportError:
            logger.warning("FastGen not available, falling back to standard DeepSpeed inference")
    
    # Use standard DeepSpeed wrapper
    ds_model = init_inference(model, config, tokenizer)
    return DeepSpeedModelWrapper(ds_model, tokenizer, config)