"""
Quantization module for DistributedLLM.

This module implements quantization techniques inspired by DeepSpeed's ZeroQuant.
Features include:
1. INT8 quantization of weights
2. Mixed precision inference
3. Group-wise quantization for improved accuracy
4. Model compression for reduced memory footprint
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

logger = logging.getLogger(__name__)

class QuantizedLinear(nn.Module):
    """
    Quantized linear layer with configurable precision.
    
    This implementation supports:
    - INT8 and INT4 quantization
    - Per-channel quantization scales
    - Optional bias term
    - Efficient matrix multiplication with dequantization
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        group_size: int = 128,
        symmetric: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = min(group_size, in_features)
        self.symmetric = symmetric
        
        # Calculate number of groups
        self.num_groups = (in_features + self.group_size - 1) // self.group_size
        
        # Initialize quantized weight placeholder
        # For int8, we use int8 tensor; for int4, we pack two values per int8
        if bits == 8:
            self.register_buffer(
                'quantized_weight', 
                torch.zeros((out_features, in_features), dtype=torch.int8)
            )
        elif bits == 4:
            # For int4, we pack two values per int8, so we need half the width
            self.register_buffer(
                'quantized_weight', 
                torch.zeros((out_features, (in_features + 1) // 2), dtype=torch.int8)
            )
        else:
            raise ValueError(f"Unsupported bits: {bits}. Choose 8 or 4.")
        
        # Initialize scales and zero points
        # We have one scale per output row and per group
        self.register_buffer(
            'scales', 
            torch.zeros((out_features, self.num_groups), dtype=torch.float16)
        )
        
        # We only need zero points for asymmetric quantization
        if not symmetric:
            self.register_buffer(
                'zero_points', 
                torch.zeros((out_features, self.num_groups), dtype=torch.int8)
            )
        
        # Optional bias
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with efficient quantized computation"""
        # Cast input to correct precision
        orig_dtype = x.dtype
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        
        # Initialize output
        output = torch.zeros(
            x.shape[0], 
            self.out_features, 
            dtype=torch.float16, 
            device=x.device
        )
        
        # Process each group
        for group_idx in range(self.num_groups):
            # Calculate start and end indices for this group
            start_idx = group_idx * self.group_size
            end_idx = min(start_idx + self.group_size, self.in_features)
            
            # Get a slice of the input
            if len(x.shape) == 3:  # For batched sequence input
                x_slice = x[:, :, start_idx:end_idx]
            else:  # For simple input
                x_slice = x[:, start_idx:end_idx]
            
            # Get scales for this group
            scales_group = self.scales[:, group_idx].view(1, -1)
            
            # Extract quantized weights for this group
            if self.bits == 8:
                weights_group = self.quantized_weight[:, start_idx:end_idx].to(torch.float16)
                
                # Dequantize weights
                if hasattr(self, 'zero_points'):  # Asymmetric quantization
                    zero_points_group = self.zero_points[:, group_idx].view(-1, 1)
                    dequantized_weights = (weights_group - zero_points_group) * scales_group.view(-1, 1)
                else:  # Symmetric quantization
                    dequantized_weights = weights_group * scales_group.view(-1, 1)
            
            elif self.bits == 4:
                # Extract packed weights and unpack them
                packed_weights = self.quantized_weight[:, (start_idx // 2):((end_idx + 1) // 2)]
                
                # Unpack int4 values (low 4 bits and high 4 bits)
                weights_low = (packed_weights & 0x0F).to(torch.float16)
                weights_high = (packed_weights >> 4).to(torch.float16)
                
                # Interleave to get the original order
                if (end_idx - start_idx) % 2 == 0:  # Even number of elements
                    weights_group = torch.stack([weights_low, weights_high], dim=2).view(self.out_features, end_idx - start_idx)
                else:  # Odd number of elements, ignore last high bits
                    weights_group = torch.stack([weights_low, weights_high], dim=2).view(self.out_features, -1)[:, :(end_idx - start_idx)]
                
                # Dequantize weights
                if hasattr(self, 'zero_points'):  # Asymmetric quantization
                    zero_points_group = self.zero_points[:, group_idx].view(-1, 1)
                    dequantized_weights = (weights_group - zero_points_group) * scales_group.view(-1, 1)
                else:  # Symmetric quantization
                    dequantized_weights = weights_group * scales_group.view(-1, 1)
            
            # Perform matrix multiplication for this group
            if len(x.shape) == 3:  # For batched sequence input
                group_output = torch.bmm(
                    x_slice, 
                    dequantized_weights.t().unsqueeze(0).expand(x.shape[0], -1, -1)
                )
                output += group_output
            else:  # For simple input
                group_output = F.linear(x_slice, dequantized_weights)
                output += group_output
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias
        
        # Return output in original dtype
        return output.to(orig_dtype)
    
    @classmethod
    def from_float(cls, linear: nn.Linear, bits: int = 8, group_size: int = 128, symmetric: bool = True) -> 'QuantizedLinear':
        """Convert a floating point linear layer to a quantized one"""
        weight = linear.weight.data
        bias = linear.bias.data if linear.bias is not None else None
        
        # Create quantized layer
        quantized_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=bias is not None,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric
        )
        
        # Quantize weights
        for group_idx in range(quantized_linear.num_groups):
            # Calculate start and end indices for this group
            start_idx = group_idx * group_size
            end_idx = min(start_idx + group_size, linear.in_features)
            
            # Get weight slice for this group
            weight_slice = weight[:, start_idx:end_idx]
            
            if symmetric:
                # Symmetric quantization
                max_val = torch.max(torch.abs(weight_slice), dim=1).values
                scales = max_val / (2**(bits-1) - 1)
                
                # Scale and round weights
                scaled_weights = torch.round(weight_slice / scales.view(-1, 1)).to(torch.int8)
                
                # Ensure values are within range
                scaled_weights = torch.clamp(scaled_weights, -2**(bits-1), 2**(bits-1) - 1)
                
                # Store scales and quantized weights
                quantized_linear.scales[:, group_idx] = scales
                
                if bits == 8:
                    quantized_linear.quantized_weight[:, start_idx:end_idx] = scaled_weights
                elif bits == 4:
                    # Pack two int4 values into one int8
                    # First, ensure values are in range [0, 15]
                    scaled_weights = torch.clamp(scaled_weights, -8, 7)
                    # Convert to unsigned (0-15)
                    unsigned_weights = (scaled_weights + 8).to(torch.uint8)
                    
                    # Pack pairs of int4 values
                    for i in range(start_idx, end_idx, 2):
                        if i+1 < end_idx:  # We have a pair
                            col_idx = i // 2
                            low = unsigned_weights[:, i-start_idx]
                            high = unsigned_weights[:, i+1-start_idx]
                            packed = (high << 4) | low
                            quantized_linear.quantized_weight[:, col_idx] = packed
                        else:  # We have a single value at the end
                            col_idx = i // 2
                            low = unsigned_weights[:, i-start_idx]
                            packed = low  # Only use low bits
                            quantized_linear.quantized_weight[:, col_idx] = packed
            else:
                # Asymmetric quantization
                min_val = torch.min(weight_slice, dim=1).values
                max_val = torch.max(weight_slice, dim=1).values
                scales = (max_val - min_val) / (2**bits - 1)
                zero_points = torch.round(-min_val / scales).to(torch.int8)
                
                # Scale and round weights
                scaled_weights = torch.round(weight_slice / scales.view(-1, 1) + zero_points.view(-1, 1)).to(torch.int8)
                
                # Ensure values are within range
                scaled_weights = torch.clamp(scaled_weights, 0, 2**bits - 1)
                
                # Store scales, zero points, and quantized weights
                quantized_linear.scales[:, group_idx] = scales
                quantized_linear.zero_points[:, group_idx] = zero_points
                
                if bits == 8:
                    quantized_linear.quantized_weight[:, start_idx:end_idx] = scaled_weights
                elif bits == 4:
                    # Pack two int4 values into one int8
                    # First, ensure values are in range [0, 15]
                    scaled_weights = torch.clamp(scaled_weights, 0, 15)
                    
                    # Pack pairs of int4 values
                    for i in range(start_idx, end_idx, 2):
                        if i+1 < end_idx:  # We have a pair
                            col_idx = i // 2
                            low = scaled_weights[:, i-start_idx]
                            high = scaled_weights[:, i+1-start_idx]
                            packed = (high << 4) | low
                            quantized_linear.quantized_weight[:, col_idx] = packed
                        else:  # We have a single value at the end
                            col_idx = i // 2
                            low = scaled_weights[:, i-start_idx]
                            packed = low  # Only use low bits
                            quantized_linear.quantized_weight[:, col_idx] = packed
        
        # Set bias if present
        if bias is not None:
            quantized_linear.bias = bias.to(torch.float16)
        
        return quantized_linear


def quantize_model(
    model: nn.Module, 
    bits: int = 8, 
    group_size: int = 128, 
    symmetric: bool = True,
    layer_types: List[type] = [nn.Linear],
    excluded_layers: Optional[List[str]] = None
) -> nn.Module:
    """
    Quantize specific layers in a model.
    
    Args:
        model: Model to quantize
        bits: Bit width for quantization (4 or 8)
        group_size: Group size for quantization
        symmetric: Whether to use symmetric quantization
        layer_types: Types of layers to quantize
        excluded_layers: Names of layers to exclude
    
    Returns:
        Quantized model
    """
    if excluded_layers is None:
        excluded_layers = []
    
    # Track quantized layers
    quantized_layers = {}
    
    # Define a recursive function to apply quantization
    def _quantize_module(module, path=""):
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            
            # Skip excluded layers
            if child_path in excluded_layers:
                logger.info(f"Skipping excluded layer: {child_path}")
                continue
            
            # Quantize layer if it's a target type
            if any(isinstance(child, layer_type) for layer_type in layer_types):
                if isinstance(child, nn.Linear):
                    logger.info(f"Quantizing linear layer: {child_path}")
                    quantized_layer = QuantizedLinear.from_float(
                        child, 
                        bits=bits, 
                        group_size=group_size, 
                        symmetric=symmetric
                    )
                    setattr(module, name, quantized_layer)
                    quantized_layers[child_path] = "Linear"
            
            # Recurse into child module
            _quantize_module(child, child_path)
    
    # Apply quantization
    _quantize_module(model)
    
    logger.info(f"Quantized {len(quantized_layers)} layers to {bits}-bit precision")
    return model


class MoQConfig:
    """
    Configuration for Model Quantization (MoQ).
    
    Inspired by DeepSpeed's ZeroQuant, this class configures quantization
    settings for different layer types in the model.
    """
    
    def __init__(
        self,
        bits: int = 8,
        group_size: int = 128,
        symmetric: bool = True,
        mlp_extra_grouping: bool = False,
        excluded_modules: Optional[List[str]] = None
    ):
        """
        Initialize MoQ configuration.
        
        Args:
            bits: Bit width for quantization (4 or 8)
            group_size: Group size for quantization
            symmetric: Whether to use symmetric quantization
            mlp_extra_grouping: Use different group size for MLP
            excluded_modules: Module names to exclude from quantization
        """
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.mlp_extra_grouping = mlp_extra_grouping
        self.excluded_modules = excluded_modules or []
        
        # Different configurations for different layer types
        if mlp_extra_grouping:
            # Use smaller groups for MLP to preserve accuracy
            self.mlp_group_size = group_size // 2
        else:
            self.mlp_group_size = group_size
    
    def get_module_config(self, module_path: str) -> Dict[str, Any]:
        """Get configuration for a specific module"""
        if module_path in self.excluded_modules:
            return None
        
        # Check if it's an MLP module
        if 'mlp' in module_path.lower() or 'feed_forward' in module_path.lower():
            return {
                'bits': self.bits,
                'group_size': self.mlp_group_size,
                'symmetric': self.symmetric
            }
        
        # Default configuration
        return {
            'bits': self.bits,
            'group_size': self.group_size,
            'symmetric': self.symmetric
        }


def apply_moq(
    model: nn.Module, 
    config: MoQConfig
) -> nn.Module:
    """
    Apply Model Quantization (MoQ) to a model.
    
    Args:
        model: Model to quantize
        config: MoQ configuration
    
    Returns:
        Quantized model
    """
    # Track quantized layers
    quantized_layers = {}
    
    # Define a recursive function to apply quantization
    def _quantize_module(module, path=""):
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            
            # Get config for this module
            module_config = config.get_module_config(child_path)
            if module_config is None:
                logger.info(f"Skipping excluded module: {child_path}")
                continue
            
            # Quantize layer if it's a linear layer
            if isinstance(child, nn.Linear):
                logger.info(f"Quantizing linear layer: {child_path}")
                quantized_layer = QuantizedLinear.from_float(
                    child, 
                    bits=module_config['bits'], 
                    group_size=module_config['group_size'], 
                    symmetric=module_config['symmetric']
                )
                setattr(module, name, quantized_layer)
                quantized_layers[child_path] = {
                    'type': "Linear", 
                    'config': module_config
                }
            
            # Recurse into child module
            _quantize_module(child, child_path)
    
    # Apply quantization
    _quantize_module(model)
    
    logger.info(f"Applied MoQ to {len(quantized_layers)} layers")
    return model


class QuantizedModelForInference:
    """
    Wrapper for a quantized model optimized for inference.
    
    Handles:
    - Mixed precision execution
    - Efficient batching
    - Memory optimizations
    """
    
    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.float16,
        device: Union[torch.device, str] = 'cuda'
    ):
        """
        Initialize the inference wrapper.
        
        Args:
            model: Quantized model
            dtype: Data type for non-quantized operations
            device: Device to run inference on
        """
        self.model = model
        self.dtype = dtype
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Move model to device and set eval mode
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, *args, **kwargs):
        """Run the model's generate method with optimized settings"""
        # Disable gradient computation
        with torch.no_grad():
            return self.model.generate(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Run the model's forward method with optimized settings"""
        # Disable gradient computation
        with torch.no_grad():
            return self.model(*args, **kwargs)
    
    def optimize_memory(self):
        """Apply additional memory optimizations"""
        # Clear CUDA cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Offload unused layers to CPU
        # This is a simplified version - a real implementation would be more complex
        logger.info("Optimizing memory usage")