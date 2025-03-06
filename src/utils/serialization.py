"""
Serialization utilities for DistributedLLM.

Provides functions for efficient serialization and deserialization of
model weights, tensors, and other data structures.
"""

import logging
import io
import os
import pickle
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some serialization features will be limited.")


def serialize_tensor(tensor, compress: bool = True) -> bytes:
    """
    Serialize a tensor to bytes, with optional compression.
    
    Args:
        tensor: PyTorch tensor or NumPy array to serialize
        compress: Whether to apply compression
    
    Returns:
        Serialized tensor as bytes
    """
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        # Convert PyTorch tensor to NumPy array
        tensor = tensor.detach().cpu().numpy()
    
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")
    
    # Get tensor metadata
    dtype = str(tensor.dtype)
    shape = tensor.shape
    
    # Serialize metadata
    metadata = {
        'dtype': dtype,
        'shape': shape,
    }
    metadata_bytes = pickle.dumps(metadata)
    
    # Serialize tensor data
    tensor_bytes = tensor.tobytes()
    
    # Combine metadata and tensor data
    metadata_size = len(metadata_bytes)
    combined = struct.pack('>I', metadata_size) + metadata_bytes + tensor_bytes
    
    # Apply compression if requested
    if compress:
        compressed = zlib.compress(combined)
        # Add compression flag
        result = b'C' + compressed
    else:
        # Add no-compression flag
        result = b'N' + combined
    
    return result


def deserialize_tensor(data: bytes) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Deserialize a tensor from bytes.
    
    Args:
        data: Serialized tensor bytes
    
    Returns:
        Deserialized tensor (NumPy array or PyTorch tensor based on availability)
    """
    # Check compression flag
    compression_flag = data[0:1]
    data = data[1:]
    
    # Decompress if needed
    if compression_flag == b'C':
        data = zlib.decompress(data)
    
    # Extract metadata size
    metadata_size = struct.unpack('>I', data[:4])[0]
    
    # Extract metadata
    metadata_bytes = data[4:4+metadata_size]
    metadata = pickle.loads(metadata_bytes)
    
    # Extract tensor data
    tensor_bytes = data[4+metadata_size:]
    
    # Reconstruct tensor
    dtype = np.dtype(metadata['dtype'])
    shape = metadata['shape']
    
    array = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
    
    # Convert to PyTorch tensor if available
    if TORCH_AVAILABLE:
        return torch.from_numpy(array)
    else:
        return array


def serialize_model_weights(weights: Dict[str, Any], compress: bool = True) -> bytes:
    """
    Serialize model weights to bytes.
    
    Args:
        weights: Dictionary of weight tensors
        compress: Whether to apply compression
    
    Returns:
        Serialized weights as bytes
    """
    # Serialize each tensor in the weights dictionary
    serialized_weights = {}
    for key, tensor in weights.items():
        serialized_weights[key] = serialize_tensor(tensor, compress=False)
    
    # Serialize the dictionary
    result = pickle.dumps(serialized_weights)
    
    # Apply compression if requested
    if compress:
        result = zlib.compress(result)
    
    return result


def deserialize_model_weights(data: bytes) -> Dict[str, Any]:
    """
    Deserialize model weights from bytes.
    
    Args:
        data: Serialized weights bytes
    
    Returns:
        Dictionary of weight tensors
    """
    try:
        # Try to decompress first (in case it's compressed)
        try:
            data = zlib.decompress(data)
        except zlib.error:
            # Not compressed, continue with original data
            pass
        
        # Deserialize the dictionary
        serialized_weights = pickle.loads(data)
        
        # Deserialize each tensor in the dictionary
        weights = {}
        for key, tensor_bytes in serialized_weights.items():
            weights[key] = deserialize_tensor(tensor_bytes)
        
        return weights
    
    except Exception as e:
        logger.error(f"Error deserializing model weights: {e}")
        raise


def save_model_weights(weights: Dict[str, Any], file_path: str, compress: bool = True):
    """
    Save model weights to a file.
    
    Args:
        weights: Dictionary of weight tensors
        file_path: Path to save the weights to
        compress: Whether to apply compression
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Serialize weights
    serialized = serialize_model_weights(weights, compress=compress)
    
    # Save to file
    with open(file_path, 'wb') as f:
        f.write(serialized)
    
    logger.info(f"Saved model weights to {file_path}")


def load_model_weights(file_path: str) -> Dict[str, Any]:
    """
    Load model weights from a file.
    
    Args:
        file_path: Path to load the weights from
    
    Returns:
        Dictionary of weight tensors
    """
    # Load from file
    with open(file_path, 'rb') as f:
        serialized = f.read()
    
    # Deserialize weights
    weights = deserialize_model_weights(serialized)
    
    logger.info(f"Loaded model weights from {file_path}")
    return weights


def serialize_cache(cache_data: Dict[str, Any]) -> bytes:
    """
    Serialize a key-value cache for efficient storage.
    
    Args:
        cache_data: Dictionary of cache data
    
    Returns:
        Serialized cache as bytes
    """
    # Special handling for tensors in the cache
    serialized_cache = {}
    for key, value in cache_data.items():
        if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            serialized_cache[key] = ('tensor', serialize_tensor(value))
        elif isinstance(value, np.ndarray):
            serialized_cache[key] = ('array', serialize_tensor(value))
        else:
            serialized_cache[key] = ('pickle', pickle.dumps(value))
    
    # Serialize the dictionary
    result = pickle.dumps(serialized_cache)
    
    # Apply compression
    return zlib.compress(result)


def deserialize_cache(data: bytes) -> Dict[str, Any]:
    """
    Deserialize a key-value cache.
    
    Args:
        data: Serialized cache bytes
    
    Returns:
        Dictionary of cache data
    """
    try:
        # Decompress the data
        decompressed = zlib.decompress(data)
        
        # Deserialize the dictionary
        serialized_cache = pickle.loads(decompressed)
        
        # Deserialize each value in the dictionary
        cache_data = {}
        for key, (value_type, value_bytes) in serialized_cache.items():
            if value_type in ('tensor', 'array'):
                cache_data[key] = deserialize_tensor(value_bytes)
            elif value_type == 'pickle':
                cache_data[key] = pickle.loads(value_bytes)
            else:
                logger.warning(f"Unknown value type: {value_type}")
                cache_data[key] = None
        
        return cache_data
    
    except Exception as e:
        logger.error(f"Error deserializing cache: {e}")
        raise


def chunk_large_tensor(tensor, chunk_size_mb: int = 100) -> List[bytes]:
    """
    Split a large tensor into smaller chunks for efficient transmission.
    
    Args:
        tensor: Tensor to chunk
        chunk_size_mb: Maximum chunk size in megabytes
    
    Returns:
        List of serialized tensor chunks
    """
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")
    
    # Calculate chunk size in elements
    bytes_per_element = tensor.itemsize
    elements_per_mb = 1024 * 1024 // bytes_per_element
    chunk_size_elements = chunk_size_mb * elements_per_mb
    
    # Flatten tensor
    flat_tensor = tensor.reshape(-1)
    total_elements = flat_tensor.size
    
    # Split into chunks
    chunks = []
    for i in range(0, total_elements, chunk_size_elements):
        chunk = flat_tensor[i:i+chunk_size_elements]
        chunk_tensor = chunk.reshape((-1,) + tensor.shape[1:])
        chunks.append(serialize_tensor(chunk_tensor))
    
    return chunks


def reassemble_tensor_chunks(chunks: List[bytes], original_shape: Optional[Tuple[int, ...]] = None) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Reassemble tensor chunks back into a single tensor.
    
    Args:
        chunks: List of serialized tensor chunks
        original_shape: Optional original tensor shape
    
    Returns:
        Reassembled tensor
    """
    # Deserialize chunks
    deserialized_chunks = [deserialize_tensor(chunk) for chunk in chunks]
    
    if not deserialized_chunks:
        raise ValueError("No chunks provided")
    
    # Concatenate chunks
    if TORCH_AVAILABLE and isinstance(deserialized_chunks[0], torch.Tensor):
        concatenated = torch.cat([chunk.flatten() for chunk in deserialized_chunks])
        if original_shape:
            concatenated = concatenated.reshape(original_shape)
    else:
        concatenated = np.concatenate([chunk.flatten() for chunk in deserialized_chunks])
        if original_shape:
            concatenated = concatenated.reshape(original_shape)
    
    return concatenated