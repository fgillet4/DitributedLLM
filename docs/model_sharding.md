# DistributedLLM: Model Sharding and Distribution

This document explains how the DistributedLLM system distributes and shards large language models across multiple devices and machines.

## Core Concepts

DistributedLLM enables running large language models on distributed hardware by splitting model components across multiple devices (GPUs and CPUs) and multiple machines. This allows running models that would otherwise be too large for any single device.

## Model Sharding Architecture

### 1. Device Mapping Strategies

The system supports multiple distribution strategies through the `ShardedModelLoader` class:

- **Auto**: Automatically places model components based on available memory
- **Balanced**: Distributes layers evenly across available devices
- **Custom**: Enables explicit mapping of components to specific devices

```python
# Example device mapping
device_map = {
    "embedding": "cuda:0",
    "transformer.blocks.0": "cuda:0",
    "transformer.blocks.1": "cuda:1",
    "transformer.blocks.2": "cuda:1",
    "lm_head": "cpu"
}
```

### 2. Layer-Based Distribution

Models are divided into distinct components that can be placed on different devices:

- **Embeddings**: Token embedding lookup tables
- **Transformer Blocks**: Multiple attention and feed-forward layers
- **Layer Normalization**: Normalization layers between components
- **LM Head**: Output projection layer

### 3. Cross-Device Communication

When running inference, the system automatically handles data transfer between devices:

```python
# From ShardedModel.forward()
for block in self.blocks:
    # Ensure hidden states are on the right device for this block
    block_device = block.input_layernorm.weight.device
    if hidden_states.device != block_device:
        hidden_states = hidden_states.to(block_device)
    
    hidden_states = block(hidden_states, attention_mask)
```

### 4. Cross-Machine Distribution

The system distributes computation across multiple worker machines:

1. **Layer Allocation**: The coordinator assigns specific layers to specific workers
2. **Task Distribution**: Computation is broken into tasks that are assigned to workers
3. **Result Aggregation**: Results from different workers are combined into the final output

## Distributed Inference Process

The inference process follows these steps when distributed across multiple machines:

1. **Input Processing**:
   - Coordinator tokenizes the input prompt
   - Task planning for each model component

2. **Forward Pass**:
   - Workers compute their assigned layers sequentially
   - Data flows between workers: Worker1 (embedding) → Worker2 (layers 0-3) → Worker3 (layers 4-8) → etc.

3. **Token Generation**:
   - The final layer produces token probabilities
   - Coordinator samples the next token and prepares for the next step
   - Process repeats for each new token

4. **Result Collection**:
   - Generated tokens are assembled into the final output text
   - Results returned to the client

## Dynamic Load Balancing

The system continually optimizes distribution based on observed performance:

1. **Performance Monitoring**:
   - Track execution times per worker per task type
   - Monitor resource utilization (CPU, memory, GPU)

2. **Workload Rebalancing**:
   - Redistributes layers based on worker performance
   - Adjusts task assignments to balance the load

```python
# From scheduler.py
def _recalculate_layer_allocation(self):
    # Distribute layers proportionally to performance scores
    for worker_id, score in worker_metrics.items():
        layer_count = max(1, int(total_layers * score))
        # ...assign layers based on performance...
```

## Memory Optimization Techniques

The system implements several techniques to optimize memory usage:

1. **Selective Loading**: Only load model weights needed for assigned layers
2. **CPU Offloading**: Move less frequently used layers to CPU memory
3. **KV-Cache Management**: Efficient management of attention key-value caches
4. **Quantization Support**: Int8, FP16, and other reduced precision formats
5. **Gradient-Free Inference**: No need to store activation gradients during inference

## Fault Tolerance

The distributed architecture includes mechanisms for fault tolerance:

1. **Worker Failure Detection**: Through heartbeat monitoring
2. **Task Reassignment**: Failed tasks are reassigned to other workers
3. **Graceful Degradation**: System can continue with fewer workers if needed
4. **Straggler Mitigation**: Slow workers get fewer or simpler tasks

## Performance Considerations

Key factors affecting distributed inference performance:

1. **Communication Overhead**: Network bandwidth and latency between machines
2. **Load Balancing Quality**: How evenly work is distributed
3. **Worker Heterogeneity**: Different capabilities across machines
4. **Layer Allocation Strategy**: Which layers are assigned to which workers