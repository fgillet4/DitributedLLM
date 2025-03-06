"""
Compute Engine for DistributedLLM worker nodes.

The ComputeEngine handles computational tasks assigned by the coordinator,
executes them efficiently, and reports results back to the coordinator.
"""

import logging
import socket
import threading
import time
import json
import os
import platform
import psutil
import numpy as np
import queue
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid

from src.worker.resource_monitor import ResourceMonitor
from src.utils.networking import send_message, receive_message

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Representation of a task assigned to this worker."""
    id: str
    type: str
    parameters: Dict[str, Any]
    received_at: float = time.time()
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: Optional[str] = None


class ComputeEngine:
    """
    Handles computational tasks for the worker node, including:
    1. Executing assigned tasks (model layer computation, token generation, etc.)
    2. Managing local resources (memory, compute)
    3. Reporting results and performance metrics to the coordinator
    """
    
    def __init__(self, worker_config, coordinator_config, network_config):
        """Initialize the compute engine with configuration details."""
        self.worker_config = worker_config
        self.coordinator_config = coordinator_config
        self.network_config = network_config
        
        # Generate a unique worker ID if not provided
        self.worker_id = worker_config.get("id", f"worker_{uuid.uuid4().hex[:8]}")
        
        # Task management
        self.tasks = {}
        self.task_queue = queue.Queue()
        self.current_task = None
        
        # Model data and state
        self.model_layers = {}
        self.loaded_weights = set()
        self.tokenizer = None
        
        # Communication with coordinator
        self.coordinator_socket = None
        self.heartbeat_interval = network_config.get("heartbeat_interval_seconds", 5)
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Execution control
        self.running = False
        self.threads = []
    
    def connect(self):
        """Connect to the coordinator."""
        try:
            # Create a socket connection to the coordinator
            self.coordinator_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.coordinator_socket.connect((
                self.coordinator_config["host"],
                self.coordinator_config["port"]
            ))
            
            # Send registration message
            registration_msg = {
                "type": "register",
                "worker_id": self.worker_id,
                "capabilities": self._gather_capabilities(),
                "port": self.worker_config.get("port", 5556)
            }
            
            send_message(self.coordinator_socket, registration_msg)
            
            # Wait for acknowledgment
            response = receive_message(self.coordinator_socket)
            
            if response["type"] != "register_ack" or response["status"] != "success":
                logger.error(f"Registration failed: {response}")
                self.coordinator_socket.close()
                self.coordinator_socket = None
                return False
            
            logger.info(f"Successfully connected to coordinator as {self.worker_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to coordinator: {e}")
            if self.coordinator_socket:
                self.coordinator_socket.close()
                self.coordinator_socket = None
            return False
    
    def start(self):
        """Start the compute engine and all its threads."""
        if self.running:
            logger.warning("Compute engine is already running")
            return
        
        if not self.coordinator_socket:
            logger.error("Cannot start compute engine: not connected to coordinator")
            return
        
        self.running = True
        
        # Start the main message handling thread
        message_thread = threading.Thread(target=self._handle_messages, daemon=True)
        message_thread.start()
        self.threads.append(message_thread)
        
        # Start the task processing thread
        task_thread = threading.Thread(target=self._process_tasks, daemon=True)
        task_thread.start()
        self.threads.append(task_thread)
        
        # Start the heartbeat thread
        heartbeat_thread = threading.Thread(target=self._send_heartbeats, daemon=True)
        heartbeat_thread.start()
        self.threads.append(heartbeat_thread)
        
        # Start the resource monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        monitor_thread.start()
        self.threads.append(monitor_thread)
        
        logger.info(f"Compute engine started on {self.worker_id}")
    
    def shutdown(self):
        """Gracefully shut down the compute engine."""
        if not self.running:
            return
        
        logger.info("Shutting down compute engine...")
        self.running = False
        
        # Close connection to coordinator
        if self.coordinator_socket:
            try:
                self.coordinator_socket.close()
            except:
                pass
            self.coordinator_socket = None
        
        # Wait for threads to terminate
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        logger.info("Compute engine shutdown complete")
    
    def _gather_capabilities(self):
        """Gather system capabilities and specifications."""
        capabilities = {}
        
        # CPU information
        capabilities["cpu_cores"] = os.cpu_count()
        capabilities["cpu_type"] = platform.processor()
        
        # Memory information
        mem_info = psutil.virtual_memory()
        capabilities["ram_gb"] = round(mem_info.total / (1024 ** 3), 1)
        
        # GPU information (could be extended with proper GPU detection)
        capabilities["gpu"] = "none"  # Default to none
        capabilities["gpu_memory_gb"] = 0
        
        # OS information
        capabilities["os"] = platform.system().lower()
        capabilities["platform"] = platform.platform()
        
        # Disk information
        disk_info = psutil.disk_usage('/')
        capabilities["disk_total_gb"] = round(disk_info.total / (1024 ** 3), 1)
        capabilities["disk_free_gb"] = round(disk_info.free / (1024 ** 3), 1)
        
        # Network information
        capabilities["hostname"] = socket.gethostname()
        
        # Worker config overrides (if specified in config)
        for key, value in self.worker_config.get("capabilities", {}).items():
            capabilities[key] = value
        
        return capabilities
    
    def _handle_messages(self):
        """Handle incoming messages from the coordinator."""
        while self.running and self.coordinator_socket:
            try:
                message = receive_message(self.coordinator_socket)
                
                if not message:
                    logger.error("Connection to coordinator lost")
                    self.running = False
                    break
                
                message_type = message.get("type")
                
                if message_type == "task_assignment":
                    self._handle_task_assignment(message)
                
                elif message_type == "heartbeat_ack":
                    # Just a heartbeat acknowledgment, nothing to do
                    pass
                
                elif message_type == "shutdown":
                    logger.info("Received shutdown command from coordinator")
                    self.shutdown()
                    break
                
                elif message_type == "model_update":
                    # Handle model update (e.g., new weights, configuration)
                    self._handle_model_update(message)
                
                elif message_type == "status_request":
                    # Send detailed status information
                    self._send_status_update()
                
                else:
                    logger.warning(f"Received unknown message type: {message_type}")
            
            except Exception as e:
                logger.error(f"Error handling messages: {e}")
                if self.running:
                    # Try to reconnect
                    self._attempt_reconnect()
    
    def _handle_task_assignment(self, message):
        """Handle a task assignment message from the coordinator."""
        task_id = message["task_id"]
        task_type = message["task_type"]
        parameters = message["parameters"]
        
        logger.info(f"Received task assignment: {task_id} ({task_type})")
        
        # Create task object
        task = Task(
            id=task_id,
            type=task_type,
            parameters=parameters
        )
        
        # Store and queue the task
        self.tasks[task_id] = task
        self.task_queue.put(task_id)
    
    def _handle_model_update(self, message):
        """Handle model update messages from the coordinator."""
        update_type = message.get("update_type")
        
        if update_type == "layer_weights":
            # Update specific layer weights
            layer_id = message["layer_id"]
            weights_data = message["weights"]
            
            logger.info(f"Updating weights for layer {layer_id}")
            
            # In a real implementation, this would deserialize and load the weights
            self.model_layers[layer_id] = {"weights": weights_data}
            self.loaded_weights.add(layer_id)
        
        elif update_type == "tokenizer":
            # Update tokenizer
            tokenizer_data = message["tokenizer"]
            logger.info("Updating tokenizer")
            
            # In a real implementation, this would load the tokenizer
            self.tokenizer = tokenizer_data
        
        elif update_type == "config":
            # Update model configuration
            config = message["config"]
            logger.info("Updating model configuration")
            
            # Store the configuration
            self.model_config = config
    
    def _process_tasks(self):
        """Process tasks from the task queue."""
        while self.running:
            try:
                # Get a task from the queue
                try:
                    task_id = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                task = self.tasks[task_id]
                
                # Update task status
                task.status = "running"
                task.started_at = time.time()
                self.current_task = task_id
                
                # Send status update
                self._send_status_update()
                
                # Execute the task based on its type
                try:
                    if task.type == "layer_computation":
                        self._execute_layer_computation(task)
                    
                    elif task.type == "token_generation":
                        self._execute_token_generation(task)
                    
                    elif task.type == "embedding":
                        self._execute_embedding(task)
                    
                    else:
                        raise ValueError(f"Unknown task type: {task.type}")
                    
                    # Task completed successfully
                    task.status = "completed"
                    task.completed_at = time.time()
                    task.error = None
                    
                    # Send result back to coordinator
                    self._send_task_result(task)
                    
                except Exception as e:
                    logger.error(f"Error executing task {task_id}: {e}")
                    
                    # Mark task as failed
                    task.status = "failed"
                    task.completed_at = time.time()
                    task.error = str(e)
                    
                    # Send failure notification
                    self._send_task_failure(task)
                
                finally:
                    # Clean up
                    self.current_task = None
                    self.task_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                time.sleep(1)  # Avoid spinning in case of persistent errors
    
    def _execute_layer_computation(self, task):
        """Execute a layer computation task."""
        # Extract task parameters
        layer_index = task.parameters.get("layer_index")
        batch_size = task.parameters.get("batch_size", 1)
        input_data = task.parameters.get("input_data")
        
        logger.info(f"Executing layer computation for layer {layer_index} with batch size {batch_size}")
        
        # Check if we have the layer weights
        if layer_index not in self.loaded_weights:
            raise ValueError(f"Layer {layer_index} weights not loaded")
        
        # In a real implementation, this would perform actual layer computation
        # For now, we'll simulate the computation with a sleep and random output
        computation_time = 0.1 + (0.05 * batch_size)  # Simulate computation time
        time.sleep(computation_time)
        
        # Generate simulated output (would be actual tensor computation in real impl)
        output_shape = task.parameters.get("output_shape", [batch_size, 1024])
        output_data = np.random.randn(*output_shape).tolist()
        
        # Return result
        task.result = {
            "layer_index": layer_index,
            "output_data": output_data,
            "computation_time": computation_time
        }
    
    def _execute_token_generation(self, task):
        """Execute a token generation task."""
        # Extract task parameters
        input_ids = task.parameters.get("input_ids")
        max_length = task.parameters.get("max_length", 20)
        temperature = task.parameters.get("temperature", 0.7)
        
        logger.info(f"Executing token generation with max_length {max_length}")
        
        # Ensure we have the tokenizer
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
        
        # In a real implementation, this would perform actual token generation
        # For now, we'll simulate generation with a sleep and random tokens
        generation_time = 0.2 + (0.1 * max_length)  # Simulate generation time
        time.sleep(generation_time)
        
        # Generate simulated tokens (would be actual generation in real impl)
        output_length = min(max_length, len(input_ids) + np.random.randint(5, 15))
        output_ids = input_ids + [np.random.randint(1000, 30000) for _ in range(output_length - len(input_ids))]
        
        # Return result
        task.result = {
            "output_ids": output_ids,
            "generation_time": generation_time
        }
    
    def _execute_embedding(self, task):
        """Execute an embedding computation task."""
        # Extract task parameters
        input_text = task.parameters.get("input_text")
        
        logger.info(f"Executing embedding computation for text: {input_text[:50]}...")
        
        # In a real implementation, this would compute actual embeddings
        # For now, we'll simulate computation with a sleep and random embeddings
        computation_time = 0.1 + (0.01 * len(input_text.split()))  # Simulate computation time
        time.sleep(computation_time)
        
        # Generate simulated embeddings
        embedding_dim = task.parameters.get("embedding_dim", 768)
        embeddings = np.random.randn(embedding_dim).tolist()
        
        # Return result
        task.result = {
            "embeddings": embeddings,
            "computation_time": computation_time
        }
    
    def _send_task_result(self, task):
        """Send task result back to the coordinator."""
        try:
            # Prepare result message
            result_msg = {
                "type": "task_result",
                "task_id": task.id,
                "status": "completed",
                "result": task.result,
                "metrics": {
                    "execution_time": task.completed_at - task.started_at,
                    "memory_used_mb": self.resource_monitor.get_memory_usage(),
                    "cpu_percent": self.resource_monitor.get_cpu_percent()
                }
            }
            
            # Send to coordinator
            send_message(self.coordinator_socket, result_msg)
            logger.info(f"Sent result for task {task.id}")
        
        except Exception as e:
            logger.error(f"Failed to send task result for {task.id}: {e}")
            self._attempt_reconnect()
    
    def _send_task_failure(self, task):
        """Send task failure notification to the coordinator."""
        try:
            # Prepare failure message
            failure_msg = {
                "type": "task_result",
                "task_id": task.id,
                "status": "failed",
                "error_message": task.error,
                "metrics": {
                    "execution_time": task.completed_at - task.started_at,
                    "memory_used_mb": self.resource_monitor.get_memory_usage(),
                    "cpu_percent": self.resource_monitor.get_cpu_percent()
                }
            }
            
            # Send to coordinator
            send_message(self.coordinator_socket, failure_msg)
            logger.info(f"Sent failure notification for task {task.id}")
        
        except Exception as e:
            logger.error(f"Failed to send task failure for {task.id}: {e}")
            self._attempt_reconnect()
    
    def _send_heartbeats(self):
        """Send periodic heartbeats to the coordinator."""
        while self.running and self.coordinator_socket:
            try:
                # Prepare heartbeat message
                heartbeat_msg = {
                    "type": "heartbeat",
                    "worker_id": self.worker_id,
                    "timestamp": time.time(),
                    "status": "busy" if self.current_task else "idle",
                    "current_task": self.current_task
                }
                
                # Send to coordinator
                send_message(self.coordinator_socket, heartbeat_msg)
                
                # Wait for next heartbeat interval
                time.sleep(self.heartbeat_interval)
            
            except Exception as e:
                logger.error(f"Failed to send heartbeat: {e}")
                self._attempt_reconnect()
                time.sleep(self.heartbeat_interval)
    
    def _send_status_update(self):
        """Send detailed status update to the coordinator."""
        try:
            # Get system metrics
            metrics = self.resource_monitor.get_metrics()
            
            # Prepare status message
            status_msg = {
                "type": "status_update",
                "worker_id": self.worker_id,
                "timestamp": time.time(),
                "status": "busy" if self.current_task else "idle",
                "current_task": self.current_task,
                "metrics": metrics,
                "loaded_weights": list(self.loaded_weights),
                "queue_length": self.task_queue.qsize()
            }
            
            # Send to coordinator
            send_message(self.coordinator_socket, status_msg)
            logger.debug("Sent status update to coordinator")
        
        except Exception as e:
            logger.error(f"Failed to send status update: {e}")
            self._attempt_reconnect()
    
    def _monitor_resources(self):
        """Monitor system resources and report metrics periodically."""
        while self.running:
            try:
                # Update resource metrics
                self.resource_monitor.update()
                
                # Log resource usage if a task is running
                if self.current_task:
                    metrics = self.resource_monitor.get_metrics()
                    logger.debug(f"Resource usage: Memory {metrics['memory_percent']}%, CPU {metrics['cpu_percent']}%")
                
                # Sleep before next update
                time.sleep(5)
            
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                time.sleep(10)
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to the coordinator if disconnected."""
        if not self.running:
            return False
        
        logger.info("Attempting to reconnect to coordinator...")
        
        # Close existing socket if any
        if self.coordinator_socket:
            try:
                self.coordinator_socket.close()
            except:
                pass
            self.coordinator_socket = None
        
        # Try to reconnect
        max_attempts = self.network_config.get("retry_attempts", 3)
        retry_delay = 5
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Reconnection attempt {attempt + 1}/{max_attempts}")
                if self.connect():
                    logger.info("Successfully reconnected to coordinator")
                    return True
            
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
            
            if attempt < max_attempts - 1:
                logger.info(f"Waiting {retry_delay}s before next attempt...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        logger.error(f"Failed to reconnect after {max_attempts} attempts")
        return False
        
    # Model sharding methods, inspired by the HuggingFace diffusers implementation
    def load_sharded_model(self, model_id, device_map=None):
        """
        Load a model that's sharded across multiple devices.
        Inspired by the HuggingFace approach to model sharding.
        
        Args:
            model_id: Identifier for the model to load
            device_map: How to distribute model across devices. Can be:
                - "auto": Use accelerate to automatically place layers
                - "balanced": Distribute evenly across GPUs
                - Dict mapping layer names to device IDs
        """
        from src.model.layers import ShardedModelLoader
        
        logger.info(f"Loading sharded model {model_id} with device_map={device_map}")
        
        # In a real implementation, this would use accelerate/transformers
        # to actually load the model across devices
        
        # For this simulated implementation, we'll just record the configuration
        self.model_id = model_id
        self.device_map = device_map
        
        # Simulate loading time based on worker capabilities
        gpu_memory = self.worker_config.get("capabilities", {}).get("gpu_memory_gb", 0)
        has_gpu = self.worker_config.get("capabilities", {}).get("gpu", "none") != "none"
        
        # Simulate longer load times for CPU-only or low memory scenarios
        if not has_gpu:
            load_time = 20 + np.random.rand() * 10  # 20-30 seconds
        else:
            load_time = max(1, 10 - gpu_memory/4) + np.random.rand() * 5  # Faster with more GPU memory
        
        logger.info(f"Loading model will take approximately {load_time:.1f} seconds")
        time.sleep(load_time / 10)  # Simulate partial loading
        
        # Simulate the sharded model structures
        self.model_parts = {
            "encoder": {"device": 0 if has_gpu else "cpu", "loaded": True},
            "decoder": {"device": 0 if has_gpu else "cpu", "loaded": True},
            "transformer_blocks": {}
        }
        
        # Simulate transformer blocks distribution
        num_layers = 24  # Typical for medium-sized models
        
        if device_map == "balanced" and has_gpu:
            # Distribute layers evenly
            devices = [0, 1] if gpu_memory >= 8 else [0]  # Use 2 GPUs if enough memory
            for i in range(num_layers):
                device_id = devices[i % len(devices)]
                self.model_parts["transformer_blocks"][f"layer_{i}"] = {
                    "device": device_id,
                    "loaded": True
                }
                # Simulate progressive loading
                if i % 4 == 0:
                    time.sleep(load_time / 20)
        
        elif device_map == "auto" or not has_gpu:
            # Auto or CPU fallback
            for i in range(num_layers):
                # For "auto", simulate accelerate's placement strategy
                if has_gpu and gpu_memory >= 16:
                    # All on GPU for high memory
                    device = 0
                elif has_gpu and gpu_memory >= 8:
                    # Early layers on GPU, later on CPU
                    device = 0 if i < 16 else "cpu"
                else:
                    # All on CPU for low memory
                    device = "cpu"
                
                self.model_parts["transformer_blocks"][f"layer_{i}"] = {
                    "device": device,
                    "loaded": True
                }
                # Simulate progressive loading
                if i % 4 == 0:
                    time.sleep(load_time / 20)
        
        logger.info(f"Model {model_id} loaded and sharded across devices")
        return True
    
    def offload_to_cpu(self, layers=None):
        """
        Offload specified layers to CPU to free up GPU memory.
        
        Args:
            layers: List of layer names to offload, or None for all
        """
        if not hasattr(self, 'model_parts'):
            logger.warning("No model loaded, nothing to offload")
            return
        
        logger.info(f"Offloading {'specified' if layers else 'all'} layers to CPU")
        
        # Simulate offloading with a short delay
        time.sleep(0.5)
        
        if layers is None:
            # Offload all GPU-resident layers
            for part_name, part_info in self.model_parts.items():
                if isinstance(part_info, dict) and part_info.get("device") not in ["cpu", None]:
                    part_info["device"] = "cpu"
                    logger.debug(f"Offloaded {part_name} to CPU")
            
            # Handle nested structures like transformer blocks
            if "transformer_blocks" in self.model_parts:
                for layer_name, layer_info in self.model_parts["transformer_blocks"].items():
                    if layer_info.get("device") not in ["cpu", None]:
                        layer_info["device"] = "cpu"
                        logger.debug(f"Offloaded {layer_name} to CPU")
        else:
            # Offload only specified layers
            for layer_name in layers:
                if layer_name in self.model_parts:
                    if self.model_parts[layer_name].get("device") not in ["cpu", None]:
                        self.model_parts[layer_name]["device"] = "cpu"
                        logger.debug(f"Offloaded {layer_name} to CPU")
                else:
                    # Check in transformer blocks
                    for block_name, block_info in self.model_parts.get("transformer_blocks", {}).items():
                        if block_name == layer_name and block_info.get("device") not in ["cpu", None]:
                            block_info["device"] = "cpu"
                            logger.debug(f"Offloaded {layer_name} to CPU")
        
        logger.info("Layer offloading complete")
        
        # Report memory status after offloading
        metrics = self.resource_monitor.get_metrics()
        logger.info(f"GPU memory after offloading: {metrics.get('gpu_memory_used_mb', 0)}/{metrics.get('gpu_memory_total_mb', 0)} MB")
        
    def free_memory(self):
        """Free up memory by running garbage collection and clearing caches."""
        import gc
        import torch
        
        logger.info("Freeing memory and clearing caches")
        
        # Run Python's garbage collector
        gc.collect()
        
        # Clear PyTorch caches if available
        if 'torch' in sys.modules:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
        
        # Clear local caches
        self.model_parts = {}
        self.loaded_weights = set()
        
        # Report memory status after cleanup
        metrics = self.resource_monitor.get_metrics()
        logger.info(f"Memory after cleanup: {metrics['memory_used_mb']}/{metrics['memory_total_mb']} MB")
        if 'gpu_memory_used_mb' in metrics:
            logger.info(f"GPU memory after cleanup: {metrics['gpu_memory_used_mb']}/{metrics['gpu_memory_total_mb']} MB")