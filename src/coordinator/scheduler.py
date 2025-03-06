"""
Scheduler for the DistributedLLM system.

The Scheduler is the core of the Boss-Worker model, responsible for:
1. Managing worker nodes
2. Partitioning the model across workers
3. Distributing workload based on worker capabilities
4. Handling worker failures
5. Collecting and aggregating results
"""

import logging
import threading
import time
import json
import socket
import queue
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Any

from src.coordinator.performance_monitor import PerformanceMonitor
from src.coordinator.result_aggregator import ResultAggregator
from src.utils.networking import send_message, receive_message

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Representation of a computational task to be assigned to a worker."""
    id: str
    type: str  # "layer_computation", "token_generation", "embedding", etc.
    parameters: Dict[str, Any]
    priority: int
    assigned_worker: Optional[str] = None
    status: str = "pending"  # pending, assigned, completed, failed
    created_at: float = time.time()
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    retries: int = 0


@dataclass
class Worker:
    """Representation of a worker node."""
    id: str
    host: str
    port: int
    capabilities: Dict[str, Any]
    status: str = "disconnected"  # disconnected, connected, busy, idle
    current_task: Optional[str] = None
    performance_history: List[Dict[str, float]] = None
    last_heartbeat: float = 0
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []


class Scheduler:
    """
    Responsible for distributing tasks to workers and managing the overall workflow.
    Implements the Boss node in the Boss-Worker pattern.
    """
    
    def __init__(self, coordinator_config, workers_config, network_config, discovery_config):
        """Initialize the scheduler with configuration details."""
        self.coordinator_config = coordinator_config
        self.network_config = network_config
        self.discovery_config = discovery_config
        
        # Initialize data structures
        self.workers: Dict[str, Worker] = {}
        self.task_queue = queue.PriorityQueue()  # Priority queue of tasks
        self.tasks: Dict[str, Task] = {}  # Tasks indexed by ID
        self.active_connections: Dict[str, socket.socket] = {}  # Active worker sockets
        
        # Layer allocation tracking
        self.layer_allocation: Dict[str, List[int]] = {}  # Worker ID -> list of layer indices
        
        # Result aggregation
        self.result_aggregator = ResultAggregator()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Threading management
        self.running = False
        self.threads = []
        
        # Initialize workers from config
        for worker_config in workers_config:
            worker = Worker(
                id=worker_config["id"],
                host=worker_config["host"],
                port=worker_config["port"],
                capabilities=worker_config["capabilities"]
            )
            self.workers[worker.id] = worker
    
    def start(self):
        """Start the scheduler and all its threads."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        
        # Start the server socket to accept connections from workers
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()
        self.threads.append(server_thread)
        
        # Start heartbeat monitoring thread
        heartbeat_thread = threading.Thread(target=self._monitor_heartbeats, daemon=True)
        heartbeat_thread.start()
        self.threads.append(heartbeat_thread)
        
        # Start task dispatcher thread
        dispatcher_thread = threading.Thread(target=self._dispatch_tasks, daemon=True)
        dispatcher_thread.start()
        self.threads.append(dispatcher_thread)
        
        # Start performance monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        monitor_thread.start()
        self.threads.append(monitor_thread)
        
        logger.info(f"Scheduler started with {len(self.workers)} configured workers")
    
    def start_discovery(self):
        """Start the automatic worker discovery service."""
        discovery_thread = threading.Thread(target=self._run_discovery_service, daemon=True)
        discovery_thread.start()
        self.threads.append(discovery_thread)
        logger.info("Worker discovery service started")
    
    def shutdown(self):
        """Gracefully shut down the scheduler."""
        if not self.running:
            return
        
        logger.info("Shutting down scheduler...")
        self.running = False
        
        # Notify all workers to disconnect
        for worker_id, conn in self.active_connections.items():
            try:
                send_message(conn, {"type": "shutdown"})
                conn.close()
            except:
                pass
        
        # Wait for threads to terminate
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        logger.info("Scheduler shutdown complete")
    
    def _run_server(self):
        """Run the server socket to accept connections from workers."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.coordinator_config["host"], self.coordinator_config["port"]))
            server_socket.listen(self.coordinator_config["max_workers"])
            logger.info(f"Server listening on {self.coordinator_config['host']}:{self.coordinator_config['port']}")
            
            while self.running:
                try:
                    client_socket, address = server_socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_worker_connection,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                except Exception as e:
                    if self.running:  # Only log if we're still supposed to be running
                        logger.error(f"Error accepting worker connection: {e}")
        
        except Exception as e:
            logger.critical(f"Failed to start server: {e}")
            self.running = False
        finally:
            server_socket.close()
    
    def _handle_worker_connection(self, client_socket, address):
        """Handle a connection from a worker."""
        worker_id = None
        
        try:
            # Set timeout for initial registration
            client_socket.settimeout(self.network_config["timeout_seconds"])
            
            # Wait for worker registration
            message = receive_message(client_socket)
            
            if message["type"] != "register":
                logger.warning(f"Received non-registration message from {address}: {message['type']}")
                client_socket.close()
                return
            
            # Extract worker information
            worker_id = message["worker_id"]
            capabilities = message["capabilities"]
            
            # Update or create worker record
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.status = "connected"
                worker.host = address[0]
                worker.capabilities.update(capabilities)
                logger.info(f"Worker {worker_id} reconnected from {address}")
            else:
                # New worker discovered
                worker = Worker(
                    id=worker_id,
                    host=address[0],
                    port=message.get("port", self.coordinator_config["port"]),
                    capabilities=capabilities,
                    status="connected"
                )
                self.workers[worker_id] = worker
                logger.info(f"New worker {worker_id} registered from {address}")
            
            # Send acknowledgment
            send_message(client_socket, {
                "type": "register_ack",
                "status": "success",
                "worker_id": worker_id
            })
            
            # Update connection tracking
            if worker_id in self.active_connections:
                # Close existing connection if there is one
                try:
                    self.active_connections[worker_id].close()
                except:
                    pass
            
            self.active_connections[worker_id] = client_socket
            worker.last_heartbeat = time.time()
            
            # Set socket to non-blocking for continued communication
            client_socket.settimeout(None)
            
            # Handle ongoing communication with this worker
            self._worker_communication_loop(worker_id, client_socket)
            
        except Exception as e:
            logger.error(f"Error handling worker connection from {address} ({worker_id if worker_id else 'unknown'}): {e}")
        finally:
            # Clean up on disconnect
            try:
                client_socket.close()
            except:
                pass
            
            if worker_id and worker_id in self.active_connections:
                del self.active_connections[worker_id]
                if worker_id in self.workers:
                    self.workers[worker_id].status = "disconnected"
                    logger.info(f"Worker {worker_id} disconnected")
    
    def _worker_communication_loop(self, worker_id, client_socket):
        """Handle ongoing communication with a connected worker."""
        worker = self.workers[worker_id]
        
        while self.running and worker_id in self.active_connections:
            try:
                message = receive_message(client_socket)
                
                if not message:
                    # Connection closed
                    break
                
                message_type = message.get("type")
                
                if message_type == "heartbeat":
                    # Update last heartbeat time
                    worker.last_heartbeat = time.time()
                    # Send heartbeat acknowledgment
                    send_message(client_socket, {"type": "heartbeat_ack"})
                
                elif message_type == "task_result":
                    # Process completed task result
                    task_id = message["task_id"]
                    status = message["status"]
                    result = message.get("result")
                    metrics = message.get("metrics", {})
                    
                    self._handle_task_completion(worker_id, task_id, status, result, metrics)
                
                elif message_type == "status_update":
                    # Update worker status information
                    new_status = message["status"]
                    metrics = message.get("metrics", {})
                    
                    worker.status = new_status
                    
                    # Update performance metrics
                    self.performance_monitor.update_worker_metrics(worker_id, metrics)
                
                elif message_type == "error":
                    # Handle worker-reported error
                    error_type = message.get("error_type", "unknown")
                    error_msg = message.get("error_message", "No message provided")
                    task_id = message.get("task_id")
                    
                    logger.error(f"Worker {worker_id} reported error: {error_type} - {error_msg}")
                    
                    if task_id:
                        self._handle_task_failure(worker_id, task_id, f"{error_type}: {error_msg}")
                
                else:
                    logger.warning(f"Received unknown message type '{message_type}' from worker {worker_id}")
            
            except Exception as e:
                logger.error(f"Error in communication with worker {worker_id}: {e}")
                break
    
    def _handle_task_completion(self, worker_id, task_id, status, result, metrics):
        """Process a completed task from a worker."""
        if task_id not in self.tasks:
            logger.warning(f"Received result for unknown task {task_id} from worker {worker_id}")
            return
        
        task = self.tasks[task_id]
        worker = self.workers[worker_id]
        
        # Update task status
        task.status = status
        task.completed_at = time.time()
        task.result = result
        
        # Update worker status
        worker.status = "idle"
        worker.current_task = None
        
        # Record performance metrics
        elapsed_time = task.completed_at - task.started_at
        performance_record = {
            "task_id": task_id,
            "task_type": task.type,
            "elapsed_time": elapsed_time,
            **metrics
        }
        worker.performance_history.append(performance_record)
        
        # Update performance monitor
        self.performance_monitor.record_task_completion(worker_id, task_id, performance_record)
        
        logger.info(f"Task {task_id} completed by worker {worker_id} in {elapsed_time:.2f}s with status {status}")
        
        # Process result if task was successful
        if status == "completed":
            self.result_aggregator.add_result(task)
    
    def _handle_task_failure(self, worker_id, task_id, error_message):
        """Handle a failed task."""
        if task_id not in self.tasks:
            logger.warning(f"Received failure for unknown task {task_id} from worker {worker_id}")
            return
        
        task = self.tasks[task_id]
        worker = self.workers[worker_id]
        
        # Update task status
        task.status = "failed"
        task.completed_at = time.time()
        task.retries += 1
        
        # Update worker status
        worker.status = "idle"
        worker.current_task = None
        
        logger.warning(f"Task {task_id} failed on worker {worker_id}: {error_message}")
        
        # Check if the task should be retried
        max_retries = 3  # Should come from config
        if task.retries < max_retries:
            logger.info(f"Requeuing task {task_id} for retry (attempt {task.retries}/{max_retries})")
            # Reset task for requeuing
            task.status = "pending"
            task.assigned_worker = None
            task.started_at = None
            task.completed_at = None
            # Put it back in the queue with higher priority
            self.task_queue.put((task.priority - 10, task.id))  # Lower number = higher priority
        else:
            logger.error(f"Task {task_id} failed permanently after {task.retries} attempts")
            # Could implement fallback strategies here
    
    def _dispatch_tasks(self):
        """Continuously dispatch tasks to available workers."""
        while self.running:
            try:
                # Find idle workers
                idle_workers = [
                    worker_id for worker_id, worker in self.workers.items()
                    if worker.status == "idle" and worker_id in self.active_connections
                ]
                
                if not idle_workers or self.task_queue.empty():
                    # No work to do or no workers available
                    time.sleep(0.1)
                    continue
                
                # Get the highest priority task
                _, task_id = self.task_queue.get()
                task = self.tasks[task_id]
                
                # Choose the best worker for this task
                worker_id = self._select_worker_for_task(task, idle_workers)
                
                if not worker_id:
                    # No suitable worker found, put the task back
                    self.task_queue.put((task.priority, task_id))
                    time.sleep(0.1)
                    continue
                
                # Assign the task
                self._assign_task_to_worker(task, worker_id)
            
            except Exception as e:
                logger.error(f"Error in task dispatch loop: {e}")
                time.sleep(1)  # Avoid spinning in case of persistent errors
    
    def _select_worker_for_task(self, task, idle_workers):
        """Select the best worker for a given task based on capabilities and performance history."""
        if not idle_workers:
            return None
        
        # For model layer computation tasks, try to respect existing layer allocation
        if task.type == "layer_computation":
            layer_index = task.parameters.get("layer_index")
            if layer_index is not None:
                # Check if this layer is already allocated to a worker
                for worker_id, layers in self.layer_allocation.items():
                    if layer_index in layers and worker_id in idle_workers:
                        return worker_id
        
        # Calculate a score for each worker
        worker_scores = {}
        for worker_id in idle_workers:
            worker = self.workers[worker_id]
            
            # Start with the base capability score
            score = self._calculate_capability_score(worker, task)
            
            # Adjust based on performance history
            if worker.performance_history:
                perf_score = self._calculate_performance_score(worker, task)
                score *= perf_score
            
            # Network proximity could be considered here
            
            worker_scores[worker_id] = score
        
        # Return the worker with the highest score
        if worker_scores:
            return max(worker_scores.items(), key=lambda x: x[1])[0]
        
        # If no suitable worker found, just return the first idle one
        return idle_workers[0] if idle_workers else None
    
    def _calculate_capability_score(self, worker, task):
        """Calculate a score representing how well a worker's capabilities match a task's requirements."""
        capabilities = worker.capabilities
        
        # Base score starts at 1.0
        score = 1.0
        
        # Adjust score based on relevant capabilities for this task type
        if task.type == "layer_computation":
            # For layer computation, GPU memory and compute power are important
            if "gpu_memory_gb" in capabilities:
                score *= (1.0 + capabilities["gpu_memory_gb"] / 10.0)
            if "gpu" in capabilities and capabilities["gpu"] != "none":
                score *= 1.5  # Prefer GPU workers for layer computation
        
        elif task.type == "token_generation":
            # For token generation, CPU cores and memory are important
            if "cpu_cores" in capabilities:
                score *= (1.0 + capabilities["cpu_cores"] / 16.0)
            if "ram_gb" in capabilities:
                score *= (1.0 + capabilities["ram_gb"] / 32.0)
        
        # Priority adjustment
        if "priority" in capabilities:
            score /= max(1, capabilities["priority"])  # Lower priority number = higher priority
        
        return score
    
    def _calculate_performance_score(self, worker, task):
        """Calculate a performance score based on historical performance for similar tasks."""
        # Get recent history for similar tasks
        similar_task_history = [
            record for record in worker.performance_history[-10:]
            if record["task_type"] == task.type
        ]
        
        if not similar_task_history:
            return 1.0  # Neutral score if no history
        
        # Calculate average completion time
        avg_time = sum(record["elapsed_time"] for record in similar_task_history) / len(similar_task_history)
        
        # Normalize against expected time (could be refined with global average across workers)
        expected_time = 1.0  # This would ideally come from global statistics
        
        # Convert to a score where faster is better (inverse relationship)
        time_score = expected_time / max(avg_time, 0.001)
        
        # Cap to reasonable range
        return max(0.5, min(2.0, time_score))
    
    def _assign_task_to_worker(self, task, worker_id):
        """Assign a task to a specific worker and send the instructions."""
        worker = self.workers[worker_id]
        
        # Update task status
        task.assigned_worker = worker_id
        task.status = "assigned"
        task.started_at = time.time()
        
        # Update worker status
        worker.status = "busy"
        worker.current_task = task.id
        
        # For layer computation tasks, update layer allocation
        if task.type == "layer_computation":
            layer_index = task.parameters.get("layer_index")
            if layer_index is not None:
                if worker_id not in self.layer_allocation:
                    self.layer_allocation[worker_id] = []
                if layer_index not in self.layer_allocation[worker_id]:
                    self.layer_allocation[worker_id].append(layer_index)
        
        # Prepare task message
        task_message = {
            "type": "task_assignment",
            "task_id": task.id,
            "task_type": task.type,
            "parameters": task.parameters
        }
        
        # Send task to worker
        try:
            connection = self.active_connections[worker_id]
            send_message(connection, task_message)
            logger.info(f"Task {task.id} assigned to worker {worker_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send task {task.id} to worker {worker_id}: {e}")
            # Revert task and worker status
            task.assigned_worker = None
            task.status = "pending"
            task.started_at = None
            worker.status = "idle"
            worker.current_task = None
            # Requeue the task
            self.task_queue.put((task.priority, task.id))
            return False
    
    def _monitor_heartbeats(self):
        """Monitor worker heartbeats and handle disconnections."""
        while self.running:
            try:
                current_time = time.time()
                
                for worker_id, worker in list(self.workers.items()):
                    # Skip disconnected workers
                    if worker.status == "disconnected":
                        continue
                    
                    # Check if heartbeat is overdue
                    heartbeat_age = current_time - worker.last_heartbeat
                    timeout = self.network_config["timeout_seconds"]
                    
                    if heartbeat_age > timeout:
                        logger.warning(f"Worker {worker_id} heartbeat timeout ({heartbeat_age:.1f}s > {timeout}s)")
                        
                        # Handle any assigned tasks
                        if worker.current_task and worker.current_task in self.tasks:
                            task = self.tasks[worker.current_task]
                            self._handle_task_failure(worker_id, task.id, "Worker heartbeat timeout")
                        
                        # Mark as disconnected
                        worker.status = "disconnected"
                        
                        # Close connection if it exists
                        if worker_id in self.active_connections:
                            try:
                                self.active_connections[worker_id].close()
                            except:
                                pass
                            del self.active_connections[worker_id]
                
                # Sleep before next check
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in heartbeat monitoring: {e}")
                time.sleep(5)  # Longer sleep on error
    
    def _monitor_performance(self):
        """Monitor system performance and adjust workload distribution if needed."""
        while self.running:
            try:
                # Get current performance metrics
                metrics = self.performance_monitor.get_current_metrics()
                
                # Check if rebalancing is needed
                if self._should_rebalance_workload(metrics):
                    logger.info("Performance imbalance detected, rebalancing workload")
                    self._rebalance_workload()
                
                # Sleep before next check
                time.sleep(10)  # Check every 10 seconds
            
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(30)  # Longer sleep on error
    
    def _should_rebalance_workload(self, metrics):
        """Determine if workload rebalancing is needed based on performance metrics."""
        # Implementation will depend on specific metrics tracked
        if not metrics or not metrics.get("worker_metrics"):
            return False
        
        worker_metrics = metrics["worker_metrics"]
        
        # Only consider active workers
        active_workers = [
            worker_id for worker_id in worker_metrics
            if self.workers.get(worker_id) and self.workers[worker_id].status != "disconnected"
        ]
        
        if len(active_workers) < 2:
            return False  # Need at least 2 active workers to rebalance
        
        # Check utilization variance
        if "utilization" in worker_metrics[active_workers[0]]:
            utilizations = [worker_metrics[w_id]["utilization"] for w_id in active_workers]
            util_variance = np.var(utilizations)
            util_mean = np.mean(utilizations)
            
            # If variance is high relative to mean, rebalance
            if util_variance > 0.1 * util_mean:
                return True
        
        # Check task completion time variance
        if "avg_task_time" in worker_metrics[active_workers[0]]:
            task_times = [worker_metrics[w_id]["avg_task_time"] for w_id in active_workers]
            time_variance = np.var(task_times)
            time_mean = np.mean(task_times)
            
            # If time variance is high relative to mean, rebalance
            if time_variance > 0.25 * time_mean:
                return True
        
        return False
    
    def _rebalance_workload(self):
        """Rebalance workload distribution among workers."""
        # This would typically involve:
        # 1. Recalculating layer allocations
        # 2. Potentially reassigning pending tasks
        # 3. Updating worker weights for future task assignment
        
        # For model layers, reallocation is more complex since reassigning
        # already-running computations is expensive
        
        # For now, just recalculate layer allocations for future tasks
        self._recalculate_layer_allocation()
    
    def _recalculate_layer_allocation(self):
        """Recalculate the allocation of model layers to workers based on performance."""
        # Get active workers
        active_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if worker.status != "disconnected" and worker_id in self.active_connections
        ]
        
        if not active_workers:
            return
        
        # Get worker performance metrics
        worker_metrics = {}
        for worker_id in active_workers:
            worker = self.workers[worker_id]
            
            # Calculate a performance score for this worker
            if worker.performance_history:
                recent_history = worker.performance_history[-10:]
                avg_time = sum(rec["elapsed_time"] for rec in recent_history) / len(recent_history)
                performance_score = 1.0 / max(avg_time, 0.001)  # Inverse of time
            else:
                # No history, use capability-based estimation
                capabilities = worker.capabilities
                performance_score = (
                    capabilities.get("cpu_cores", 1) * 0.3 +
                    capabilities.get("ram_gb", 1) / 8.0 * 0.3 +
                    (10.0 if capabilities.get("gpu") != "none" else 1.0) * 0.4
                )
            
            worker_metrics[worker_id] = performance_score
        
        # Normalize scores to sum to 1.0
        total_score = sum(worker_metrics.values())
        for worker_id in worker_metrics:
            worker_metrics[worker_id] /= total_score
        
        # Reset layer allocation
        self.layer_allocation = {worker_id: [] for worker_id in active_workers}
        
        # Assume we know the total number of layers
        total_layers = 40  # Should come from model config
        
        # Distribute layers proportionally to performance scores
        remaining_layers = list(range(total_layers))
        
        # First pass: distribute based on proportion
        for worker_id, score in worker_metrics.items():
            layer_count = max(1, int(total_layers * score))
            
            # Take layers from the remaining pool
            if remaining_layers:
                assigned_layers = remaining_layers[:layer_count]
                remaining_layers = remaining_layers[layer_count:]
                self.layer_allocation[worker_id].extend(assigned_layers)
        
        # Second pass: assign any remaining layers
        for worker_id in active_workers:
            if not remaining_layers:
                break
            self.layer_allocation[worker_id].append(remaining_layers.pop(0))
        
        logger.info(f"Recalculated layer allocation: {self.layer_allocation}")
    
    def _run_discovery_service(self):
        """Run the automatic worker discovery service."""
        # Create UDP socket for broadcasting
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        discovery_port = self.discovery_config["broadcast_port"]
        interval = self.discovery_config["broadcast_interval_seconds"]
        
        try:
            # Prepare discovery message
            discovery_message = {
                "type": "coordinator_announcement",
                "coordinator_host": self.coordinator_config["host"],
                "coordinator_port": self.coordinator_config["port"],
                "timestamp": time.time()
            }
            
            # Binary representation
            message_data = json.dumps(discovery_message).encode('utf-8')
            
            logger.info(f"Starting discovery broadcasts on port {discovery_port}")
            
            while self.running:
                try:
                    # Broadcast discovery message
                    sock.sendto(message_data, ('<broadcast>', discovery_port))
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error broadcasting discovery message: {e}")
                    time.sleep(interval)
        
        finally:
            sock.close()
    
    def add_task(self, task_type, parameters, priority=100):
        """Add a new task to the queue."""
        task_id = f"task_{int(time.time())}_{len(self.tasks)}"
        
        task = Task(
            id=task_id,
            type=task_type,
            parameters=parameters,
            priority=priority
        )
        
        self.tasks[task_id] = task
        self.task_queue.put((priority, task_id))
        
        logger.info(f"Added task {task_id} of type {task_type} with priority {priority}")
        return task_id
    
    def get_task_status(self, task_id):
        """Get the current status of a task."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        return {
            "id": task.id,
            "type": task.type,
            "status": task.status,
            "assigned_worker": task.assigned_worker,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "elapsed_time": (task.completed_at - task.started_at) if task.completed_at and task.started_at else None,
            "result_available": task.result is not None
        }
    
    def get_result(self, task_id):
        """Get the result of a completed task."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        if task.status != "completed" or task.result is None:
            return None
        
        return task.result
    
    def get_worker_status(self):
        """Get status information about all workers."""
        worker_status = {}
        
        for worker_id, worker in self.workers.items():
            is_connected = worker_id in self.active_connections
            
            worker_status[worker_id] = {
                "status": worker.status,
                "connected": is_connected,
                "current_task": worker.current_task,
                "capabilities": worker.capabilities,
                "last_heartbeat": worker.last_heartbeat,
                "heartbeat_age": time.time() - worker.last_heartbeat if is_connected else None,
                "performance_summary": self._summarize_worker_performance(worker_id)
            }
        
        return worker_status
    
    def _summarize_worker_performance(self, worker_id):
        """Summarize performance metrics for a worker."""
        worker = self.workers.get(worker_id)
        if not worker or not worker.performance_history:
            return {}
        
        # Get recent history
        recent_history = worker.performance_history[-20:]
        
        # Calculate averages
        avg_time = sum(rec["elapsed_time"] for rec in recent_history) / len(recent_history)
        
        # Group by task type
        task_types = {}
        for record in recent_history:
            task_type = record["task_type"]
            if task_type not in task_types:
                task_types[task_type] = []
            task_types[task_type].append(record["elapsed_time"])
        
        type_averages = {
            t: sum(times) / len(times)
            for t, times in task_types.items()
        }
        
        return {
            "avg_task_time": avg_time,
            "task_count": len(recent_history),
            "task_type_averages": type_averages
        }