"""
Communication module for DistributedLLM worker nodes.

Handles all network communication between workers and the coordinator,
including task assignment, result reporting, and heartbeats.
"""

import logging
import socket
import threading
import time
import queue
from typing import Dict, List, Optional, Any, Callable, Tuple

from src.utils.networking import send_message, receive_message, discover_nodes

logger = logging.getLogger(__name__)


class CoordinatorClient:
    """
    Client for communicating with the coordinator node.
    
    Handles sending and receiving messages to/from the coordinator,
    including task results, heartbeats, and status updates.
    """
    
    def __init__(
        self,
        coordinator_host: str,
        coordinator_port: int,
        worker_id: str,
        heartbeat_interval: float = 5.0,
        timeout: float = 30.0
    ):
        """
        Initialize the coordinator client.
        
        Args:
            coordinator_host: Host address of the coordinator
            coordinator_port: Port the coordinator is listening on
            worker_id: ID of this worker
            heartbeat_interval: Time between heartbeats in seconds
            timeout: Connection timeout in seconds
        """
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.worker_id = worker_id
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        
        # Communication socket
        self.socket = None
        
        # Message queues
        self.send_queue = queue.Queue()
        self.receive_queue = queue.Queue()
        
        # Callback registry for message types
        self.callbacks = {}
        
        # Task management
        self.pending_tasks = {}  # task_id -> task_data
        self.completed_tasks = {}  # task_id -> result
        
        # Threading control
        self.running = False
        self.threads = []
    
    def connect(self) -> bool:
        """
        Connect to the coordinator.
        
        Returns:
            True if connection succeeded, False otherwise
        """
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            
            # Connect to coordinator
            logger.info(f"Connecting to coordinator at {self.coordinator_host}:{self.coordinator_port}")
            self.socket.connect((self.coordinator_host, self.coordinator_port))
            
            # Send registration message
            registration_msg = {
                "type": "register",
                "worker_id": self.worker_id,
                "timestamp": time.time()
            }
            
            if not send_message(self.socket, registration_msg):
                logger.error("Failed to send registration message")
                self.socket.close()
                self.socket = None
                return False
            
            # Wait for acknowledgment
            response = receive_message(self.socket)
            if not response or response.get("type") != "register_ack":
                logger.error(f"Registration failed: {response}")
                self.socket.close()
                self.socket = None
                return False
            
            logger.info(f"Successfully connected to coordinator as {self.worker_id}")
            
            # Start communication threads
            self.running = True
            self._start_threads()
            
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to coordinator: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    def disconnect(self):
        """Disconnect from the coordinator."""
        if not self.running:
            return
        
        logger.info("Disconnecting from coordinator")
        self.running = False
        
        # Stop threads
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        logger.info("Disconnected from coordinator")
    
    def _start_threads(self):
        """Start communication threads."""
        # Receiver thread
        receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        receiver_thread.start()
        self.threads.append(receiver_thread)
        
        # Sender thread
        sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        sender_thread.start()
        self.threads.append(sender_thread)
        
        # Heartbeat thread
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        self.threads.append(heartbeat_thread)
        
        # Message handler thread
        handler_thread = threading.Thread(target=self._message_handler_loop, daemon=True)
        handler_thread.start()
        self.threads.append(handler_thread)
    
    def _receiver_loop(self):
        """Continuously receive messages from the coordinator."""
        while self.running and self.socket:
            try:
                message = receive_message(self.socket)
                if not message:
                    logger.error("Connection to coordinator lost")
                    self.running = False
                    break
                
                # Add message to receive queue
                self.receive_queue.put(message)
            
            except socket.timeout:
                # This is expected, just continue
                continue
            
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                self.running = False
                break
    
    def _sender_loop(self):
        """Continuously send messages to the coordinator."""
        while self.running and self.socket:
            try:
                # Get message from send queue
                try:
                    message = self.send_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Send message
                if not send_message(self.socket, message):
                    logger.error("Failed to send message")
                    self.running = False
                    break
                
                # Mark as done
                self.send_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                self.running = False
                break
    
    def _heartbeat_loop(self):
        """Continuously send heartbeats to the coordinator."""
        while self.running and self.socket:
            try:
                # Create heartbeat message
                heartbeat_msg = {
                    "type": "heartbeat",
                    "worker_id": self.worker_id,
                    "timestamp": time.time()
                }
                
                # Add to send queue
                self.send_queue.put(heartbeat_msg)
                
                # Wait for next heartbeat
                time.sleep(self.heartbeat_interval)
            
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(1)  # Avoid tight loop on persistent errors
    
    def _message_handler_loop(self):
        """Handle received messages."""
        while self.running:
            try:
                # Get message from receive queue
                try:
                    message = self.receive_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process message
                message_type = message.get("type")
                
                # Call registered callback for this message type
                if message_type in self.callbacks:
                    try:
                        self.callbacks[message_type](message)
                    except Exception as e:
                        logger.error(f"Error in callback for message type {message_type}: {e}")
                
                # Handle built-in message types
                if message_type == "heartbeat_ack":
                    # Heartbeat acknowledgment, nothing to do
                    pass
                
                elif message_type == "task_assignment":
                    # New task assignment
                    task_id = message.get("task_id")
                    if task_id:
                        self.pending_tasks[task_id] = message
                        logger.info(f"Received task assignment: {task_id}")
                
                elif message_type == "shutdown":
                    # Shutdown command
                    logger.info("Received shutdown command from coordinator")
                    self.running = False
                    break
                
                # Mark as processed
                self.receive_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in message handler loop: {e}")
                time.sleep(1)  # Avoid tight loop on persistent errors
    
    def register_callback(self, message_type: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback for a specific message type.
        
        Args:
            message_type: Type of message to register callback for
            callback: Function to call when a message of this type is received
        """
        self.callbacks[message_type] = callback
    
    def send_message(self, message):
        """
        Send a message to the coordinator.
        
        Args:
            message: Message to send
        """
        self.send_queue.put(message)
    
    def get_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the next pending task, if any.
        
        Returns:
            Task data dictionary, or None if no tasks are pending
        """
        if not self.pending_tasks:
            return None
        
        # Get the first pending task
        task_id, task_data = next(iter(self.pending_tasks.items()))
        
        # Remove from pending tasks
        del self.pending_tasks[task_id]
        
        return task_data
    
    def send_task_result(self, task_id: str, result: Any, status: str = "completed", metrics: Optional[Dict[str, Any]] = None):
        """
        Send a task result to the coordinator.
        
        Args:
            task_id: ID of the task
            result: Result data
            status: Task status (completed, failed)
            metrics: Optional performance metrics
        """
        # Create result message
        result_msg = {
            "type": "task_result",
            "task_id": task_id,
            "status": status,
            "result": result,
            "metrics": metrics or {},
            "timestamp": time.time()
        }
        
        # Add to completed tasks
        self.completed_tasks[task_id] = result
        
        # Send to coordinator
        self.send_message(result_msg)
        
        logger.info(f"Sent result for task {task_id}")
    
    def send_status_update(self, status: str, metrics: Optional[Dict[str, Any]] = None):
        """
        Send a status update to the coordinator.
        
        Args:
            status: Worker status (idle, busy)
            metrics: Optional performance metrics
        """
        # Create status message
        status_msg = {
            "type": "status_update",
            "worker_id": self.worker_id,
            "status": status,
            "metrics": metrics or {},
            "timestamp": time.time()
        }
        
        # Send to coordinator
        self.send_message(status_msg)
    
    def send_error(self, error_type: str, error_message: str, task_id: Optional[str] = None):
        """
        Send an error to the coordinator.
        
        Args:
            error_type: Type of error
            error_message: Error message
            task_id: Optional associated task ID
        """
        # Create error message
        error_msg = {
            "type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "task_id": task_id,
            "timestamp": time.time()
        }
        
        # Send to coordinator
        self.send_message(error_msg)
        
        logger.error(f"Sent error to coordinator: {error_type} - {error_message}")
    
    def submit_tasks(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """
        Submit tasks to the coordinator.
        
        Args:
            tasks: List of task dictionaries
        
        Returns:
            List of task IDs assigned by the coordinator
        """
        # Create task submission message
        submission_msg = {
            "type": "task_submission",
            "tasks": tasks,
            "timestamp": time.time()
        }
        
        # Send to coordinator
        self.send_message(submission_msg)
        
        # Wait for task IDs
        # Note: In a real implementation, we would use a response mechanism
        # For this mock implementation, we'll just generate task IDs
        task_ids = [f"task_{int(time.time())}_{i}" for i in range(len(tasks))]
        
        return task_ids
    
    def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Wait for tasks to complete.
        
        Args:
            task_ids: List of task IDs to wait for
            timeout: Optional timeout in seconds
        
        Returns:
            List of task results
        """
        # In a real implementation, we would wait for results from the coordinator
        # For this mock implementation, we'll just return dummy results
        results = []
        for task_id in task_ids:
            results.append({
                "task_id": task_id,
                "status": "completed",
                "result": {"output_ids": [1, 2, 3, 4, 5]},
                "sequence_index": int(task_id.split("_")[-1])
            })
        
        return results
    
    @staticmethod
    def discover_coordinator(broadcast_port: int = 5557, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Discover a coordinator node on the local network.
        
        Args:
            broadcast_port: Port to listen for broadcasts on
            timeout: Time to listen for broadcasts in seconds
        
        Returns:
            Coordinator information, or None if no coordinator found
        """
        nodes = discover_nodes(broadcast_port, timeout)
        
        if nodes:
            # Return the first discovered coordinator
            return nodes[0]
        
        return None


class TaskProcessor:
    """
    Processes tasks assigned by the coordinator.
    
    This class works in conjunction with CoordinatorClient to execute
    assigned tasks and report results back to the coordinator.
    """
    
    def __init__(self, coordinator_client: CoordinatorClient, compute_engine=None):
        """
        Initialize the task processor.
        
        Args:
            coordinator_client: Client for communicating with coordinator
            compute_engine: Optional compute engine for executing tasks
        """
        self.coordinator_client = coordinator_client
        self.compute_engine = compute_engine
        
        # Register callbacks
        self.coordinator_client.register_callback("task_assignment", self._handle_task_assignment)
        
        # Task processing thread
        self.running = False
        self.process_thread = None
    
    def start(self):
        """Start the task processor."""
        if self.running:
            return
        
        self.running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        logger.info("Task processor started")
    
    def stop(self):
        """Stop the task processor."""
        if not self.running:
            return
        
        self.running = False
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=5.0)
        
        logger.info("Task processor stopped")
    
    def _handle_task_assignment(self, message):
        """
        Handle a task assignment message.
        
        Args:
            message: Task assignment message
        """
        # Task is automatically queued by the coordinator client
        # Nothing to do here
        pass
    
    def _process_loop(self):
        """Process tasks in a loop."""
        while self.running:
            try:
                # Get next task
                task_data = self.coordinator_client.get_task()
                if not task_data:
                    # No tasks available, wait a bit
                    time.sleep(0.1)
                    continue
                
                # Extract task information
                task_id = task_data.get("task_id")
                task_type = task_data.get("task_type")
                parameters = task_data.get("parameters", {})
                
                if not task_id or not task_type:
                    logger.error(f"Invalid task data: {task_data}")
                    continue
                
                # Update status to busy
                self.coordinator_client.send_status_update("busy")
                
                # Process the task
                try:
                    logger.info(f"Processing task {task_id} of type {task_type}")
                    start_time = time.time()
                    
                    # Execute the task
                    result = self._execute_task(task_type, parameters)
                    
                    # Calculate metrics
                    processing_time = time.time() - start_time
                    metrics = {
                        "execution_time": processing_time,
                        "memory_used_mb": 0,  # Would be populated in a real implementation
                        "cpu_percent": 0,     # Would be populated in a real implementation
                    }
                    
                    # Send result back to coordinator
                    self.coordinator_client.send_task_result(
                        task_id=task_id,
                        result=result,
                        status="completed",
                        metrics=metrics
                    )
                    
                    logger.info(f"Completed task {task_id} in {processing_time:.2f}s")
                
                except Exception as e:
                    logger.error(f"Error processing task {task_id}: {e}")
                    
                    # Send error to coordinator
                    self.coordinator_client.send_error(
                        error_type="task_execution_error",
                        error_message=str(e),
                        task_id=task_id
                    )
                    
                    # Mark task as failed
                    self.coordinator_client.send_task_result(
                        task_id=task_id,
                        result=None,
                        status="failed"
                    )
                
                finally:
                    # Update status to idle
                    self.coordinator_client.send_status_update("idle")
            
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                time.sleep(1)  # Avoid tight loop on persistent errors
    
    def _execute_task(self, task_type: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute a task based on its type and parameters.
        
        Args:
            task_type: Type of task to execute
            parameters: Task parameters
        
        Returns:
            Task result
        """
        if not self.compute_engine:
            raise RuntimeError("No compute engine available for task execution")
        
        if task_type == "layer_computation":
            # Execute a single layer computation
            return self.compute_engine._execute_layer_computation(task_type, parameters)
        
        elif task_type == "token_generation":
            # Generate tokens
            return self.compute_engine._execute_token_generation(task_type, parameters)
        
        elif task_type == "embedding":
            # Generate embeddings
            return self.compute_engine._execute_embedding(task_type, parameters)
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")