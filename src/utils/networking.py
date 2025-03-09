"""
Networking utilities for DistributedLLM.

Provides functions for communication between nodes in the distributed system,
supporting different operating systems and network configurations.
"""

import logging
import socket
import json
import struct
import time
import threading
import queue
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Message size constants
MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100 MB default limit
HEADER_SIZE = 8  # Size of message length header in bytes


def send_message(sock: socket.socket, message: Dict[str, Any], compress: bool = False) -> bool:
    """
    Send a message over a socket.
    
    Args:
        sock: Socket to send message over
        message: Dictionary containing the message
        compress: Whether to compress the message (for large payloads)
    
    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    try:
        # Convert message to JSON
        message_json = json.dumps(message)
        message_bytes = message_json.encode('utf-8')
        
        # Compress if requested and message is large
        if compress and len(message_bytes) > 1024:
            import zlib
            message_bytes = zlib.compress(message_bytes)
            # Add compression flag to message
            message_bytes = b'C' + message_bytes
        else:
            # Add no-compression flag
            message_bytes = b'N' + message_bytes
        
        # Ensure we don't exceed max message size
        if len(message_bytes) > MAX_MESSAGE_SIZE:
            logger.error(f"Message size ({len(message_bytes)} bytes) exceeds maximum allowed size ({MAX_MESSAGE_SIZE} bytes)")
            return False
        
        # Send message length as header
        header = struct.pack('>Q', len(message_bytes))
        sock.sendall(header)
        
        # Send the message
        sock.sendall(message_bytes)
        return True
    
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return False


def receive_message(sock: socket.socket, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Receive a message from a socket.
    
    Args:
        sock: Socket to receive message from
        timeout: Timeout in seconds, or None for no timeout
    
    Returns:
        dict: Received message, or None if an error occurred
    """
    try:
        # Set timeout if specified
        if timeout is not None:
            sock.settimeout(timeout)
        
        # Receive the header with message length
        header_data = recv_all(sock, HEADER_SIZE)
        if not header_data or len(header_data) < HEADER_SIZE:
            return None
        
        # Unpack message length
        message_length = struct.unpack('>Q', header_data)[0]
        
        # Ensure message length is reasonable
        if message_length > MAX_MESSAGE_SIZE:
            logger.error(f"Incoming message too large: {message_length} bytes")
            return None
        
        # Receive the message
        message_data = recv_all(sock, message_length)
        if not message_data:
            return None
        
        # Check for compression flag
        compression_flag = message_data[0:1]
        message_data = message_data[1:]
        
        # Decompress if needed
        if compression_flag == b'C':
            import zlib
            message_data = zlib.decompress(message_data)
        
        # Decode JSON
        message_json = message_data.decode('utf-8')
        message = json.loads(message_json)
        
        # Reset timeout
        if timeout is not None:
            sock.settimeout(None)
        
        return message
    
    except socket.timeout:
        logger.debug("Socket timeout while receiving message")
        return None
    except ConnectionResetError:
        logger.error("Connection reset by peer")
        return None
    except Exception as e:
        logger.error(f"Error receiving message: {e}")
        return None


def recv_all(sock: socket.socket, n: int) -> Optional[bytes]:
    """
    Receive exactly n bytes from a socket.
    
    Args:
        sock: Socket to receive from
        n: Number of bytes to receive
    
    Returns:
        bytes: Received data, or None if connection closed or error
    """
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None  # Connection closed
        data.extend(packet)
    return data


def discover_nodes(broadcast_port: int, timeout: float = 5.0) -> List[Dict[str, Any]]:
    """
    Discover coordinator nodes on the local network via UDP broadcast.
    
    Args:
        broadcast_port: Port to listen for broadcasts on
        timeout: Time to listen for broadcasts in seconds
    
    Returns:
        List of discovered coordinator nodes with their information
    """
    discovered_nodes = []
    
    try:
        # Create UDP socket for listening
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to broadcast port
        sock.bind(('', broadcast_port))
        
        # Set timeout
        sock.settimeout(timeout)
        
        logger.info(f"Listening for coordinator broadcasts on port {broadcast_port}")
        
        # Listen for broadcasts
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data, addr = sock.recvfrom(8192)
                
                try:
                    # Parse JSON data
                    message = json.loads(data.decode('utf-8'))
                    
                    if message.get("type") == "coordinator_announcement":
                        # Add source IP
                        message["source_ip"] = addr[0]
                        
                        # Check if already discovered
                        if not any(node.get("coordinator_host") == message.get("coordinator_host") 
                                   for node in discovered_nodes):
                            discovered_nodes.append(message)
                            logger.info(f"Discovered coordinator at {message.get('coordinator_host')}:{message.get('coordinator_port')}")
                
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from {addr}")
            
            except socket.timeout:
                # This is expected, just continue listening
                pass
    
    except Exception as e:
        logger.error(f"Error discovering nodes: {e}")
    
    finally:
        sock.close()
    
    return discovered_nodes


def register_node(coordinator_config: Dict[str, Any], network_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register this worker with the coordinator and get worker configuration.
    
    Args:
        coordinator_config: Configuration for connecting to the coordinator
        network_config: Network configuration
    
    Returns:
        Worker configuration received from coordinator, or empty dict on failure
    """
    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Set timeout
        timeout = network_config.get("timeout_seconds", 30)
        sock.settimeout(timeout)
        
        # Connect to coordinator
        coordinator_host = coordinator_config.get("host")
        coordinator_port = coordinator_config.get("port")
        
        logger.info(f"Connecting to coordinator at {coordinator_host}:{coordinator_port}")
        sock.connect((coordinator_host, coordinator_port))
        
        # Generate a unique worker ID based on hostname
        worker_id = f"worker_{socket.gethostname().replace('.', '_')}_{time.time()}"
        
        # Send registration message
        registration_msg = {
            "type": "register",
            "worker_id": worker_id,
            "capabilities": get_system_capabilities(),
            "auto_generated": True
        }
        
        send_message(sock, registration_msg)
        
        # Wait for response
        response = receive_message(sock, timeout)
        
        if not response or response.get("type") != "register_ack":
            logger.error(f"Registration failed. Response: {response}")
            sock.close()
            return {}
        
        # Extract worker configuration
        worker_config = response.get("worker_config", {})
        worker_config["id"] = response.get("worker_id", worker_id)
        
        logger.info(f"Successfully registered as {worker_config['id']}")
        
        # Close connection
        sock.close()
        
        return worker_config
    
    except Exception as e:
        logger.error(f"Error registering with coordinator: {e}")
        return {}


def get_system_capabilities() -> Dict[str, Any]:
    """
    Get system capabilities for registration.
    
    Returns:
        Dict containing system capabilities
    """
    import platform
    import os
    import psutil
    
    capabilities = {}
    
    # System information
    capabilities["hostname"] = socket.gethostname()
    capabilities["platform"] = platform.platform()
    capabilities["os"] = platform.system().lower()
    capabilities["python_version"] = platform.python_version()
    
    # CPU information
    capabilities["cpu_cores"] = psutil.cpu_count(logical=True)
    capabilities["cpu_physical_cores"] = psutil.cpu_count(logical=False)
    if hasattr(psutil, "cpu_freq") and psutil.cpu_freq():
        capabilities["cpu_frequency_mhz"] = psutil.cpu_freq().max
    
    # Memory information
    mem = psutil.virtual_memory()
    capabilities["ram_gb"] = round(mem.total / (1024**3), 1)
    
    # Disk information
    disk = psutil.disk_usage('/')
    capabilities["disk_total_gb"] = round(disk.total / (1024**3), 1)
    capabilities["disk_free_gb"] = round(disk.free / (1024**3), 1)
    
    # GPU information
    capabilities["gpu"] = "none"  # Default to no GPU
    
    # Check for NVIDIA GPU with torch
    try:
        import torch
        if torch.cuda.is_available():
            capabilities["gpu"] = "nvidia"
            capabilities["gpu_count"] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                capabilities["gpu_name"] = torch.cuda.get_device_name(0)
                # Try to get memory info
                try:
                    if hasattr(torch.cuda, 'get_device_properties'):
                        props = torch.cuda.get_device_properties(0)
                        capabilities["gpu_memory_gb"] = round(props.total_memory / (1024**3), 1)
                except:
                    pass
    except ImportError:
        pass
    
    # Network information
    capabilities["ip_address"] = get_local_ip()
    
    return capabilities


def get_local_ip() -> str:
    """
    Get the local IP address.
    
    Returns:
        Local IP address as string
    """
    try:
        # Create a socket to determine the IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't need to be reachable
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'  # Fallback


def is_port_available(port: int, host: str = '127.0.0.1') -> bool:
    """
    Check if a port is available for use.
    
    Args:
        port: Port number to check
        host: Host to check on
    
    Returns:
        True if port is available, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0  # If result is 0, connection succeeded, port is in use
    except:
        return False  # Error occurred, assume port is unavailable


def find_available_port(start_port: int = 5000, end_port: int = 6000) -> Optional[int]:
    """
    Find an available port in the given range.
    
    Args:
        start_port: Starting port number
        end_port: Ending port number
    
    Returns:
        Available port number, or None if no ports are available
    """
    for port in range(start_port, end_port):
        if is_port_available(port):
            return port
    return None


def create_tcp_server(host: str, port: int) -> Optional[socket.socket]:
    """
    Create a TCP server socket.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    
    Returns:
        Socket object, or None if an error occurred
    """
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        return server_socket
    except Exception as e:
        logger.error(f"Error creating TCP server: {e}")
        return None
    # In src/utils/networking.py
# Add TLS/SSL support to socket communications

def create_ssl_context(cert_file, key_file=None):
    """Create an SSL context for secure communications."""
    import ssl
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    return context

def create_tcp_server_secure(host, port, cert_file, key_file):
    """Create a secure TCP server socket."""
    import ssl
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    
    context = create_ssl_context(cert_file, key_file)
    return context.wrap_socket(server_socket, server_side=True)