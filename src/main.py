#!/usr/bin/env python3
"""
DistributedLLM - Main Entry Point

This script initializes the distributed LLM system, either as a coordinator node
or as a worker node, based on the provided configuration and command line arguments.
"""

import argparse
import logging
import os
import sys
import yaml
import signal
import time
from pathlib import Path

# Add src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import local modules
from src.coordinator.scheduler import Scheduler
from src.worker.compute_engine import ComputeEngine
from src.utils.networking import discover_nodes, register_node


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'{project_root}/logs/distributed_llm.log')
    ]
)
logger = logging.getLogger("DistributedLLM")


def load_config(config_path):
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        sys.exit(1)


def start_coordinator(config):
    """Initialize and start the coordinator node."""
    logger.info("Starting DistributedLLM coordinator node...")
    
    # Initialize the scheduler
    scheduler = Scheduler(
        config["coordinator"],
        config["workers"],
        config["network"],
        config["discovery"]
    )
    
    # Start node discovery if enabled
    if config["discovery"]["enabled"]:
        logger.info("Starting node auto-discovery...")
        scheduler.start_discovery()
    
    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, gracefully stopping coordinator...")
        scheduler.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the scheduler
    scheduler.start()
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down coordinator...")
        scheduler.shutdown()


def start_worker(config, worker_id=None):
    """Initialize and start a worker node."""
    logger.info(f"Starting DistributedLLM worker node{f' {worker_id}' if worker_id else ''}...")
    
    # Identify this worker in the config
    worker_config = None
    if worker_id:
        for worker in config["workers"]:
            if worker["id"] == worker_id:
                worker_config = worker
                break
    
    if not worker_config and config["discovery"]["auto_register"]:
        # Auto-register this worker with the coordinator
        logger.info("Worker not found in config, attempting auto-registration...")
        worker_config = register_node(config["coordinator"], config["network"])
    
    if not worker_config:
        logger.error("Could not find or register worker configuration. Exiting.")
        sys.exit(1)
    
    # Initialize compute engine
    compute_engine = ComputeEngine(
        worker_config,
        config["coordinator"],
        config["network"]
    )
    
    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, gracefully stopping worker...")
        compute_engine.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Connect to coordinator and start processing
    compute_engine.connect()
    compute_engine.start()
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down worker...")
        compute_engine.shutdown()


def main():
    """Parse command line arguments and start the appropriate node type."""
    parser = argparse.ArgumentParser(description="DistributedLLM - Run large language models across heterogeneous hardware.")
    parser.add_argument("--mode", choices=["coordinator", "worker"], required=True,
                        help="Run as coordinator or worker node")
    parser.add_argument("--config-dir", default=f"{project_root}/config",
                        help="Directory containing configuration files")
    parser.add_argument("--worker-id", help="ID of this worker (if running in worker mode)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Set logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create logs directory if it doesn't exist
    os.makedirs(f"{project_root}/logs", exist_ok=True)
    
    # Load configurations
    cluster_config = load_config(f"{args.config_dir}/cluster_config.yaml")
    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    workload_config = load_config(f"{args.config_dir}/workload_config.yaml")
    
    # Combine configurations
    config = {
        **cluster_config,
        "model": model_config["model"],
        "deployment": model_config["deployment"],
        "load_balancing": workload_config["load_balancing"],
        "worker_allocation": workload_config["worker_allocation"],
        "task_scheduling": workload_config["task_scheduling"],
        "fault_tolerance": workload_config["fault_tolerance"]
    }
    
    # Start the appropriate node type
    if args.mode == "coordinator":
        start_coordinator(config)
    else:  # worker mode
        start_worker(config, args.worker_id)


if __name__ == "__main__":
    main()