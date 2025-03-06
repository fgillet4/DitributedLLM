#!/bin/bash
# Setup script for DistributedLLM worker node on Linux

# Ensure we're in the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo "Setting up DistributedLLM worker node for Linux..."
echo "Project root: $PROJECT_ROOT"

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p models/cache
mkdir -p data

# Check Python version (require 3.9+)
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d ' ' -f 2)
python_major=$(echo "$python_version" | cut -d '.' -f 1)
python_minor=$(echo "$python_version" | cut -d '.' -f 2)

if [ "$python_major" -lt 3 ] || [ "$python_major" -eq 3 -a "$python_minor" -lt 9 ]; then
    echo "Python 3.9+ is required. Found Python $python_version"
    echo "Please install Python 3.9 or newer."
    exit 1
fi

echo "Found Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for GPU support
echo "Checking for GPU support..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "GPU support detected."
    HAS_GPU=true
    
    # Get GPU info
    python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    
    # Check if NVIDIA drivers are properly installed
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA drivers found."
        nvidia-smi
    else
        echo "NVIDIA drivers not found. Please install NVIDIA drivers for optimal performance."
    fi
else
    echo "No GPU support detected. Performance may be limited."
    HAS_GPU=false
fi

# Check network connectivity
echo "Checking network connectivity..."
hostname -I 2>/dev/null || echo "Could not determine IP address."
ip -4 addr | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "Could not find network interfaces."

# Create a default worker config if not exists
if [ ! -f "config/worker_config.yaml" ]; then
    echo "Creating default worker configuration..."
    
    # Get local IP address
    local_ip=$(hostname -I | awk '{print $1}')
    
    # Create config directory if it doesn't exist
    mkdir -p config
    
    # Generate a unique worker ID
    worker_id="worker_linux_$(hostname | tr -d '.')_$(date +%s)"
    
    # Create default worker configuration
    cat > config/worker_config.yaml << EOL
# DistributedLLM Worker Configuration

# Worker identification
worker:
  id: "${worker_id}"
  host: "${local_ip}"
  port: 5556

# Coordinator connection
coordinator:
  host: "192.168.1.100"  # CHANGE THIS to your coordinator's IP
  port: 5555

# Resource management
resources:
  max_memory_percent: 80
  max_cpu_percent: 90
  gpu_available: ${HAS_GPU}
EOL

    echo "Default worker configuration created. Please edit config/worker_config.yaml to set the correct coordinator IP."
fi

# Create systemd service file (optional)
if [ -d "/etc/systemd/system" ] && [ "$(id -u)" -eq 0 ]; then
    echo "Creating systemd service file..."
    
    cat > /etc/systemd/system/distributed-llm-worker.service << EOL
[Unit]
Description=DistributedLLM Worker Service
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${PROJECT_ROOT}
ExecStart=${PROJECT_ROOT}/venv/bin/python ${PROJECT_ROOT}/src/main.py --mode worker
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOL

    echo "Systemd service file created. You can start the service with:"
    echo "sudo systemctl enable distributed-llm-worker.service"
    echo "sudo systemctl start distributed-llm-worker.service"
fi

# Display setup completion message
echo "Worker setup complete."
echo ""
echo "Before starting the worker, make sure to:"
echo "1. Update the coordinator IP in config/worker_config.yaml"
echo "2. Ensure the coordinator node is running"
echo ""
echo "To start the worker, run:"
echo "source venv/bin/activate"
echo "python src/main.py --mode worker"
echo ""
echo "If you want to enable auto-discovery, run:"
echo "python src/main.py --mode worker --discover"

# Return to the original directory
cd - > /dev/null