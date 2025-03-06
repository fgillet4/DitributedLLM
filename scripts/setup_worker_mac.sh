#!/bin/bash
# Setup script for DistributedLLM worker node on macOS

# Ensure we're in the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo "Setting up DistributedLLM worker node for macOS..."
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

# Check for Apple Silicon vs Intel
echo "Checking Mac hardware..."
if [[ $(uname -m) == 'arm64' ]]; then
    echo "Apple Silicon (M1/M2) detected."
    
    # Install PyTorch with MPS support
    echo "Installing PyTorch with MPS support..."
    pip install torch torchvision
    
    # Test MPS availability
    if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        echo "MPS acceleration support detected."
        HAS_MPS=true
    else
        echo "MPS support not available."
        HAS_MPS=false
    fi
else
    echo "Intel Mac detected."
    
    # Install regular PyTorch
    pip install torch torchvision
    
    # Check for GPU (unlikely on Intel Mac but check anyway)
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        echo "GPU support detected."
        HAS_GPU=true
    else
        echo "No GPU support detected. Performance will be CPU-bound."
        HAS_GPU=false
    fi
fi

# Check network connectivity
echo "Checking network connectivity..."
ipconfig getifaddr en0 || echo "Could not determine en0 IP address."
ifconfig en0 | grep "inet " || echo "Could not find en0 interface."

# Create a default worker config if not exists
if [ ! -f "config/worker_config.yaml" ]; then
    echo "Creating default worker configuration..."
    
    # Get local IP address
    local_ip=$(ipconfig getifaddr en0)
    if [ -z "$local_ip" ]; then
        local_ip=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -n 1 | awk '{print $2}')
    fi
    
    # Create config directory if it doesn't exist
    mkdir -p config
    
    # Generate a unique worker ID
    worker_id="worker_mac_$(hostname | tr -d '.')_$(date +%s)"
    
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
EOL

    echo "Default worker configuration created. Please edit config/worker_config.yaml to set the correct coordinator IP."
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