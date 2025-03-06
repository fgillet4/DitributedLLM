#!/bin/bash
# Setup script for DistributedLLM coordinator node

# Ensure we're in the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo "Setting up DistributedLLM coordinator node..."
echo "Project root: $PROJECT_ROOT"

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p models
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
else
    echo "No GPU support detected. Performance may be limited."
    HAS_GPU=false
fi

# Check network connectivity
echo "Checking network connectivity..."
hostname -I 2>/dev/null || echo "Could not determine IP address."

# Create a default config if not exists
if [ ! -f "config/cluster_config.yaml" ]; then
    echo "Creating default configuration..."
    
    # Get local IP address (platform dependent)
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        local_ip=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -n 1 | awk '{print $2}')
    else
        # Linux
        local_ip=$(hostname -I | awk '{print $1}')
    fi
    
    # Create config directory if it doesn't exist
    mkdir -p config
    
    # Create default configuration
    cat > config/cluster_config.yaml << EOL
# DistributedLLM Cluster Configuration

# Coordinator node configuration
coordinator:
  host: "${local_ip}"
  port: 5555
  dashboard_port: 8080
  max_workers: 10

# Network configuration
network:
  timeout_seconds: 30
  retry_attempts: 3
  heartbeat_interval_seconds: 5
  max_message_size_mb: 500
  compression: true
  encryption: true
  protocol: "tcp"

# Auto-discovery settings
discovery:
  enabled: true
  broadcast_port: 5557
  broadcast_interval_seconds: 10
  auto_register: true
EOL

    # Create model config
    cat > config/model_config.yaml << EOL
# LLM Model Configuration

model:
  name: "llama-2-7b"
  family: "llama"
  version: "2.0"
  size_billion_parameters: 7
  
  # File locations
  paths:
    base_path: "./models"
    weights_file: "llama-2-7b.pth"
    tokenizer_file: "tokenizer.model"
  
  # Quantization settings
  quantization:
    enabled: true
    method: "int8"
EOL

    # Create workload config
    cat > config/workload_config.yaml << EOL
# Workload Balancing Configuration

# Load balancing strategy
load_balancing:
  strategy: "dynamic"
  rebalance_interval_seconds: 30
  
  # Thresholds for rebalancing
  thresholds:
    worker_idle_percent_trigger: 15.0
    performance_variance_trigger: 25.0
EOL

    echo "Default configuration created. Please review and edit as needed."
fi

# Display setup completion message
echo "Coordinator setup complete."
echo ""
echo "To start the coordinator, run:"
echo "source venv/bin/activate"
echo "python src/main.py --mode coordinator"
echo ""
echo "Make sure to update the configuration in the config/ directory if needed."

# Return to the original directory
cd - > /dev/null