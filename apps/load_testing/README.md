# DistributedLLM Load Testing Tools

This directory contains tools for testing the performance of your DistributedLLM system under various load conditions. Use these tools to evaluate throughput, latency, and stability before deploying to production.

## Features

- Simulate multiple concurrent users
- Customizable message patterns and frequencies
- Detailed performance metrics
- Support for WebSocket and REST API testing
- Configurable ramp-up periods to simulate gradual load increase

## Installation

Ensure you have the required dependencies:

```bash
pip install aiohttp asyncio
```

## Usage

### Basic Load Test

To run a basic load test against a running chat server:

```bash
# Simulate 10 users, each sending 5 messages
python load_test.py --server http://localhost:8000 --users 10 --messages 5
```

### Advanced Options

```bash
# Use REST API instead of WebSockets
python load_test.py --server http://localhost:8000 --users 10 --messages 5 --api

# Gradually ramp up load over 30 seconds
python load_test.py --server http://localhost:8000 --users 20 --messages 5 --ramp-up 30

# Save results to a file for further analysis
python load_test.py --server http://localhost:8000 --users 10 --messages 5 --output results.json
```

### Testing Against Remote Servers

You can test against a chat server running on any accessible host:

```bash
# Test against a remote server
python load_test.py --server http://192.168.1.100:8000 --users 10 --messages 5
```

## Examples

### Light Load Test (Development Testing)

```bash
python load_test.py --server http://localhost:8000 --users 5 --messages 3 --ramp-up 5
```

### Medium Load Test (QA Testing)

```bash
python load_test.py --server http://localhost:8000 --users 20 --messages 10 --ramp-up 30
```

### Heavy Load Test (Stress Testing)

```bash
python load_test.py --server http://localhost:8000 --users 50 --messages 20 --ramp-up 60
```

### Endurance Test (Stability Testing)

```bash
python load_test.py --server http://localhost:8000 --users 10 --messages 100 --ramp-up 30
```

## Understanding Results

The load test will output comprehensive statistics including:

- Total requests processed
- Success/failure rates
- Latency statistics (min, max, mean, median, standard deviation)
- Percentile breakdowns (50th, 90th, 95th, 99th)
- Error summaries
- Overall throughput (requests per second)

Example output:

```
===== LOAD TEST RESULTS =====
Total requests: 50
Successful: 48 (96.0%)
Failed: 2 (4.0%)

Latency statistics (seconds):
  Min: 0.52
  Max: 3.78
  Mean: 1.24
  Median: 1.05
  StdDev: 0.73

Percentiles:
  50th percentile: 1.05s
  90th percentile: 2.31s
  95th percentile: 2.87s
  99th percentile: 3.62s

Error summary:
  Connection refused: 2 occurrences

Throughput:
  Requests per second: 4.21
=============================
```

## Analyzing and Visualizing Results

If you save results to a JSON file using the `--output` option, you can perform further analysis or create visualizations:

```python
import json
import matplotlib.pyplot as plt
import pandas as pd

# Load results
with open('results.json', 'r') as f:
    results = json.load(f)

# Convert to pandas DataFrame
df = pd.DataFrame(results)

# Filter successful requests
successful = df[df['success'] == True]

# Plot latency distribution
plt.figure(figsize=(10, 6))
plt.hist(successful['latency'], bins=20)
plt.title('Response Latency Distribution')
plt.xlabel('Latency (seconds)')
plt.ylabel('Count')
plt.savefig('latency_distribution.png')
```

## Troubleshooting

### Common Issues

1. **Connection errors**: Ensure the chat server is running and accessible
2. **WebSocket failures**: Some environments may have restrictions on WebSocket connections
3. **Memory issues**: For very large tests, you may need to increase your system's ulimit

### Debug Mode

For more detailed logging during test execution:

```bash
python -m logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    },
}) && python load_test.py --server http://localhost:8000 --users 5 --messages 3
```