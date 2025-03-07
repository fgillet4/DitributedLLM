# DistributedLLM Chat Application

This web application provides a simple chat interface for interacting with language models running on your DistributedLLM infrastructure. It includes both a web UI and REST APIs.

## Features

- Web-based chat interface
- Real-time communication using WebSockets
- REST API endpoints for programmatic access
- Support for chat history and conversation context
- Built-in load testing capabilities

## Installation

1. Make sure you have the required dependencies:

```bash
pip install fastapi uvicorn aiohttp jinja2 websockets
```

2. Ensure your DistributedLLM coordinator and at least one worker are running:

```bash
# On coordinator machine (e.g., your desktop with 3090)
python ../../src/main.py --mode coordinator

# On worker machine(s) (e.g., your MacBook M2)
python ../../src/main.py --mode worker
```

## Usage

### Starting the Chat Server

```bash
# Basic usage with default settings
python chat_server.py

# Connect to a specific coordinator
python chat_server.py --coordinator-host 192.168.1.100 --coordinator-port 5555

# Specify a different port for the web interface
python chat_server.py --host 0.0.0.0 --port 9000

# Specify the model to use
python chat_server.py --model llama-2-13b
```

### Accessing the Chat Interface

Once the server is running, open a web browser and navigate to:

```
http://localhost:8000
```

Or if you specified a different host/port:

```
http://[HOST]:[PORT]
```

### Using the REST API

The chat server provides REST API endpoints that can be used programmatically:

```bash
# Send a message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"text":"Explain quantum computing in simple terms"}'

# Check server status
curl http://localhost:8000/api/status
```

### Running a Load Test from the API

You can trigger a load test directly from the API:

```bash
# Start a load test with 10 concurrent users
curl "http://localhost:8000/api/load_test?concurrent_users=10&messages_per_user=5"
```

## Configuration

You can customize the chat server by editing the following:

- `templates/index.html`: Modify the web interface
- `static/styles.css`: Add custom CSS styles
- `chat_server.py`: Edit server behavior

## Troubleshooting

### Common Issues

1. **Connection errors**: Ensure your coordinator is running and the host/port are correct
2. **WebSocket errors**: Check if any firewalls are blocking WebSocket connections
3. **Slow responses**: Try reducing the model size or adding more workers

### Logs

Check `chat_server.log` for detailed logs of server operations.

## Advanced Usage

### Customizing the Chat Interface

You can modify the HTML template (`templates/index.html`) to customize the chat interface. The WebSocket communication is handled by JavaScript in the template.

### Setting Up for Production

For production deployments, consider:

1. Adding HTTPS using a reverse proxy like Nginx
2. Implementing authentication and rate limiting
3. Using a process manager like supervisord or systemd
4. Setting up monitoring with Prometheus/Grafana

Example nginx configuration:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```