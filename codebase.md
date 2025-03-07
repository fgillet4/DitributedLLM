# apps/chat/chat_server.py

```py
#!/usr/bin/env python3
"""
Chatbot server for DistributedLLM.

This server provides a web interface and REST API for interacting with
the distributed language model.
"""

import argparse
import logging
import time
import uuid
import os
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, WebSocket, Request, Form, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import from distributed_llm
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent  # Navigate up from apps/chat to root
sys.path.append(str(project_root))

from src.worker.communication import CoordinatorClient
from src.model.inference import DistributedInference
from src.model.tokenizer import Tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chat_server.log')
    ]
)
logger = logging.getLogger("ChatServer")

# Create FastAPI app
app = FastAPI(title="DistributedLLM Chat")

# Create templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)

# Create static directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Load templates
templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Create a simple HTML template if it doesn't exist
index_template = templates_dir / "index.html"
if not index_template.exists():
    with open(index_template, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>DistributedLLM Chat</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        #chat-container {
            height: 400px;
            border: 1px solid #ccc;
            padding: 10px;
            overflow-y: auto;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e6f7ff;
            border-radius: 15px;
            padding: 8px 15px;
            margin: 5px 0;
            max-width: 70%;
            margin-left: auto;
            clear: both;
            float: right;
        }
        .assistant-message {
            background-color: #f1f1f1;
            border-radius: 15px;
            padding: 8px 15px;
            margin: 5px 0;
            max-width: 70%;
            clear: both;
            float: left;
        }
        #message-form {
            display: flex;
        }
        #message-input {
            flex-grow: 1;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        #send-button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            margin-left: 10px;
            border-radius: 5px;
            cursor: pointer;
        }
        #send-button:hover {
            background-color: #45a049;
        }
        #system-info {
            margin-top: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
            font-size: 12px;
        }
        .loader {
            border: 4px solid #f3f3f3;
            border-radius: 50%;
            border-top: 4px solid #3498db;
            width: 12px;
            height: 12px;
            animation: spin 2s linear infinite;
            display: inline-block;
            margin-left: 10px;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <h1>DistributedLLM Chat</h1>
    <div id="chat-container"></div>
    <form id="message-form" onsubmit="sendMessage(event)">
        <input type="text" id="message-input" placeholder="Type your message...">
        <button type="submit" id="send-button">Send <span id="loader" class="loader"></span></button>
    </form>
    <div id="system-info">
        <p><strong>System Info:</strong> {{ system_info }}</p>
    </div>

    <script>
        let ws = new WebSocket(`ws://${window.location.host}/ws/chat`);
        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('message-input');
        const loader = document.getElementById('loader');
        
        ws.onopen = function(event) {
            console.log("Connection established");
        };
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            if (message.type === 'assistant') {
                const msgDiv = document.createElement('div');
                msgDiv.className = 'assistant-message';
                msgDiv.textContent = message.text;
                chatContainer.appendChild(msgDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                loader.style.display = 'none';
            }
        };
        
        ws.onclose = function(event) {
            console.log("Connection closed");
        };
        
        function sendMessage(event) {
            event.preventDefault();
            const message = messageInput.value;
            if (message.trim()) {
                // Display user message
                const userMsgDiv = document.createElement('div');
                userMsgDiv.className = 'user-message';
                userMsgDiv.textContent = message;
                chatContainer.appendChild(userMsgDiv);
                
                // Send to server
                ws.send(JSON.stringify({text: message}));
                
                // Clear input and show loader
                messageInput.value = '';
                loader.style.display = 'inline-block';
                
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
    </script>
</body>
</html>
""")

# Create CSS file
css_file = static_dir / "styles.css"
if not css_file.exists():
    with open(css_file, "w") as f:
        f.write("""
/* Add your custom styles here */
""")

# Models for API requests
class ChatMessage(BaseModel):
    text: str


# Global variables
coordinator_client = None
inference_engine = None


# Initialize the connection to the coordinator
def initialize_client(coordinator_host, coordinator_port, model_id):
    global coordinator_client, inference_engine
    
    try:
        # Create a unique ID for this chat server
        client_id = f"chat_server_{uuid.uuid4().hex[:8]}"
        
        # Connect to coordinator
        coordinator_client = CoordinatorClient(
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port,
            worker_id=client_id,
            heartbeat_interval=10.0
        )
        
        if not coordinator_client.connect():
            logger.error("Failed to connect to coordinator")
            return False
        
        # Initialize inference engine
        inference_engine = DistributedInference(
            coordinator_client=coordinator_client,
            model_id=model_id
        )
        
        logger.info(f"Successfully connected to coordinator at {coordinator_host}:{coordinator_port}")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing client: {e}")
        return False


# Chat history for each session
chat_sessions = {}


# WebSocket endpoint for chat
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create a session ID for this connection
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = []
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = eval(data)  # Simple conversion from JSON string
            user_text = message.get("text", "")
            
            # Add to chat history
            chat_sessions[session_id].append({"role": "user", "content": user_text})
            
            # Process with the model
            if inference_engine:
                try:
                    # Build prompt from chat history
                    prompt = ""
                    for msg in chat_sessions[session_id]:
                        if msg["role"] == "user":
                            prompt += f"User: {msg['content']}\n"
                        else:
                            prompt += f"Assistant: {msg['content']}\n"
                    prompt += "Assistant: "
                    
                    # Generate response
                    start_time = time.time()
                    response = inference_engine.generate(
                        prompt=prompt,
                        max_new_tokens=150,
                        temperature=0.7,
                    )
                    end_time = time.time()
                    
                    # Extract just the assistant's response
                    assistant_response = response.split("Assistant: ")[-1].strip()
                    
                    # Add to chat history
                    chat_sessions[session_id].append({"role": "assistant", "content": assistant_response})
                    
                    # Send response to client
                    await websocket.send_json({
                        "type": "assistant",
                        "text": assistant_response,
                        "time_taken": round(end_time - start_time, 2)
                    })
                    
                    logger.info(f"Generated response in {end_time - start_time:.2f}s")
                
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    await websocket.send_json({
                        "type": "assistant",
                        "text": f"I'm sorry, I encountered an error: {str(e)}"
                    })
            else:
                await websocket.send_json({
                    "type": "assistant",
                    "text": "The model is not initialized. Please check the server logs."
                })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        # Clean up session
        if session_id in chat_sessions:
            del chat_sessions[session_id]


# Web interface route
@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    system_info = "Not connected to coordinator"
    
    if coordinator_client and inference_engine:
        system_info = f"Connected to coordinator. Model: {inference_engine.model_id}"
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "system_info": system_info}
    )


# API route for chat
@app.post("/api/chat")
async def chat(message: ChatMessage):
    if not inference_engine:
        return {"error": "Model not initialized"}
    
    try:
        # Generate response
        start_time = time.time()
        response = inference_engine.generate(
            prompt=message.text,
            max_new_tokens=150,
            temperature=0.7,
        )
        end_time = time.time()
        
        return {
            "response": response,
            "time_taken": round(end_time - start_time, 2)
        }
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {"error": str(e)}


# Load test route
@app.get("/api/load_test")
async def load_test(
    concurrent_users: int = 10,
    messages_per_user: int = 5,
    background_tasks: BackgroundTasks = None
):
    """Start a load test with simulated users."""
    if not inference_engine:
        return {"error": "Model not initialized"}
    
    background_tasks.add_task(
        run_load_test,
        concurrent_users=concurrent_users,
        messages_per_user=messages_per_user
    )
    
    return {
        "status": "Load test started",
        "concurrent_users": concurrent_users,
        "messages_per_user": messages_per_user
    }


async def run_load_test(concurrent_users: int, messages_per_user: int):
    """Run a load test by simulating multiple concurrent users."""
    import asyncio
    import random
    
    logger.info(f"Starting load test with {concurrent_users} users, {messages_per_user} messages each")
    
    # Sample prompts for testing
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short poem about technology.",
        "What are some good books to read about artificial intelligence?",
        "How does distributed computing work?",
        "Describe the process of making chocolate chip cookies.",
        "What is the theory of relativity?",
        "Tell me about the history of the internet.",
        "How do I learn Python programming?",
        "What are the benefits of exercise?",
        "Explain how neural networks work."
    ]
    
    # Track metrics
    total_latency = 0
    total_requests = 0
    failed_requests = 0
    
    start_time = time.time()
    
    # Create tasks for each user
    tasks = []
    for user_id in range(concurrent_users):
        for msg_id in range(messages_per_user):
            # Random delay to simulate real user behavior
            await asyncio.sleep(random.uniform(0.1, 2.0))
            
            # Select a random prompt
            prompt = random.choice(test_prompts)
            
            # Request processing
            try:
                req_start = time.time()
                
                response = inference_engine.generate(
                    prompt=prompt,
                    max_new_tokens=50,  # Shorter for testing
                    temperature=0.7,
                )
                
                req_latency = time.time() - req_start
                total_latency += req_latency
                total_requests += 1
                
                logger.debug(f"User {user_id}, Request {msg_id}: Latency {req_latency:.2f}s")
            
            except Exception as e:
                logger.error(f"Error in load test: {e}")
                failed_requests += 1
    
    total_time = time.time() - start_time
    
    # Log results
    logger.info(f"Load test completed in {total_time:.2f}s")
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Failed requests: {failed_requests}")
    
    if total_requests > 0:
        logger.info(f"Average latency: {total_latency / total_requests:.2f}s")
        logger.info(f"Throughput: {total_requests / total_time:.2f} requests/second")


# Status check endpoint
@app.get("/api/status")
async def get_status():
    if coordinator_client and inference_engine:
        # Get statistics from inference engine
        stats = inference_engine.get_stats()
        
        return {
            "status": "connected",
            "model_id": inference_engine.model_id,
            "stats": stats
        }
    else:
        return {
            "status": "disconnected"
        }


# Main entry point
def main():
    parser = argparse.ArgumentParser(description="DistributedLLM Chat Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--coordinator-host", default="localhost", help="Coordinator host")
    parser.add_argument("--coordinator-port", type=int, default=5555, help="Coordinator port")
    parser.add_argument("--model", default="llama-2-7b", help="Model ID to use")
    
    args = parser.parse_args()
    
    # Initialize connection to coordinator
    if not initialize_client(args.coordinator_host, args.coordinator_port, args.model):
        logger.warning("Failed to initialize connection to coordinator. Chat will not work.")
    
    # Start server
    uvicorn.run(
        "chat_server:app",
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
```

# apps/chat/README.md

```md
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

\`\`\`bash
pip install fastapi uvicorn aiohttp jinja2 websockets
\`\`\`

2. Ensure your DistributedLLM coordinator and at least one worker are running:

\`\`\`bash
# On coordinator machine (e.g., your desktop with 3090)
python ../../src/main.py --mode coordinator

# On worker machine(s) (e.g., your MacBook M2)
python ../../src/main.py --mode worker
\`\`\`

## Usage

### Starting the Chat Server

\`\`\`bash
# Basic usage with default settings
python chat_server.py

# Connect to a specific coordinator
python chat_server.py --coordinator-host 192.168.1.100 --coordinator-port 5555

# Specify a different port for the web interface
python chat_server.py --host 0.0.0.0 --port 9000

# Specify the model to use
python chat_server.py --model llama-2-13b
\`\`\`

### Accessing the Chat Interface

Once the server is running, open a web browser and navigate to:

\`\`\`
http://localhost:8000
\`\`\`

Or if you specified a different host/port:

\`\`\`
http://[HOST]:[PORT]
\`\`\`

### Using the REST API

The chat server provides REST API endpoints that can be used programmatically:

\`\`\`bash
# Send a message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"text":"Explain quantum computing in simple terms"}'

# Check server status
curl http://localhost:8000/api/status
\`\`\`

### Running a Load Test from the API

You can trigger a load test directly from the API:

\`\`\`bash
# Start a load test with 10 concurrent users
curl "http://localhost:8000/api/load_test?concurrent_users=10&messages_per_user=5"
\`\`\`

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

\`\`\`nginx
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
\`\`\`
```

# apps/load_testing/load_test.py

```py
#!/usr/bin/env python3
"""
Load testing script for DistributedLLM chat server.

This script simulates multiple concurrent users sending requests to the
chat server to evaluate performance under load.
"""

import argparse
import asyncio
import time
import random
import logging
import aiohttp
import statistics
from typing import List, Dict, Any
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('load_test.log')
    ]
)
logger = logging.getLogger("LoadTest")


class ChatUser:
    """Simulates a single chat user."""
    
    def __init__(self, user_id: int, server_url: str, ws_url: str, messages: List[str]):
        self.user_id = user_id
        self.server_url = server_url
        self.ws_url = ws_url
        self.messages = messages
        self.results = []
    
    async def run_session(self, use_websocket: bool = True):
        """Run a chat session for this user."""
        if use_websocket:
            await self.run_websocket_session()
        else:
            await self.run_api_session()
    
    async def run_websocket_session(self):
        """Run a chat session using WebSockets."""
        try:
            session = aiohttp.ClientSession()
            async with session.ws_connect(self.ws_url) as ws:
                logger.debug(f"User {self.user_id}: WebSocket connected")
                
                for i, message in enumerate(self.messages):
                    # Send message
                    start_time = time.time()
                    await ws.send_json({"text": message})
                    
                    # Wait for response
                    response = await ws.receive_json()
                    latency = time.time() - start_time
                    
                    # Record result
                    self.results.append({
                        "user_id": self.user_id,
                        "message_id": i,
                        "latency": latency,
                        "success": True,
                        "error": None
                    })
                    
                    logger.debug(f"User {self.user_id}, Message {i}: Latency {latency:.2f}s")
                    
                    # Simulate thinking time
                    await asyncio.sleep(random.uniform(0.5, 3.0))
            
            await session.close()
        
        except Exception as e:
            logger.error(f"User {self.user_id}: WebSocket error: {e}")
            self.results.append({
                "user_id": self.user_id,
                "message_id": len(self.results),
                "latency": 0,
                "success": False,
                "error": str(e)
            })
    
    async def run_api_session(self):
        """Run a chat session using the REST API."""
        async with aiohttp.ClientSession() as session:
            for i, message in enumerate(self.messages):
                try:
                    # Send message
                    start_time = time.time()
                    async with session.post(
                        f"{self.server_url}/api/chat",
                        json={"text": message}
                    ) as response:
                        result = await response.json()
                        latency = time.time() - start_time
                        
                        # Record result
                        self.results.append({
                            "user_id": self.user_id,
                            "message_id": i,
                            "latency": latency,
                            "success": response.status == 200,
                            "error": None if response.status == 200 else result.get("error")
                        })
                        
                        logger.debug(f"User {self.user_id}, Message {i}: Latency {latency:.2f}s")
                
                except Exception as e:
                    logger.error(f"User {self.user_id}, Message {i}: Error: {e}")
                    self.results.append({
                        "user_id": self.user_id,
                        "message_id": i,
                        "latency": 0,
                        "success": False,
                        "error": str(e)
                    })
                
                # Simulate thinking time
                await asyncio.sleep(random.uniform(0.5, 3.0))


async def run_load_test(
    server_url: str,
    num_users: int,
    messages_per_user: int,
    ramp_up_time: float = 10.0,
    use_websocket: bool = True
):
    """Run a load test with multiple simulated users."""
    # Sample prompts for testing
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short poem about technology.",
        "What are some good books to read about artificial intelligence?",
        "How does distributed computing work?",
        "Describe the process of making chocolate chip cookies.",
        "What is the theory of relativity?",
        "Tell me about the history of the internet.",
        "How do I learn Python programming?",
        "What are the benefits of exercise?",
        "Explain how neural networks work."
    ]
    
    # Create users
    users = []
    for i in range(num_users):
        # Select random messages for this user
        messages = random.sample(test_prompts, min(messages_per_user, len(test_prompts)))
        
        # Create WebSocket URL
        ws_url = f"ws://{server_url.split('://')[-1]}/ws/chat"
        if server_url.startswith("https"):
            ws_url = f"wss://{server_url.split('://')[-1]}/ws/chat"
        
        users.append(ChatUser(i, server_url, ws_url, messages))
    
    # Start each user with a delay to ramp up
    tasks = []
    for i, user in enumerate(users):
        # Calculate delay for this user
        delay = (i / num_users) * ramp_up_time
        
        # Schedule user task
        tasks.append(
            asyncio.create_task(
                start_user_with_delay(user, delay, use_websocket)
            )
        )
    
    # Wait for all users to complete
    await asyncio.gather(*tasks)
    
    # Collect results
    all_results = []
    for user in users:
        all_results.extend(user.results)
    
    return all_results


async def start_user_with_delay(user, delay, use_websocket):
    """Start a user session after a delay."""
    await asyncio.sleep(delay)
    logger.info(f"Starting user {user.user_id}")
    await user.run_session(use_websocket)


def analyze_results(results):
    """Analyze test results and print statistics."""
    if not results:
        logger.error("No results to analyze")
        return
    
    # Calculate statistics
    latencies = [r["latency"] for r in results if r["success"]]
    successes = sum(1 for r in results if r["success"])
    failures = sum(1 for r in results if not r["success"])
    
    if not latencies:
        logger.error("No successful requests to analyze")
        return
    
    # Print report
    print("\n===== LOAD TEST RESULTS =====")
    print(f"Total requests: {len(results)}")
    print(f"Successful: {successes} ({successes / len(results) * 100:.1f}%)")
    print(f"Failed: {failures} ({failures / len(results) * 100:.1f}%)")
    print("\nLatency statistics (seconds):")
    print(f"  Min: {min(latencies):.2f}")
    print(f"  Max: {max(latencies):.2f}")
    print(f"  Mean: {statistics.mean(latencies):.2f}")
    print(f"  Median: {statistics.median(latencies):.2f}")
    if len(latencies) > 1:
        print(f"  StdDev: {statistics.stdev(latencies):.2f}")
    
    # Calculate percentiles
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.5)]
    p90 = latencies[int(len(latencies) * 0.9)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    
    print("\nPercentiles:")
    print(f"  50th percentile: {p50:.2f}s")
    print(f"  90th percentile: {p90:.2f}s")
    print(f"  95th percentile: {p95:.2f}s")
    print(f"  99th percentile: {p99:.2f}s")
    
    # Print error summary if there are failures
    if failures > 0:
        error_types = {}
        for r in results:
            if not r["success"] and r["error"]:
                error_types[r["error"]] = error_types.get(r["error"], 0) + 1
        
        print("\nError summary:")
        for error, count in error_types.items():
            print(f"  {error}: {count} occurrences")
    
    print("\nThroughput:")
    total_time = max(r["latency"] for r in results)
    print(f"  Requests per second: {len(results) / total_time:.2f}")
    print("=============================\n")


def main():
    parser = argparse.ArgumentParser(description="Load Test for DistributedLLM Chat Server")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--messages", type=int, default=5, help="Messages per user")
    parser.add_argument("--ramp-up", type=float, default=10.0, help="Ramp-up time in seconds")
    parser.add_argument("--api", action="store_true", help="Use REST API instead of WebSockets")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting load test with {args.users} users, {args.messages} messages per user")
    logger.info(f"Server URL: {args.server}")
    
    # Run the load test
    results = asyncio.run(
        run_load_test(
            server_url=args.server,
            num_users=args.users,
            messages_per_user=args.messages,
            ramp_up_time=args.ramp_up,
            use_websocket=not args.api
        )
    )
    
    # Analyze and print results
    analyze_results(results)
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
```

# apps/load_testing/README.md

```md
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

\`\`\`bash
pip install aiohttp asyncio
\`\`\`

## Usage

### Basic Load Test

To run a basic load test against a running chat server:

\`\`\`bash
# Simulate 10 users, each sending 5 messages
python load_test.py --server http://localhost:8000 --users 10 --messages 5
\`\`\`

### Advanced Options

\`\`\`bash
# Use REST API instead of WebSockets
python load_test.py --server http://localhost:8000 --users 10 --messages 5 --api

# Gradually ramp up load over 30 seconds
python load_test.py --server http://localhost:8000 --users 20 --messages 5 --ramp-up 30

# Save results to a file for further analysis
python load_test.py --server http://localhost:8000 --users 10 --messages 5 --output results.json
\`\`\`

### Testing Against Remote Servers

You can test against a chat server running on any accessible host:

\`\`\`bash
# Test against a remote server
python load_test.py --server http://192.168.1.100:8000 --users 10 --messages 5
\`\`\`

## Examples

### Light Load Test (Development Testing)

\`\`\`bash
python load_test.py --server http://localhost:8000 --users 5 --messages 3 --ramp-up 5
\`\`\`

### Medium Load Test (QA Testing)

\`\`\`bash
python load_test.py --server http://localhost:8000 --users 20 --messages 10 --ramp-up 30
\`\`\`

### Heavy Load Test (Stress Testing)

\`\`\`bash
python load_test.py --server http://localhost:8000 --users 50 --messages 20 --ramp-up 60
\`\`\`

### Endurance Test (Stability Testing)

\`\`\`bash
python load_test.py --server http://localhost:8000 --users 10 --messages 100 --ramp-up 30
\`\`\`

## Understanding Results

The load test will output comprehensive statistics including:

- Total requests processed
- Success/failure rates
- Latency statistics (min, max, mean, median, standard deviation)
- Percentile breakdowns (50th, 90th, 95th, 99th)
- Error summaries
- Overall throughput (requests per second)

Example output:

\`\`\`
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
\`\`\`

## Analyzing and Visualizing Results

If you save results to a JSON file using the `--output` option, you can perform further analysis or create visualizations:

\`\`\`python
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
\`\`\`

## Troubleshooting

### Common Issues

1. **Connection errors**: Ensure the chat server is running and accessible
2. **WebSocket failures**: Some environments may have restrictions on WebSocket connections
3. **Memory issues**: For very large tests, you may need to increase your system's ulimit

### Debug Mode

For more detailed logging during test execution:

\`\`\`bash
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
\`\`\`
```

# apps/synthbot/.eslintrc.js

```js

```

# apps/synthbot/.gitignore

```

```

# apps/synthbot/.prettierrc.js

```js

```

# apps/synthbot/config/default.js

```js

```

# apps/synthbot/config/schema.js

```js

```

# apps/synthbot/jest.config.js

```js

```

# apps/synthbot/package.json

```json
{
    "name": "synthbot",
    "version": "0.1.0",
    "description": "Terminal-based coding agent powered by distributed LLMs",
    "main": "src/index.js",
    "bin": {
      "synthbot": "./bin/synthbot.js"
    },
    "scripts": {
      "start": "node src/index.js",
      "test": "jest",
      "lint": "eslint .",
      "format": "prettier --write \"**/*.{js,json}\"",
      "build": "npm run lint && npm run test"
    },
    "keywords": [
      "ai",
      "coding-assistant",
      "terminal",
      "tui",
      "llm",
      "distributed"
    ],
    "author": "",
    "license": "MIT",
    "dependencies": {
      "blessed": "^0.1.81",
      "blessed-contrib": "^4.11.0",
      "chalk": "^4.1.2",
      "commander": "^9.4.1",
      "cosmiconfig": "^8.0.0",
      "diff": "^5.1.0",
      "dotenv": "^16.0.3",
      "fast-glob": "^3.2.12",
      "ignore": "^5.2.4",
      "isomorphic-git": "^1.21.0",
      "keytar": "^7.9.0",
      "node-fetch": "^2.6.9",
      "simple-git": "^3.16.0",
      "tiktoken": "^1.0.3",
      "winston": "^3.8.2",
      "ws": "^8.12.1"
    },
    "devDependencies": {
      "eslint": "^8.34.0",
      "eslint-config-prettier": "^8.6.0",
      "eslint-plugin-node": "^11.1.0",
      "eslint-plugin-prettier": "^4.2.1",
      "jest": "^29.4.3",
      "prettier": "^2.8.4"
    },
    "engines": {
      "node": ">=14.0.0"
    }
  }
```

# apps/synthbot/README.md

```md
# SynthBot

SynthBot is an open-source terminal-based coding agent powered by distributed large language models. It helps developers write, modify, and understand code directly from their terminal, with full project context awareness.

<p align="center">
  <img src="assets/synthbot-logo.png" alt="SynthBot Logo" width="200"/>
</p>

## Features

- **Terminal-Based UI**: Beautiful text-based interface that runs in your terminal.
- **Project-Aware**: Continuously scans your project files to maintain context.
- **Direct Code Modifications**: Makes changes to your files with your approval.
- **Context Management**: Monitors token usage and suggests session refreshes.
- **Git-Aware**: Understands git history and project structure.
- **Distributed LLM**: Leverages your local distributed LLM network for privacy and speed.

## Installation

\`\`\`bash
# Install globally
npm install -g synthbot

# Or install locally
npm install synthbot
\`\`\`

## Usage

Navigate to your project's root directory and run:

\`\`\`bash
synthbot run
\`\`\`

Or if installed locally:

\`\`\`bash
npx synthbot run
\`\`\`

### Command Options

\`\`\`bash
# Start SynthBot with a specific configuration
synthbot run --config path/to/config.js

# Set the coordinator address for the distributed LLM
synthbot run --coordinator 192.168.1.100:5555

# Start with debug logging
synthbot run --verbose

# Show help
synthbot --help
\`\`\`

## Keyboard Shortcuts

Within the TUI:

- `Ctrl+C`: Exit SynthBot
- `Ctrl+S`: Save conversation
- `Ctrl+R`: Refresh project context
- `Ctrl+F`: Find in files
- `Tab`: Switch focus between panels
- `Ctrl+L`: Clear current conversation
- `Ctrl+Z`: Undo last file modification

## Configuration

Create a `.synthbotrc.js` file in your home or project directory:

\`\`\`javascript
module.exports = {
  // LLM connection settings
  llm: {
    coordinatorHost: "localhost",
    coordinatorPort: 5555,
    model: "llama-2-13b",
    temperature: 0.7
  },
  
  // Project context settings
  context: {
    maxTokens: 8192,
    excludeDirs: ["node_modules", "dist", ".git"],
    excludeFiles: ["*.lock", "*.log"],
    priorityFiles: ["README.md", "package.json"]
  },
  
  // UI preferences
  ui: {
    theme: "dark",
    layout: "default",
    fontSize: 14
  }
};
\`\`\`

## Architecture

SynthBot is organized into the following main components:

- **Agent**: Core logic for the LLM interaction and code understanding.
- **API**: Communication with the distributed LLM network.
- **TUI**: Terminal user interface components.
- **Utils**: Helper utilities for filesystem, git, etc.

\`\`\`
synthbot/
├── bin/           # Executable scripts
├── src/
│   ├── agent/     # LLM agent implementation
│   ├── api/       # API for LLM communication
│   ├── tui/       # Terminal UI components
│   └── utils/     # Utility functions
└── config/        # Configuration
\`\`\`

## Integrating with DistributedLLM

SynthBot is designed to work with your local DistributedLLM setup. Make sure your coordinator is running:

\`\`\`bash
# In your DistributedLLM directory
python src/main.py --mode coordinator
\`\`\`

SynthBot will automatically connect to the coordinator and use the distributed LLM network for all inferences.

## Contributing

Contributions welcome! Please check out our [contributing guidelines](CONTRIBUTING.md).

## License

MIT
```

# apps/synthbot/src/agent/agent.js

```js
/**
 * Core Agent Implementation
 * 
 * Manages interactions with the LLM, processes project context,
 * and handles file modifications.
 */

const path = require('path');
const fs = require('fs').promises;
const { logger } = require('../utils');
const { manageContext } = require('./context');
const { countTokens } = require('./tokenizer');
const { 
  readFile, 
  writeFile, 
  listDirectoryFiles,
  checkFileExists
} = require('./fileManager');

/**
 * Create the agent instance
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.apiClient API client for LLM communication
 * @param {Object} options.tokenMonitor Token usage monitor
 * @param {Array} options.projectFiles Initial list of project files
 * @param {string} options.projectRoot Project root directory
 * @returns {Object} The agent interface
 */
function createAgent({ apiClient, tokenMonitor, projectFiles, projectRoot }) {
  // Initialize conversation history
  let conversationHistory = [];
  
  // Initialize file context
  let fileContexts = new Map();
  
  // Initialize the context manager
  const contextManager = manageContext({
    tokenMonitor,
    maxSize: tokenMonitor.getMaxTokens()
  });
  
  /**
   * Load content of a file and add to context
   * 
   * @param {string} filePath Path to the file
   * @returns {Promise<string>} The file content
   */
  async function loadFileContext(filePath) {
    try {
      // Check if file exists
      const exists = await checkFileExists(filePath);
      if (!exists) {
        throw new Error(`File not found: ${filePath}`);
      }
      
      // Load file content
      const content = await readFile(filePath);
      
      // Calculate tokens
      const tokens = countTokens(content);
      
      // Store in context cache
      fileContexts.set(filePath, {
        content,
        tokens,
        lastModified: new Date()
      });
      
      // Add to context manager
      contextManager.addFileContext(filePath, content);
      
      return content;
    } catch (error) {
      logger.error(`Failed to load file context: ${filePath}`, { error });
      throw error;
    }
  }
  
  /**
   * Modify a file in the project
   * 
   * @param {string} filePath Path to the file
   * @param {string} newContent New file content
   * @returns {Promise<boolean>} Success indicator
   */
  async function modifyFile(filePath, newContent) {
    try {
      const fullPath = path.resolve(projectRoot, filePath);
      
      // Check if file exists
      const exists = await checkFileExists(fullPath);
      if (!exists) {
        throw new Error(`File not found: ${fullPath}`);
      }
      
      // Get original content for diff
      const originalContent = await readFile(fullPath);
      
      // Write new content
      await writeFile(fullPath, newContent);
      
      // Update context
      if (fileContexts.has(filePath)) {
        // Update cache
        fileContexts.set(filePath, {
          content: newContent,
          tokens: countTokens(newContent),
          lastModified: new Date()
        });
        
        // Update context manager
        contextManager.updateFileContext(filePath, newContent);
      }
      
      logger.info(`Modified file: ${filePath}`);
      return true;
    } catch (error) {
      logger.error(`Failed to modify file: ${filePath}`, { error });
      return false;
    }
  }
  
  /**
   * Scan for specific files in the project based on patterns
   * 
   * @param {string} pattern Glob pattern to match files
   * @returns {Promise<Array<string>>} Matching file paths
   */
  async function findFiles(pattern) {
    try {
      // Implementation would use something like fast-glob
      return [];
    } catch (error) {
      logger.error(`Failed to find files: ${pattern}`, { error });
      return [];
    }
  }
  
  /**
   * Process a user message and generate a response
   * 
   * @param {string} message User's message
   * @returns {Promise<Object>} Response object
   */
  async function processMessage(message) {
    try {
      // Add user message to conversation history
      conversationHistory.push({ role: 'user', content: message });
      
      // Get current context
      const context = contextManager.getCurrentContext();
      
      // Generate LLM prompt
      const prompt = generatePrompt(message, context);
      
      // Get response from LLM
      logger.debug('Sending prompt to LLM', { messageLength: message.length });
      const response = await apiClient.generateResponse(prompt);
      
      // Add assistant response to conversation history
      conversationHistory.push({ role: 'assistant', content: response.text });
      
      // Update token usage
      tokenMonitor.updateUsage(countTokens(response.text));
      
      // Check for file modifications in the response
      const fileModifications = parseFileModifications(response.text);
      
      // Return the processed response
      return {
        text: response.text,
        fileModifications,
        tokenUsage: tokenMonitor.getCurrentUsage()
      };
    } catch (error) {
      logger.error('Failed to process message', { error });
      throw error;
    }
  }
  
  /**
   * Generate a prompt for the LLM
   * 
   * @param {string} message User message
   * @param {Object} context Current context
   * @returns {string} Generated prompt
   */
  function generatePrompt(message, context) {
    // Basic prompt with system instruction and context
    const systemPrompt = `You are SynthBot, an AI coding assistant that helps developers understand and modify their codebase. 
You can read files and make changes to them directly. Always be concise and focus on providing practical solutions.
When modifying files, use the format:
\`\`\`file:path/to/file.js
// New content here
\`\`\`

Current project context:
${context.fileContexts.map(fc => `- ${fc.filePath} (${fc.summary})`).join('\n')}

Current conversation:
${conversationHistory.map(msg => `${msg.role.toUpperCase()}: ${msg.content}`).join('\n\n')}`;

    return `${systemPrompt}\n\nUSER: ${message}\n\nASSISTANT:`;
  }
  
  /**
   * Parse file modifications from LLM response
   * 
   * @param {string} response LLM response text
   * @returns {Array<Object>} Array of file modifications
   */
  function parseFileModifications(response) {
    const modifications = [];
    const fileBlockRegex = /\`\`\`file:(.*?)\n([\s\S]*?)\`\`\`/g;
    
    let match;
    while ((match = fileBlockRegex.exec(response)) !== null) {
      const filePath = match[1].trim();
      const content = match[2];
      
      modifications.push({
        filePath,
        content
      });
    }
    
    return modifications;
  }
  
  /**
   * Initialize agent context with project files
   * 
   * @returns {Promise<void>}
   */
  async function initialize() {
    logger.info('Initializing agent context...');
    
    // Load key project files
    for (const file of projectFiles.slice(0, 10)) {  // Start with first 10 for performance
      try {
        await loadFileContext(file);
      } catch (error) {
        // Continue with other files on error
        logger.warn(`Failed to load file: ${file}`, { error });
      }
    }
    
    logger.info('Agent context initialized');
  }
  
  // Initialize immediately
  initialize();
  
  // Return the agent interface
  return {
    processMessage,
    loadFileContext,
    modifyFile,
    findFiles,
    getContextManager: () => contextManager,
    getFileContexts: () => fileContexts,
    getConversationHistory: () => conversationHistory,
    reset: () => {
      conversationHistory = [];
      contextManager.reset();
    }
  };
}

module.exports = {
  createAgent
};
```

# apps/synthbot/src/agent/context.js

```js

```

# apps/synthbot/src/agent/fileManager.js

```js

```

# apps/synthbot/src/agent/index.js

```js

```

# apps/synthbot/src/agent/tokenizer.js

```js

```

# apps/synthbot/src/api/client.js

```js
/**
 * API Client for DistributedLLM
 * 
 * Handles communication with the DistributedLLM coordinator,
 * including WebSocket connections an
```

# apps/synthbot/src/api/index.js

```js

```

# apps/synthbot/src/api/queue.js

```js

```

# apps/synthbot/src/api/websocket.js

```js

```

# apps/synthbot/src/index.js

```js
/**
 * SynthBot - Main application entry point
 * 
 * This module initializes and coordinates the different components:
 * - Terminal UI
 * - Agent
 * - Distributed LLM API client
 */

const { initAgent } = require('./agent');
const { createClient } = require('./api');
const { createApp } = require('./tui');
const { 
  logger, 
  scanProjectFiles, 
  setupLogging,
  createTokenMonitor
} = require('./utils');

/**
 * Start the SynthBot application
 * @param {Object} config Configuration object
 */
async function startApp(config) {
  try {
    // Setup logging based on configuration
    setupLogging(config.logging);
    
    logger.info('Starting SynthBot...');
    logger.debug('Configuration loaded', { config });
    
    // Initial project scan
    logger.info('Scanning project files...');
    const projectFiles = await scanProjectFiles(config.projectRoot, {
      exclude: config.context.excludeDirs.concat(config.context.excludeFiles),
      prioritize: config.context.priorityFiles
    });
    
    logger.info(`Found ${projectFiles.length} files in project`);
    
    // Initialize API client
    logger.info('Connecting to DistributedLLM coordinator...');
    const apiClient = createClient({
      host: config.llm.coordinatorHost,
      port: config.llm.coordinatorPort,
      model: config.llm.model,
      temperature: config.llm.temperature
    });
    
    // Create token monitor
    const tokenMonitor = createTokenMonitor({
      maxTokens: config.context.maxTokens,
      warningThreshold: 0.8 // Warn at 80% usage
    });
    
    // Initialize agent
    logger.info('Initializing agent...');
    const agent = initAgent({
      apiClient,
      tokenMonitor,
      projectFiles,
      projectRoot: config.projectRoot
    });
    
    // Create and start the TUI
    logger.info('Starting Terminal UI...');
    const app = createApp({
      agent,
      apiClient,
      tokenMonitor,
      config: config.ui,
      projectRoot: config.projectRoot
    });
    
    // Handle application shutdown
    function shutdown() {
      logger.info('Shutting down SynthBot...');
      app.destroy();
      process.exit(0);
    }
    
    // Handle signals
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
    
    // Start the TUI
    app.start();
    
    logger.info('SynthBot started successfully');
    
  } catch (error) {
    logger.error('Failed to start SynthBot', { error });
    console.error('Failed to start SynthBot:', error.message);
    process.exit(1);
  }
}

module.exports = {
  startApp
};
```

# apps/synthbot/src/tui/app.js

```js
/**
 * Main Terminal UI Application
 * 
 * Creates and manages the terminal interface using blessed/blessed-contrib
 */

const blessed = require('blessed');
const contrib = require('blessed-contrib');
const { applyTheme } = require('./themes');
const { createInputHandler } = require('./input');
const { createOutputRenderer } = require('./output');
const { createStatusBar } = require('./statusBar');
const { createFileTree } = require('./fileTree');
const { logger } = require('../utils');

/**
 * Create the TUI application
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.agent The agent instance
 * @param {Object} options.apiClient The API client
 * @param {Object} options.tokenMonitor Token monitoring utility
 * @param {Object} options.config UI configuration
 * @param {string} options.projectRoot Project root directory
 * @returns {Object} The TUI application object
 */
function createApp({ agent, apiClient, tokenMonitor, config, projectRoot }) {
  // Create a screen object
  const screen = blessed.screen({
    smartCSR: true,
    title: 'SynthBot',
    fullUnicode: true,
    dockBorders: true,
    autoPadding: true
  });
  
  // Apply theme
  applyTheme(screen, config.theme);
  
  // Create layout grid
  const grid = new contrib.grid({ rows: 12, cols: 12, screen });
  
  // Create file tree panel (left side, 2/12 of width)
  const fileTreePanel = grid.set(0, 0, 10, 2, contrib.tree, {
    label: 'Project Files',
    style: {
      selected: {
        fg: 'white',
        bg: 'blue'
      }
    },
    template: {
      lines: true
    },
    tags: true
  });
  
  // Create conversation panel (right side, 10/12 of width)
  const conversationPanel = grid.set(0, 2, 8, 10, blessed.log, {
    label: 'Conversation',
    scrollable: true,
    alwaysScroll: true,
    scrollbar: {
      ch: ' ',
      style: {
        bg: 'blue'
      },
      track: {
        bg: 'black'
      }
    },
    mouse: true,
    tags: true
  });
  
  // Create input box (bottom right, span 10/12 of width)
  const inputBox = grid.set(8, 2, 2, 10, blessed.textarea, {
    label: 'Command',
    inputOnFocus: true,
    padding: {
      top: 1,
      left: 2
    },
    style: {
      focus: {
        border: {
          fg: 'blue'
        }
      }
    },
    border: {
      type: 'line'
    }
  });
  
  // Create status bar (bottom of screen, full width)
  const statusBar = grid.set(10, 0, 2, 12, blessed.box, {
    tags: true,
    content: ' {bold}Status:{/bold} Ready',
    style: {
      fg: 'white',
      bg: 'blue'
    },
    padding: {
      left: 1,
      right: 1
    }
  });
  
  // Initialize components
  const fileTree = createFileTree({
    widget: fileTreePanel,
    projectRoot,
    agent
  });
  
  const outputRenderer = createOutputRenderer({
    widget: conversationPanel,
    tokenMonitor
  });
  
  const statusBarController = createStatusBar({
    widget: statusBar,
    apiClient,
    tokenMonitor
  });
  
  const inputHandler = createInputHandler({
    widget: inputBox,
    outputRenderer,
    agent,
    fileTree,
    screen
  });
  
  // Set up key bindings
  screen.key(['C-c'], () => {
    return process.exit(0);
  });
  
  screen.key(['tab'], () => {
    if (screen.focused === inputBox) {
      fileTreePanel.focus();
    } else {
      inputBox.focus();
    }
  });
  
  screen.key(['C-r'], () => {
    fileTree.refresh();
    statusBarController.update('Refreshing project files...');
  });
  
  screen.key(['C-l'], () => {
    conversationPanel.setContent('');
    screen.render();
    statusBarController.update('Conversation cleared');
  });
  
  screen.key(['C-s'], () => {
    // Save conversation implementation
    statusBarController.update('Conversation saved');
  });
  
  // Focus input by default
  inputBox.focus();
  
  // Initialize file tree
  fileTree.init();
  
  // Return the application object
  return {
    screen,
    start: () => {
      // Initial rendering
      screen.render();
      
      // Welcome message
      outputRenderer.addSystemMessage('Welcome to SynthBot! Type your question or command below.');
      statusBarController.update('Ready');
    },
    destroy: () => {
      screen.destroy();
    }
  };
}

module.exports = {
  createApp
};
```

# apps/synthbot/src/tui/fileTree.js

```js

```

# apps/synthbot/src/tui/index.js

```js

```

# apps/synthbot/src/tui/input.js

```js

```

# apps/synthbot/src/tui/output.js

```js

```

# apps/synthbot/src/tui/statusBar.js

```js

```

# apps/synthbot/src/tui/themes.js

```js

```

# apps/synthbot/src/utils/config.js

```js

```

# apps/synthbot/src/utils/fileSystem.js

```js

```

# apps/synthbot/src/utils/formatters.js

```js

```

# apps/synthbot/src/utils/git.js

```js

```

# apps/synthbot/src/utils/index.js

```js

```

# apps/synthbot/src/utils/logger.js

```js

```

# config/cluster_config.yaml

```yaml
# DistributedLLM Cluster Configuration

# Coordinator node configuration
coordinator:
  host: "192.168.1.100"  # IP address of the coordinator machine
  port: 5555             # Port for worker communication
  dashboard_port: 8080   # Web dashboard for monitoring
  max_workers: 10        # Maximum number of workers to connect

# Worker nodes
workers:
  - id: "gaming-pc"
    host: "192.168.1.101"
    port: 5556
    os: "linux"
    capabilities:
      cpu_cores: 16
      cpu_type: "AMD Ryzen 9 5900X"
      ram_gb: 64
      gpu: "NVIDIA RTX 3080"
      gpu_memory_gb: 10
      disk_speed_mbps: 3000
      priority: 1  # Lower number = higher priority

  - id: "macbook-pro-m2"
    host: "192.168.1.102"
    port: 5556
    os: "macos"
    capabilities:
      cpu_cores: 10
      cpu_type: "Apple M2 Pro"
      ram_gb: 32
      gpu: "Apple Integrated"
      gpu_memory_gb: 16
      disk_speed_mbps: 2500
      priority: 2

  - id: "windows-laptop"
    host: "192.168.1.103"
    port: 5556
    os: "windows"
    capabilities:
      cpu_cores: 8
      cpu_type: "Intel i7-11800H"
      ram_gb: 16
      gpu: "NVIDIA RTX 3060 Mobile"
      gpu_memory_gb: 6
      disk_speed_mbps: 1800
      priority: 3

# Network configuration
network:
  timeout_seconds: 30
  retry_attempts: 3
  heartbeat_interval_seconds: 5
  max_message_size_mb: 500
  compression: true
  encryption: true
  protocol: "tcp"  # tcp or udp
  
  # Internet settings
  internet_mode: true
  use_ssl: true
  cert_file: "certs/server.crt"
  key_file: "certs/server.key"
  
  # Connection quality thresholds
  high_latency_threshold_ms: 100
  bandwidth_minimum_mbps: 5
  
  # Timeouts for internet
  connection_timeout_seconds: 60
  heartbeat_interval_seconds: 15
  task_timeout_multiplier: 5
  
  # Registration
  coordinator_public_address: "your-domain.com"  # Public DNS or IP
  registration_token: "your-secure-token"

# Auto-discovery settings
discovery:
  enabled: true
  broadcast_port: 5557
  broadcast_interval_seconds: 10
  auto_register: true
```

# config/model_config.yaml

```yaml
# LLM Model Configuration

model:
  name: "llama-2-13b"  # Model identifier
  family: "llama"      # Model architecture family
  version: "2.0"
  size_billion_parameters: 13
  
  # Model architecture details
  architecture:
    hidden_size: 5120
    num_attention_heads: 40
    num_hidden_layers: 40
    intermediate_size: 13824
    max_position_embeddings: 4096
    vocab_size: 32000
    attention_implementation: "flash_attention"  # Options: standard, flash_attention, xformers
  
  # File locations
  paths:
    base_path: "./models"
    weights_file: "llama-2-13b.pth"
    tokenizer_file: "tokenizer.model"
    config_file: "config.json"
  
  # Quantization settings
  quantization:
    enabled: true
    method: "int8"  # Options: none, int8, int4, fp16, nf4
    per_channel: true
    calibration_dataset: "calibration_data.jsonl"
  
  # Parallelism strategy
  parallelism:
    approach: "tensor_parallel"  # Options: tensor_parallel, pipeline_parallel, expert_parallel, hybrid
    tensor_parallel_size: 0  # 0 means auto-configure based on available devices
    pipeline_parallel_size: 0
    layer_distribution: "auto"  # Can be auto or a specific pattern like [0,0,0,1,1,1,2,2,2]
  
  # Inference settings
  inference:
    max_batch_size: 8
    max_sequence_length: 2048
    temperature: 0.7
    top_p: 0.9
    top_k: 40
    repetition_penalty: 1.1
    do_sample: true
  
  # Optimization settings
  optimization:
    kv_cache_enabled: true
    attention_implementation: "flash_attention"
    mixed_precision: true
    compile_graph: true
    optimization_level: 3  # 0-3, with 3 being most aggressive optimization
    memory_efficient_attention: true

# Deployment-specific settings
deployment:
  layer_allocation:
    strategy: "memory_balanced"  # Options: compute_balanced, memory_balanced, network_optimized, custom
    recomputation: false  # Whether to recompute forward activations during backward pass
    pin_layers: false  # Whether to pin layers to specific workers
  
  caching:
    kv_cache_enabled: true
    disk_offload_enabled: true
    max_disk_cache_size_gb: 100
  
  communication:
    compress_activations: true
    overlap_computation: true
    gradient_accumulation_steps: 1
  
  fallback:
    enabled: true  # Whether to fall back to CPU if GPU memory is exhausted
    cpu_offload_layers: ["embed"]  # Layers to offload to CPU even when GPU is available
```

# config/workload_config.yaml

```yaml
# Workload Balancing Configuration

# Load balancing strategy
load_balancing:
  strategy: "dynamic"  # Options: static, dynamic, adaptive
  rebalance_interval_seconds: 30
  performance_history_window: 10  # Number of tasks to consider for performance averaging
  
  # Performance metrics to consider for balancing
  metrics:
    - name: "compute_time"
      weight: 0.5
    - name: "memory_usage"
      weight: 0.3
    - name: "network_latency"
      weight: 0.2
  
  # Minimum task sizes to avoid excessive communication overhead
  min_task_sizes:
    inference_tokens: 64
    batch_size: 1
    layers: 1
  
  # Thresholds for rebalancing
  thresholds:
    worker_idle_percent_trigger: 15.0  # Trigger rebalance if a worker is idle >15% of time
    performance_variance_trigger: 25.0  # Trigger if >25% variance across workers
    network_saturation_percent: 80.0    # Throttle if network is >80% saturated
    memory_headroom_percent: 10.0       # Required free memory percentage

# Worker allocation
worker_allocation:
  initial_method: "capability_weighted"  # Options: equal, capability_weighted, memory_weighted
  
  # Capability weights for initial allocation
  capability_weights:
    cpu_cores: 0.3
    ram_gb: 0.4
    gpu_memory_gb: 0.3
  
  # Handling stragglers
  straggler_detection:
    enabled: true
    threshold_percent: 150  # A worker is a straggler if it takes 1.5x the median time
    action: "reassign"  # Options: reassign, duplicate, ignore
  
  # Worker specialization
  specialization:
    enabled: true
    specialize_after_tasks: 50  # Number of tasks to analyze before specializing
    max_specialized_workers_percent: 80  # Maximum % of workers to specialize

# Task scheduling
task_scheduling:
  method: "priority_queue"  # Options: fifo, priority_queue, deadline
  preemption: false
  
  # Task priority factors
  priority_factors:
    worker_affinity: 0.4
    task_size: 0.3
    request_timestamp: 0.2
    worker_load: 0.1
  
  # Worker warmup
  warmup:
    enabled: true
    tasks_per_worker: 5
    ignore_metrics_during_warmup: true

# Fault tolerance
fault_tolerance:
  worker_timeout_seconds: 60
  task_timeout_multiplier: 3.0  # Task timeout = expected_duration * multiplier
  max_retries: 3
  backup_workers_percent: 15  # Keep this % of workers as backup for critical tasks
  checkpointing_interval_seconds: 300
```

# docker/coordinator.Dockerfile

```Dockerfile

```

# docker/docker-compose.yml

```yml

```

# docker/worker.Dockerfile

```Dockerfile

```

# docs/model_sharding.md

```md
# DistributedLLM: Model Sharding and Distribution

This document explains how the DistributedLLM system distributes and shards large language models across multiple devices and machines.

## Core Concepts

DistributedLLM enables running large language models on distributed hardware by splitting model components across multiple devices (GPUs and CPUs) and multiple machines. This allows running models that would otherwise be too large for any single device.

## Model Sharding Architecture

### 1. Device Mapping Strategies

The system supports multiple distribution strategies through the `ShardedModelLoader` class:

- **Auto**: Automatically places model components based on available memory
- **Balanced**: Distributes layers evenly across available devices
- **Custom**: Enables explicit mapping of components to specific devices

\`\`\`python
# Example device mapping
device_map = {
    "embedding": "cuda:0",
    "transformer.blocks.0": "cuda:0",
    "transformer.blocks.1": "cuda:1",
    "transformer.blocks.2": "cuda:1",
    "lm_head": "cpu"
}
\`\`\`

### 2. Layer-Based Distribution

Models are divided into distinct components that can be placed on different devices:

- **Embeddings**: Token embedding lookup tables
- **Transformer Blocks**: Multiple attention and feed-forward layers
- **Layer Normalization**: Normalization layers between components
- **LM Head**: Output projection layer

### 3. Cross-Device Communication

When running inference, the system automatically handles data transfer between devices:

\`\`\`python
# From ShardedModel.forward()
for block in self.blocks:
    # Ensure hidden states are on the right device for this block
    block_device = block.input_layernorm.weight.device
    if hidden_states.device != block_device:
        hidden_states = hidden_states.to(block_device)
    
    hidden_states = block(hidden_states, attention_mask)
\`\`\`

### 4. Cross-Machine Distribution

The system distributes computation across multiple worker machines:

1. **Layer Allocation**: The coordinator assigns specific layers to specific workers
2. **Task Distribution**: Computation is broken into tasks that are assigned to workers
3. **Result Aggregation**: Results from different workers are combined into the final output

## Distributed Inference Process

The inference process follows these steps when distributed across multiple machines:

1. **Input Processing**:
   - Coordinator tokenizes the input prompt
   - Task planning for each model component

2. **Forward Pass**:
   - Workers compute their assigned layers sequentially
   - Data flows between workers: Worker1 (embedding) → Worker2 (layers 0-3) → Worker3 (layers 4-8) → etc.

3. **Token Generation**:
   - The final layer produces token probabilities
   - Coordinator samples the next token and prepares for the next step
   - Process repeats for each new token

4. **Result Collection**:
   - Generated tokens are assembled into the final output text
   - Results returned to the client

## Dynamic Load Balancing

The system continually optimizes distribution based on observed performance:

1. **Performance Monitoring**:
   - Track execution times per worker per task type
   - Monitor resource utilization (CPU, memory, GPU)

2. **Workload Rebalancing**:
   - Redistributes layers based on worker performance
   - Adjusts task assignments to balance the load

\`\`\`python
# From scheduler.py
def _recalculate_layer_allocation(self):
    # Distribute layers proportionally to performance scores
    for worker_id, score in worker_metrics.items():
        layer_count = max(1, int(total_layers * score))
        # ...assign layers based on performance...
\`\`\`

## Memory Optimization Techniques

The system implements several techniques to optimize memory usage:

1. **Selective Loading**: Only load model weights needed for assigned layers
2. **CPU Offloading**: Move less frequently used layers to CPU memory
3. **KV-Cache Management**: Efficient management of attention key-value caches
4. **Quantization Support**: Int8, FP16, and other reduced precision formats
5. **Gradient-Free Inference**: No need to store activation gradients during inference

## Fault Tolerance

The distributed architecture includes mechanisms for fault tolerance:

1. **Worker Failure Detection**: Through heartbeat monitoring
2. **Task Reassignment**: Failed tasks are reassigned to other workers
3. **Graceful Degradation**: System can continue with fewer workers if needed
4. **Straggler Mitigation**: Slow workers get fewer or simpler tasks

## Performance Considerations

Key factors affecting distributed inference performance:

1. **Communication Overhead**: Network bandwidth and latency between machines
2. **Load Balancing Quality**: How evenly work is distributed
3. **Worker Heterogeneity**: Different capabilities across machines
4. **Layer Allocation Strategy**: Which layers are assigned to which workers
```

# docs/overview.md

```md
# DistributedLLM: High-Level Architecture Overview

## Introduction

DistributedLLM is a framework that enables running large language models across multiple machines with heterogeneous hardware configurations. It's designed to address the challenges of running resource-intensive LLM inference when no single machine has enough computational power or memory to efficiently run the entire model.

The system implements a "Boss-Workers" (also known as Master-Slave) distributed computing model where a coordinator node orchestrates multiple worker nodes to collaboratively perform inference tasks.

## Core Concepts

### Heterogeneous Computing

The framework is designed to work with a mix of different hardware:
- High-end servers with multiple GPUs
- Desktop computers with consumer GPUs
- Laptops with integrated graphics
- Machines running different operating systems (Linux, macOS, Windows)

This heterogeneity is embraced rather than seen as a limitation, with the system dynamically adapting based on the capabilities of each node.

### Model Sharding

Large models are split across multiple devices using model sharding techniques:
- **Automatic sharding**: System analyzes available resources and distributes model components optimally
- **Layer distribution**: Transformer layers can be split across multiple GPUs
- **Component separation**: Model components (embeddings, transformer blocks, output heads) can be assigned to different machines
- **Memory-CPU tradeoffs**: Parts of the model can be offloaded to CPU when GPU memory is limited

### Dynamic Load Balancing

The system continuously monitors performance and adjusts workload distribution:
- Faster machines get assigned more computation tasks
- Performance history influences future task assignments
- Workers can join or leave the cluster dynamically
- The system adapts to changing conditions (like other applications competing for resources)

## System Architecture

The distributed system consists of three main components:

### 1. Coordinator Node

**Purpose**: Orchestrates the distributed computation across all workers.

**Key Responsibilities**:
- Maintains registry of available worker nodes
- Splits computation into tasks
- Assigns tasks to workers based on their capabilities
- Monitors worker performance
- Aggregates results from workers
- Handles worker failures and task reassignment
- Provides a unified interface for clients to submit inference requests

### 2. Worker Nodes

**Purpose**: Execute assigned computation tasks and report results back to the coordinator.

**Key Responsibilities**:
- Report hardware capabilities to coordinator
- Receive and execute assigned tasks
- Monitor local resource utilization
- Process model components and run inference locally
- Return results to coordinator
- Handle task failures gracefully

### 3. Client Interface

**Purpose**: Provides a simple interface for applications to use the distributed system.

**Key Features**:
- Similar API to popular ML frameworks
- Abstracts away the complexity of distributed inference
- Handles tokenization, generation parameters, and result processing

## Workflow

### System Initialization

1. The coordinator starts and begins listening for worker connections
2. Worker nodes start and connect to the coordinator
3. Each worker reports its hardware capabilities and available resources
4. Coordinator builds a registry of available computing resources
5. Model weights are distributed to workers based on their capabilities

### Inference Process

1. Client submits a prompt for inference
2. Coordinator tokenizes the input and plans the computation
3. For an autoregressive model (like GPT or LLaMA):
   - Input processing tasks are distributed to available workers
   - Initial token generation tasks are assigned
   - As tokens are generated, new tasks are created for the next token
   - Workers pull new tasks when they complete previous ones
4. Results are aggregated by the coordinator and returned to the client

### Dynamic Adaptation

Throughout the inference process:
1. Performance metrics are collected from each worker
2. The scheduler adjusts task assignments based on observed performance
3. If a worker becomes unavailable, its tasks are reassigned
4. If a new worker joins, it's incorporated into the task distribution

## Key Components

### Coordinator Components

- **Scheduler**: Distributes tasks to workers based on their capabilities
- **Performance Monitor**: Tracks worker performance for load balancing
- **Result Aggregator**: Combines outputs from distributed model components

### Worker Components

- **Compute Engine**: Executes assigned tasks using local hardware
- **Resource Monitor**: Tracks system resource usage and availability
- **Communication Client**: Handles network communication with coordinator

### Model Components

- **Model Layers**: Implementations of model architecture components
- **Tokenizer**: Converts text to tokens and vice versa
- **Inference Engine**: Handles the inference process logic

## Performance Considerations

### Communication Overhead

- Task granularity is balanced to minimize communication overhead
- Large tensors are serialized efficiently
- Results are cached to avoid redundant computation

### Memory Management

- The KV cache is distributed across workers
- Memory is managed carefully to avoid OOM errors
- Unused model components can be offloaded to save memory

### Fault Tolerance

- The system handles worker disconnections gracefully
- Tasks from failed workers are reassigned
- Results are cached to minimize lost work

## Setup and Deployment

The system includes setup scripts for all major platforms:
1. **Coordinator Setup**: Run on the machine that will coordinate tasks
2. **Worker Setup**: Run on each machine that will participate in the computation
3. **Configuration**: Edit YAML files to adjust settings for your environment
4. **Launch**: Start the coordinator, then start workers on each machine

## Usage Example

\`\`\`python
from distributed_llm import DistributedClient

# Connect to the coordinator
client = DistributedClient(coordinator_address="192.168.1.100:5555")

# Generate text
response = client.generate(
    prompt="Explain quantum computing in simple terms",
    max_new_tokens=500,
    temperature=0.7
)

print(response)
\`\`\`

## Benchmarking and Optimization

The `benchmark.py` script helps optimize the configuration for your specific hardware mix:
1. Test different sharding strategies
2. Measure throughput under different loads
3. Find the optimal batch size
4. Determine the best worker configuration

## Conclusion

DistributedLLM enables running large language models on hardware that would otherwise be insufficient, by combining the computing power of multiple machines. The Boss-Workers model with dynamic load balancing ensures efficient resource utilization, while the model sharding approach allows flexibility in how model components are distributed.

The result is a system that can run state-of-the-art language models like LLaMA, GPT, or T5 across a network of consumer-grade hardware with performance approaching that of specialized high-end servers.
```

# README.md

```md
# DistributedLLM 🌐🧠

> *Harness the collective power of heterogeneous hardware to run large language models beyond the capabilities of any single machine.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Status: Experimental](https://img.shields.io/badge/Status-Experimental-orange)
![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue)

## 🔮 Vision

DistributedLLM brings the power of supercomputing to your personal hardware collection. By combining the computational resources of diverse machines—from powerful desktops to modest laptops, across Windows, macOS, and Linux—our framework enables running inference on large language models that would typically require expensive specialized hardware.

**Think SETI@home, but for AI.**

## 🧩 How It Works

DistributedLLM implements a dynamic "Boss-Workers" model inspired by high-performance computing techniques for heterogeneous clusters:

\`\`\`mermaid
graph TD
    A[Coordinator Node] -->|Distributes Model Layers| B[Gaming PC/Linux Worker]
    A -->|Assigns Computation| C[MacBook Worker]
    A -->|Balances Workload| D[Windows Laptop Worker]
    B -->|Returns Results| A
    C -->|Returns Results| A
    D -->|Returns Results| A
\`\`\`

1. **Smart Partitioning**: The model is dynamically split across available machines based on their computational power and memory capacity
2. **Adaptive Load Balancing**: Work distribution automatically adjusts as machine availability and performance fluctuate
3. **Efficient Communication**: Optimized data transfer protocols minimize network overhead
4. **Unified Interface**: Single control point abstracts away the complexity of the distributed system

## ✨ Key Features

- **Truly Heterogeneous**: Run across Windows, macOS, and Linux machines simultaneously
- **Containerized Compatibility**: Docker support for consistent environment handling
- **Dynamic Scaling**: Add or remove machines on the fly
- **Performance Monitoring**: Real-time visualization of distributed workload
- **Fault Tolerance**: Graceful handling of machine disconnections
- **Minimal Setup**: Simple configuration with automatic capability detection

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Network connectivity between all machines
- Git

### Quick Start

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/yourusername/distributed-llm.git
   cd distributed-llm
   \`\`\`

2. Set up the coordinator (on your most reliable machine):
   \`\`\`bash
   ./scripts/setup_coordinator.sh
   \`\`\`

3. Set up workers (on each participating machine):
   \`\`\`bash
   # For Linux
   ./scripts/setup_worker_linux.sh
   
   # For macOS
   ./scripts/setup_worker_mac.sh
   
   # For Windows (run in Command Prompt)
   scripts\setup_worker_windows.bat
   \`\`\`

4. Edit the configuration:
   \`\`\`bash
   nano config/cluster_config.yaml
   \`\`\`

5. Start the system:
   \`\`\`bash
   python src/main.py
   \`\`\`

## 🌟 Use Cases

- **Research Exploration**: Run cutting-edge LLMs on your existing hardware collection
- **Collaborative Computing**: Pool resources within research groups or among friends
- **Cost-Effective Inference**: Avoid cloud computing costs by utilizing hardware you already own
- **Educational Tool**: Learn about distributed systems and machine learning
- **Hardware Recycling**: Give new purpose to older machines as dedicated inference nodes

## 📊 Performance

Performance varies based on your hardware mix, network conditions, and model configuration. Here are some example benchmarks:

| Cluster Composition | Model Size | Tokens/second | Speedup vs. Best Single Machine |
|---------------------|------------|---------------|--------------------------------|
| 1 Gaming PC, 2 MacBooks | 7B | 12.3 | 2.4x |
| 2 Gaming PCs, 1 Linux Server, 1 MacBook | 13B | 8.7 | 3.1x |
| 3 Gaming PCs, 2 MacBooks, 1 Linux Server | 30B | 4.2 | 4.8x |

## 🛠️ Architecture

DistributedLLM uses a layered architecture:

- **Coordinator Layer**: Orchestrates the distribution of work
- **Communication Layer**: Handles cross-platform networking
- **Compute Layer**: Manages the actual model inference
- **Monitoring Layer**: Tracks performance and adjusts workload distribution

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and submission process.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌐 Citation

If you use DistributedLLM in your research, please cite:

\`\`\`bibtex
@software{DistributedLLM2023,
  author = {Francis Gillet},
  title = {DistributedLLM: Heterogeneous Distributed Computing for Large Language Models},
  year = {2023},
  url = {https://github.com/fgillet4/DitributedLLM}
}
\`\`\`

## ✉️ Contact

Questions? Ideas? Reach out!

- GitHub Issues: Preferred for technical questions and bug reports
- Email: fgillet4l@gmail.com

---

*"Alone we can do so little; together we can do so much." – Helen Keller*
```

# requirements.txt

```txt
# Core dependencies
numpy>=1.20.0
psutil>=5.9.0
pyyaml>=6.0
tqdm>=4.64.0

# Networking
requests>=2.28.0
websockets>=10.3.0

# Monitoring
gpustat>=1.0.0; platform_system != "Darwin"
py-cpuinfo>=8.0.0

# Machine learning
torch>=1.13.0
accelerate>=0.19.0
transformers>=4.30.0
sentencepiece>=0.1.99
tokenizers>=0.13.3

# Visualization
matplotlib>=3.5.0
plotly>=5.10.0
dash>=2.7.0

# Utilities
filelock>=3.9.0
pydantic>=1.10.0
fastapi>=0.95.0
uvicorn>=0.20.0

# Distributed systems
ray>=2.3.0

# Command-line interface
click>=8.1.3
rich>=13.0.0

# Testing
pytest>=7.0.0
pytest-xdist>=3.0.0
```

# scripts/benchmark.py

```py
#!/usr/bin/env python3
"""
Benchmark script for the DistributedLLM system.

This script benchmarks the performance of different distributed configurations
to find the optimal settings for a given hardware setup.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.inference import ModelInference, DistributedInference
from src.worker.compute_engine import ComputeEngine
from src.worker.resource_monitor import ResourceMonitor
from src.worker.communication import CoordinatorClient, TaskProcessor
from src.coordinator.scheduler import Scheduler
from src.utils.networking import find_available_port


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger("Benchmark")


def benchmark_local_inference(model_id: str, num_iterations: int = 5, batch_size: int = 1):
    """
    Benchmark local inference performance.
    
    Args:
        model_id: ID of the model to benchmark
        num_iterations: Number of benchmark iterations
        batch_size: Batch size for inference
    
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Benchmarking local inference for model {model_id}")
    
    try:
        # Initialize model
        logger.info("Initializing model...")
        inference = ModelInference(model_id)
        
        # Prepare benchmark prompts
        prompt = "The quick brown fox jumps over the lazy dog."
        if batch_size > 1:
            prompts = [prompt] * batch_size
        else:
            prompts = prompt
        
        # Run warmup
        logger.info("Running warmup...")
        _ = inference.generate(prompts, max_new_tokens=20)
        
        # Run benchmark
        logger.info(f"Running {num_iterations} iterations with batch size {batch_size}...")
        times = []
        tokens_generated = []
        
        for i in range(num_iterations):
            start_time = time.time()
            result = inference.generate(prompts, max_new_tokens=100)
            end_time = time.time()
            
            # Calculate tokens generated
            num_tokens = inference.inference_stats["total_tokens_generated"] / inference.inference_stats["total_requests"]
            tokens_generated.append(num_tokens)
            
            # Record time
            times.append(end_time - start_time)
            
            logger.info(f"Iteration {i+1}/{num_iterations}: {times[-1]:.2f}s, {num_tokens:.1f} tokens")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_second = avg_tokens / avg_time
        
        # Get model statistics
        stats = inference.get_stats()
        
        # Compile results
        results = {
            "model_id": model_id,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "average_time": avg_time,
            "average_tokens": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "model_stats": stats
        }
        
        logger.info(f"Benchmark results: {tokens_per_second:.2f} tokens/sec")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in local inference benchmark: {e}")
        return {"error": str(e)}


def benchmark_distributed_inference(
    model_id: str,
    num_workers: int = 2,
    num_iterations: int = 5,
    batch_size: int = 1
):
    """
    Benchmark distributed inference performance.
    
    Args:
        model_id: ID of the model to benchmark
        num_workers: Number of worker processes to use
        num_iterations: Number of benchmark iterations
        batch_size: Batch size for inference
    
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Benchmarking distributed inference for model {model_id} with {num_workers} workers")
    
    try:
        # Start coordinator
        coordinator_port = find_available_port(5000, 6000)
        if not coordinator_port:
            raise RuntimeError("Could not find available port for coordinator")
        
        # Start coordinator process
        # In a real benchmark, we would start an actual coordinator process
        # For this mock implementation, we'll just simulate it
        logger.info(f"Starting coordinator on port {coordinator_port}...")
        
        # Start worker processes
        worker_ports = []
        for _ in range(num_workers):
            port = find_available_port(6001, 7000)
            if not port:
                raise RuntimeError("Could not find available port for worker")
            worker_ports.append(port)
        
        logger.info(f"Starting {num_workers} workers on ports {worker_ports}...")
        
        # In a real benchmark, we would start actual worker processes
        # For this mock implementation, we'll just simulate them
        
        # Create DistributedInference client
        logger.info("Initializing distributed inference client...")
        # For this mock implementation, we'll just create a ModelInference instance
        inference = ModelInference(model_id)
        
        # Prepare benchmark prompts
        prompt = "The quick brown fox jumps over the lazy dog."
        if batch_size > 1:
            prompts = [prompt] * batch_size
        else:
            prompts = prompt
        
        # Run warmup
        logger.info("Running warmup...")
        _ = inference.generate(prompts, max_new_tokens=20)
        
        # Run benchmark
        logger.info(f"Running {num_iterations} iterations with batch size {batch_size}...")
        times = []
        tokens_generated = []
        
        for i in range(num_iterations):
            start_time = time.time()
            result = inference.generate(prompts, max_new_tokens=100)
            end_time = time.time()
            
            # Calculate tokens generated
            num_tokens = inference.inference_stats["total_tokens_generated"] / inference.inference_stats["total_requests"]
            tokens_generated.append(num_tokens)
            
            # Record time
            times.append(end_time - start_time)
            
            logger.info(f"Iteration {i+1}/{num_iterations}: {times[-1]:.2f}s, {num_tokens:.1f} tokens")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_second = avg_tokens / avg_time
        
        # Get model statistics
        stats = inference.get_stats()
        
        # Compile results
        results = {
            "model_id": model_id,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "average_time": avg_time,
            "average_tokens": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "model_stats": stats
        }
        
        logger.info(f"Benchmark results: {tokens_per_second:.2f} tokens/sec with {num_workers} workers")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in distributed inference benchmark: {e}")
        return {"error": str(e)}


def benchmark_model_sharding(
    model_id: str,
    shard_config: List[Dict[str, Any]],
    num_iterations: int = 5
):
    """
    Benchmark model sharding configurations.
    
    Args:
        model_id: ID of the model to benchmark
        shard_config: List of sharding configurations to test
        num_iterations: Number of benchmark iterations
    
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Benchmarking model sharding for model {model_id}")
    
    try:
        results = []
        
        for config in shard_config:
            logger.info(f"Testing sharding configuration: {config}")
            
            # Initialize model with this sharding configuration
            logger.info("Initializing model...")
            inference = ModelInference(model_id, device_map=config["device_map"])
            
            # Prepare benchmark prompts
            prompt = "The quick brown fox jumps over the lazy dog."
            
            # Run warmup
            logger.info("Running warmup...")
            _ = inference.generate(prompt, max_new_tokens=20)
            
            # Run benchmark
            logger.info(f"Running {num_iterations} iterations...")
            times = []
            tokens_generated = []
            
            for i in range(num_iterations):
                start_time = time.time()
                result = inference.generate(prompt, max_new_tokens=100)
                end_time = time.time()
                
                # Calculate tokens generated
                num_tokens = inference.inference_stats["total_tokens_generated"] / inference.inference_stats["total_requests"]
                tokens_generated.append(num_tokens)
                
                # Record time
                times.append(end_time - start_time)
                
                logger.info(f"Iteration {i+1}/{num_iterations}: {times[-1]:.2f}s, {num_tokens:.1f} tokens")
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            avg_tokens = sum(tokens_generated) / len(tokens_generated)
            tokens_per_second = avg_tokens / avg_time
            
            # Compile results for this configuration
            config_results = {
                "device_map": config["device_map"],
                "average_time": avg_time,
                "average_tokens": avg_tokens,
                "tokens_per_second": tokens_per_second,
            }
            
            results.append(config_results)
            
            logger.info(f"Configuration results: {tokens_per_second:.2f} tokens/sec")
        
        # Compile overall results
        benchmark_results = {
            "model_id": model_id,
            "num_iterations": num_iterations,
            "configurations": results
        }
        
        return benchmark_results
    
    except Exception as e:
        logger.error(f"Error in model sharding benchmark: {e}")
        return {"error": str(e)}


def save_results(results, output_file: str):
    """
    Save benchmark results to a file.
    
    Args:
        results: Benchmark results to save
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark DistributedLLM system.")
    parser.add_argument("--mode", choices=["local", "distributed", "sharding"], required=True,
                       help="Benchmark mode")
    parser.add_argument("--model", default="llama-2-7b",
                       help="Model ID to benchmark")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of benchmark iterations")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--workers", type=int, default=2,
                       help="Number of worker processes for distributed mode")
    parser.add_argument("--output", default="benchmark_results.json",
                       help="Output file for benchmark results")
    
    args = parser.parse_args()
    
    # Run benchmark based on mode
    if args.mode == "local":
        results = benchmark_local_inference(
            model_id=args.model,
            num_iterations=args.iterations,
            batch_size=args.batch_size
        )
    
    elif args.mode == "distributed":
        results = benchmark_distributed_inference(
            model_id=args.model,
            num_workers=args.workers,
            num_iterations=args.iterations,
            batch_size=args.batch_size
        )
    
    elif args.mode == "sharding":
        # Define sharding configurations to test
        shard_configs = [
            {"device_map": "auto"},
            {"device_map": "balanced"},
            {"device_map": {"embedding": "cuda:0", "transformer.blocks.0": "cuda:0", "transformer.blocks.1": "cuda:0",
                            "transformer.blocks.2": "cuda:1", "transformer.blocks.3": "cuda:1", "lm_head": "cuda:1"}}
        ]
        
        results = benchmark_model_sharding(
            model_id=args.model,
            shard_config=shard_configs,
            num_iterations=args.iterations
        )
    
    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
```

# scripts/setup_coordinator.sh

```sh
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
```

# scripts/setup_worker_linux.sh

```sh
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
```

# scripts/setup_worker_mac.sh

```sh
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
```

# scripts/setup_worker_windows.bat

```bat
@echo off
REM Setup script for DistributedLLM worker node on Windows

echo Setting up DistributedLLM worker node for Windows...

REM Get the project root directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
cd %PROJECT_ROOT%

echo Project root: %PROJECT_ROOT%

REM Create necessary directories
echo Creating directories...
if not exist logs mkdir logs
if not exist models\cache mkdir models\cache
if not exist data mkdir data

REM Check Python version (require 3.9+)
echo Checking Python version...
python --version > temp.txt 2>&1
set /p python_version=<temp.txt
del temp.txt

echo Found %python_version%

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Please ensure Python 3.9+ is installed.
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Check for CUDA support
echo Checking for GPU support...
python -c "import torch; print(torch.cuda.is_available())" > temp.txt 2>&1
set /p has_cuda=<temp.txt
del temp.txt

if "%has_cuda%"=="True" (
    echo GPU support detected.
    set HAS_GPU=true
    
    REM Get GPU info
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
) else (
    echo No GPU support detected. Performance may be limited.
    set HAS_GPU=false
)

REM Check network connectivity
echo Checking network connectivity...
ipconfig | findstr IPv4 || echo Could not determine IP address.

REM Create a default worker config if not exists
if not exist config\worker_config.yaml (
    echo Creating default worker configuration...
    
    REM Get local IP address (this is a basic approach that may not work in all environments)
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
        set local_ip=%%a
        set local_ip=!local_ip:~1!
        goto :ip_found
    )
    :ip_found
    
    REM Create config directory if it doesn't exist
    if not exist config mkdir config
    
    REM Generate a unique worker ID
    for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value') do set datetime=%%a
    set worker_id=worker_win_%COMPUTERNAME%_%datetime:~0,14%
    
    REM Create default worker configuration
    echo # DistributedLLM Worker Configuration> config\worker_config.yaml
    echo.>> config\worker_config.yaml
    echo # Worker identification>> config\worker_config.yaml
    echo worker:>> config\worker_config.yaml
    echo   id: "%worker_id%">> config\worker_config.yaml
    echo   host: "%local_ip%">> config\worker_config.yaml
    echo   port: 5556>> config\worker_config.yaml
    echo.>> config\worker_config.yaml
    echo # Coordinator connection>> config\worker_config.yaml
    echo coordinator:>> config\worker_config.yaml
    echo   host: "192.168.1.100"  # CHANGE THIS to your coordinator's IP>> config\worker_config.yaml
    echo   port: 5555>> config\worker_config.yaml
    echo.>> config\worker_config.yaml
    echo # Resource management>> config\worker_config.yaml
    echo resources:>> config\worker_config.yaml
    echo   max_memory_percent: 80>> config\worker_config.yaml
    echo   max_cpu_percent: 90>> config\worker_config.yaml
    echo   gpu_available: %HAS_GPU%>> config\worker_config.yaml
    
    echo Default worker configuration created. Please edit config\worker_config.yaml to set the correct coordinator IP.
)

REM Create Windows service (optional)
echo If you want to run as a Windows service, you can use NSSM:
echo 1. Download NSSM from https://nssm.cc/
echo 2. Run: nssm install DistributedLLMWorker
echo 3. Set the path to: %PROJECT_ROOT%\venv\Scripts\python.exe
echo 4. Set the arguments to: %PROJECT_ROOT%\src\main.py --mode worker
echo 5. Configure any additional settings as needed

REM Display setup completion message
echo.
echo Worker setup complete.
echo.
echo Before starting the worker, make sure to:
echo 1. Update the coordinator IP in config\worker_config.yaml
echo 2. Ensure the coordinator node is running
echo.
echo To start the worker, run:
echo venv\Scripts\activate
echo python src\main.py --mode worker
echo.
echo If you want to enable auto-discovery, run:
echo python src\main.py --mode worker --discover

REM Return to the original directory
cd %OLDPWD%
```

# src/coordinator/__init__.py

```py
"""
Coordinator components for the DistributedLLM system.

This package contains components for the coordinator node in the distributed system,
including the scheduler, performance monitor, and result aggregator.
"""

from src.coordinator.scheduler import Scheduler, Task, Worker
from src.coordinator.performance_monitor import PerformanceMonitor
from src.coordinator.result_aggregator import ResultAggregator

__all__ = [
    "Scheduler",
    "Task",
    "Worker",
    "PerformanceMonitor",
    "ResultAggregator"
]
```

# src/coordinator/performance_monitor.py

```py
"""
Performance Monitor for the DistributedLLM system.

Tracks worker performance, resource utilization, and system health metrics
to enable intelligent workload balancing and optimization.
"""

import logging
import time
import threading
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors performance metrics across workers in the distributed system.
    Used for making workload balancing decisions and identifying bottlenecks.
    """
    
    def __init__(self, metrics_history_size: int = 100):
        """
        Initialize the performance monitor.
        
        Args:
            metrics_history_size: Number of metrics points to keep in history
        """
        self.metrics_history_size = metrics_history_size
        self.worker_metrics = defaultdict(lambda: defaultdict(deque))
        self.task_metrics = defaultdict(lambda: defaultdict(deque))
        self.system_metrics = defaultdict(deque)
        self.lock = threading.RLock()
        
        # Track task completions by worker
        self.task_completions = defaultdict(list)
        
        # Performance profiles for different task types
        self.task_type_profiles = defaultdict(dict)
        
        # Record when monitoring started
        self.start_time = time.time()
    
    def update_worker_metrics(self, worker_id: str, metrics: Dict[str, Any]):
        """
        Update metrics for a specific worker.
        
        Args:
            worker_id: ID of the worker
            metrics: Dictionary of metrics
        """
        with self.lock:
            timestamp = time.time()
            
            for metric_name, metric_value in metrics.items():
                # Only store numeric metrics
                if isinstance(metric_value, (int, float)):
                    self.worker_metrics[worker_id][metric_name].append((timestamp, metric_value))
                    
                    # Trim history if needed
                    if len(self.worker_metrics[worker_id][metric_name]) > self.metrics_history_size:
                        self.worker_metrics[worker_id][metric_name].popleft()
    
    def record_task_completion(self, worker_id: str, task_id: str, metrics: Dict[str, Any]):
        """
        Record metrics for a completed task.
        
        Args:
            worker_id: ID of the worker that completed the task
            task_id: ID of the completed task
            metrics: Dictionary of metrics for the task
        """
        with self.lock:
            timestamp = time.time()
            task_type = metrics.get('task_type', 'unknown')
            
            # Store task completion record
            self.task_completions[worker_id].append({
                'task_id': task_id,
                'task_type': task_type,
                'timestamp': timestamp,
                'metrics': metrics
            })
            
            # Trim history if needed
            if len(self.task_completions[worker_id]) > self.metrics_history_size:
                self.task_completions[worker_id].pop(0)
            
            # Update task type metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.task_metrics[task_type][metric_name].append((timestamp, metric_value))
                    
                    # Trim history if needed
                    if len(self.task_metrics[task_type][metric_name]) > self.metrics_history_size:
                        self.task_metrics[task_type][metric_name].popleft()
            
            # Update task type performance profile
            self._update_task_profile(task_type, worker_id, metrics)
    
    def _update_task_profile(self, task_type: str, worker_id: str, metrics: Dict[str, Any]):
        """
        Update performance profile for a specific task type.
        
        Args:
            task_type: Type of the task
            worker_id: ID of the worker that completed the task
            metrics: Metrics from the task completion
        """
        if 'elapsed_time' not in metrics and 'execution_time' not in metrics:
            return
        
        execution_time = metrics.get('execution_time', metrics.get('elapsed_time'))
        
        # Initialize profile for this task type and worker if needed
        if worker_id not in self.task_type_profiles[task_type]:
            self.task_type_profiles[task_type][worker_id] = {
                'count': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'recent_times': deque(maxlen=10)  # Keep last 10 times
            }
        
        profile = self.task_type_profiles[task_type][worker_id]
        
        # Update profile
        profile['count'] += 1
        profile['total_time'] += execution_time
        profile['min_time'] = min(profile['min_time'], execution_time)
        profile['max_time'] = max(profile['max_time'], execution_time)
        profile['recent_times'].append(execution_time)
        
        # Calculate rolling average
        profile['avg_time'] = profile['total_time'] / profile['count']
        profile['recent_avg_time'] = sum(profile['recent_times']) / len(profile['recent_times'])
    
    def get_worker_performance(self, worker_id: str, task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a specific worker.
        
        Args:
            worker_id: ID of the worker
            task_type: Optional task type to filter by
        
        Returns:
            Dictionary of performance metrics
        """
        with self.lock:
            performance = {
                'task_count': 0,
                'avg_execution_time': 0,
                'recent_execution_time': 0,
                'task_types': {}
            }
            
            # Get task completions for this worker
            completions = self.task_completions.get(worker_id, [])
            
            if not completions:
                return performance
            
            # Filter by task type if specified
            if task_type:
                completions = [c for c in completions if c['task_type'] == task_type]
            
            if not completions:
                return performance
            
            # Calculate overall metrics
            performance['task_count'] = len(completions)
            
            # Calculate execution times
            execution_times = []
            for completion in completions:
                metrics = completion['metrics']
                if 'execution_time' in metrics:
                    execution_times.append(metrics['execution_time'])
                elif 'elapsed_time' in metrics:
                    execution_times.append(metrics['elapsed_time'])
            
            if execution_times:
                performance['avg_execution_time'] = sum(execution_times) / len(execution_times)
                performance['recent_execution_time'] = sum(execution_times[-5:]) / min(len(execution_times), 5)
            
            # Calculate task type breakdown
            task_type_counts = defaultdict(int)
            for completion in completions:
                task_type_counts[completion['task_type']] += 1
            
            for t_type, count in task_type_counts.items():
                performance['task_types'][t_type] = {
                    'count': count,
                    'percentage': (count / performance['task_count']) * 100
                }
                
                # Add detailed profile if available
                if t_type in self.task_type_profiles and worker_id in self.task_type_profiles[t_type]:
                    performance['task_types'][t_type].update(self.task_type_profiles[t_type][worker_id])
            
            return performance
    
    def get_task_type_performance(self, task_type: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific task type across all workers.
        
        Args:
            task_type: Type of task to get metrics for
        
        Returns:
            Dictionary of performance metrics for the task type
        """
        with self.lock:
            performance = {
                'worker_count': 0,
                'global_avg_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'workers': {}
            }
            
            if task_type not in self.task_type_profiles:
                return performance
            
            profiles = self.task_type_profiles[task_type]
            performance['worker_count'] = len(profiles)
            
            # Calculate global stats
            total_count = sum(profile['count'] for profile in profiles.values())
            total_time = sum(profile['total_time'] for profile in profiles.values())
            
            if total_count > 0:
                performance['global_avg_time'] = total_time / total_count
            
            # Find min and max times
            min_times = [profile['min_time'] for profile in profiles.values() if profile['min_time'] < float('inf')]
            max_times = [profile['max_time'] for profile in profiles.values()]
            
            if min_times:
                performance['min_time'] = min(min_times)
            if max_times:
                performance['max_time'] = max(max_times)
            
            # Add per-worker performance
            for worker_id, profile in profiles.items():
                performance['workers'][worker_id] = {
                    'count': profile['count'],
                    'avg_time': profile['avg_time'],
                    'recent_avg_time': profile.get('recent_avg_time', profile['avg_time']),
                    'min_time': profile['min_time'],
                    'max_time': profile['max_time'],
                    'efficiency': performance['global_avg_time'] / profile['avg_time'] if profile['avg_time'] > 0 else 0
                }
            
            return performance
    
    def get_worker_utilization(self, worker_id: str) -> Dict[str, Any]:
        """
        Get resource utilization metrics for a specific worker.
        
        Args:
            worker_id: ID of the worker
        
        Returns:
            Dictionary of utilization metrics
        """
        with self.lock:
            utilization = {
                'cpu_percent': None,
                'memory_percent': None,
                'gpu_percent': None,
                'idle_percent': None,
                'trends': {}
            }
            
            if worker_id not in self.worker_metrics:
                return utilization
            
            metrics = self.worker_metrics[worker_id]
            
            # Get latest values for key metrics
            for metric_name in ['cpu_percent', 'memory_percent', 'gpu_percent']:
                if metric_name in metrics and metrics[metric_name]:
                    utilization[metric_name] = metrics[metric_name][-1][1]
            
            # Calculate idle time percentage
            busy_periods = self._calculate_busy_periods(worker_id)
            if busy_periods:
                total_time = time.time() - self.start_time
                busy_time = sum(end - start for start, end in busy_periods)
                utilization['idle_percent'] = 100 - (busy_time / total_time) * 100
            
            # Calculate trends for key metrics
            for metric_name in ['cpu_percent', 'memory_percent', 'gpu_percent']:
                if metric_name in metrics and len(metrics[metric_name]) >= 2:
                    history = metrics[metric_name]
                    
                    # Calculate short-term trend (last 10 points)
                    short_term = history[-min(10, len(history)):]
                    if len(short_term) >= 2:
                        values = [v for _, v in short_term]
                        utilization['trends'][f'{metric_name}_short_term'] = (values[-1] - values[0]) / len(values)
                    
                    # Calculate long-term trend (all points)
                    if len(history) >= 5:
                        values = [v for _, v in history]
                        utilization['trends'][f'{metric_name}_long_term'] = (values[-1] - values[0]) / len(values)
            
            return utilization
    
    def _calculate_busy_periods(self, worker_id: str) -> List[Tuple[float, float]]:
        """
        Calculate periods when a worker was busy.
        
        Args:
            worker_id: ID of the worker
        
        Returns:
            List of (start_time, end_time) tuples representing busy periods
        """
        completions = self.task_completions.get(worker_id, [])
        if not completions:
            return []
        
        # Sort completions by timestamp
        sorted_completions = sorted(completions, key=lambda c: c['timestamp'])
        
        # Extract start and end times from task metrics
        periods = []
        for completion in sorted_completions:
            metrics = completion['metrics']
            timestamp = completion['timestamp']
            
            if 'execution_time' in metrics:
                start_time = timestamp - metrics['execution_time']
                end_time = timestamp
                periods.append((start_time, end_time))
        
        # Merge overlapping periods
        if not periods:
            return []
        
        periods.sort()
        merged = [periods[0]]
        
        for current_start, current_end in periods[1:]:
            prev_start, prev_end = merged[-1]
            
            if current_start <= prev_end:
                # Periods overlap, merge them
                merged[-1] = (prev_start, max(prev_end, current_end))
            else:
                # No overlap, add as new period
                merged.append((current_start, current_end))
        
        return merged
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics for the entire system.
        
        Returns:
            Dictionary of system-wide metrics
        """
        with self.lock:
            metrics = {
                'worker_metrics': {},
                'task_type_metrics': {},
                'system_metrics': {
                    'total_tasks_completed': 0,
                    'tasks_per_second': 0,
                    'worker_count': len(self.worker_metrics)
                }
            }
            
            # Collect worker metrics
            for worker_id in self.worker_metrics:
                metrics['worker_metrics'][worker_id] = {
                    'utilization': self.get_worker_utilization(worker_id),
                    'performance': self.get_worker_performance(worker_id)
                }
            
            # Collect task type metrics
            for task_type in self.task_type_profiles:
                metrics['task_type_metrics'][task_type] = self.get_task_type_performance(task_type)
            
            # Calculate system-wide metrics
            total_tasks = sum(len(completions) for completions in self.task_completions.values())
            metrics['system_metrics']['total_tasks_completed'] = total_tasks
            
            # Calculate tasks per second
            elapsed_time = max(time.time() - self.start_time, 1)
            metrics['system_metrics']['tasks_per_second'] = total_tasks / elapsed_time
            
            # Calculate load balance metrics
            if self.worker_metrics:
                cpu_utilizations = []
                for worker_id in self.worker_metrics:
                    util = self.get_worker_utilization(worker_id)
                    if util['cpu_percent'] is not None:
                        cpu_utilizations.append(util['cpu_percent'])
                
                if cpu_utilizations:
                    metrics['system_metrics']['avg_cpu_utilization'] = sum(cpu_utilizations) / len(cpu_utilizations)
                    metrics['system_metrics']['cpu_utilization_stddev'] = float(np.std(cpu_utilizations)) if len(cpu_utilizations) > 1 else 0
                    metrics['system_metrics']['load_balance_index'] = metrics['system_metrics']['cpu_utilization_stddev'] / max(metrics['system_metrics']['avg_cpu_utilization'], 1) if metrics['system_metrics']['avg_cpu_utilization'] > 0 else 0
            
            return metrics
    
    def should_rebalance(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if workload rebalancing is needed.
        
        Returns:
            Tuple of (should_rebalance, reason_dict)
        """
        metrics = self.get_current_metrics()
        
        # Default to not rebalancing
        should_rebalance = False
        reason = {'reason': 'no_rebalance_needed', 'details': {}}
        
        # Check CPU utilization imbalance
        if 'avg_cpu_utilization' in metrics['system_metrics'] and 'load_balance_index' in metrics['system_metrics']:
            lbi = metrics['system_metrics']['load_balance_index']
            avg_util = metrics['system_metrics']['avg_cpu_utilization']
            
            # If LBI is high (> 0.25) and average utilization is significant (> 30%)
            if lbi > 0.25 and avg_util > 30:
                should_rebalance = True
                reason = {
                    'reason': 'cpu_utilization_imbalance',
                    'details': {
                        'load_balance_index': lbi,
                        'avg_cpu_utilization': avg_util
                    }
                }
        
        # Check task execution time imbalance
        if not should_rebalance and len(self.task_type_profiles) > 0:
            for task_type, profiles in self.task_type_profiles.items():
                if len(profiles) < 2:
                    continue
                
                perf = self.get_task_type_performance(task_type)
                
                # Check for significant performance difference between workers
                worker_times = [w['avg_time'] for w in perf['workers'].values() if w['count'] > 2]
                if len(worker_times) < 2:
                    continue
                
                max_time = max(worker_times)
                min_time = min(worker_times)
                
                # If fastest worker is at least 1.5x faster than slowest
                if min_time > 0 and max_time / min_time > 1.5:
                    should_rebalance = True
                    reason = {
                        'reason': 'task_execution_time_imbalance',
                        'details': {
                            'task_type': task_type,
                            'max_time': max_time,
                            'min_time': min_time,
                            'ratio': max_time / min_time
                        }
                    }
                    break
        
        return should_rebalance, reason
    
    def suggest_workload_distribution(self) -> Dict[str, float]:
        """
        Suggest workload distribution among workers based on performance.
        
        Returns:
            Dictionary mapping worker IDs to workload fractions (0-1)
        """
        with self.lock:
            distribution = {}
            
            # If we have performance profiles, use them
            if self.task_type_profiles:
                # Average across all task types, weighted by count
                worker_speeds = defaultdict(float)
                total_tasks = 0
                
                for task_type, profiles in self.task_type_profiles.items():
                    type_total_tasks = sum(profile['count'] for profile in profiles.values())
                    total_tasks += type_total_tasks
                    
                    for worker_id, profile in profiles.items():
                        if profile['avg_time'] > 0:
                            # Speed is inverse of time (faster workers have higher speed)
                            speed = 1.0 / profile['avg_time']
                            worker_speeds[worker_id] += speed * profile['count']
                
                # Normalize to fractions
                total_speed = sum(worker_speeds.values())
                if total_speed > 0:
                    for worker_id, speed in worker_speeds.items():
                        distribution[worker_id] = speed / total_speed
                
                # If we have workers with no tasks yet, give them a small share
                for worker_id in self.worker_metrics:
                    if worker_id not in distribution:
                        # Give them a small fraction (10% of an equal share)
                        equal_share = 1.0 / len(self.worker_metrics)
                        distribution[worker_id] = equal_share * 0.1
                
                # Renormalize
                total = sum(distribution.values())
                if total > 0:
                    for worker_id in distribution:
                        distribution[worker_id] /= total
            
            # If no performance data, distribute equally
            if not distribution:
                worker_count = len(self.worker_metrics)
                if worker_count > 0:
                    equal_share = 1.0 / worker_count
                    for worker_id in self.worker_metrics:
                        distribution[worker_id] = equal_share
            
            return distribution
```

# src/coordinator/result_aggregator.py

```py
"""
Result Aggregator for the DistributedLLM system.

The ResultAggregator collects and combines model outputs from different workers,
handling the reassembly of sharded model outputs during distributed inference.
"""

import logging
import numpy as np
import threading
import time
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Collects and combines results from distributed computation tasks.
    Handles reassembly of model outputs that were sharded across workers.
    """
    
    def __init__(self):
        """Initialize the result aggregator."""
        self.results = {}
        self.task_groups = defaultdict(set)  # Group ID -> set of task IDs
        self.group_results = {}  # Group ID -> aggregated results
        self.group_status = {}  # Group ID -> status information
        self.lock = threading.RLock()  # Reentrant lock for thread safety
    
    def add_result(self, task):
        """
        Add a completed task result to the aggregator.
        
        Args:
            task: Completed task object
        """
        with self.lock:
            # Store individual task result
            self.results[task.id] = {
                'type': task.type,
                'result': task.result,
                'parameters': task.parameters,
                'completed_at': task.completed_at
            }
            
            # Check if this task belongs to a group
            group_id = task.parameters.get('group_id')
            if group_id:
                self.task_groups[group_id].add(task.id)
                
                # Check if all tasks in the group are complete
                if self._is_group_complete(group_id):
                    self._aggregate_group_results(group_id)
    
    def get_result(self, task_id):
        """
        Get the result for a specific task.
        
        Args:
            task_id: ID of the task
        
        Returns:
            Result data if available, None otherwise
        """
        with self.lock:
            if task_id in self.results:
                return self.results[task_id]['result']
            return None
    
    def get_group_result(self, group_id):
        """
        Get the aggregated result for a group of tasks.
        
        Args:
            group_id: ID of the task group
        
        Returns:
            Aggregated result if available, None otherwise
        """
        with self.lock:
            if group_id in self.group_results:
                return self.group_results[group_id]
            
            # If the group exists but results aren't aggregated yet,
            # check if we can aggregate now
            if group_id in self.task_groups:
                if self._is_group_complete(group_id):
                    self._aggregate_group_results(group_id)
                    return self.group_results.get(group_id)
            
            return None
    
    def get_group_status(self, group_id):
        """
        Get the status of a task group.
        
        Args:
            group_id: ID of the task group
        
        Returns:
            Status information including completion percentage
        """
        with self.lock:
            if group_id in self.group_status:
                return self.group_status[group_id]
            
            if group_id in self.task_groups:
                total_tasks = len(self.task_groups[group_id])
                completed_tasks = sum(1 for task_id in self.task_groups[group_id] if task_id in self.results)
                
                status = {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'percent_complete': (completed_tasks / max(total_tasks, 1)) * 100,
                    'is_complete': completed_tasks == total_tasks
                }
                
                self.group_status[group_id] = status
                return status
            
            return None
    
    def _is_group_complete(self, group_id):
        """
        Check if all tasks in a group are complete.
        
        Args:
            group_id: ID of the task group
        
        Returns:
            True if all tasks are complete, False otherwise
        """
        if group_id not in self.task_groups:
            return False
        
        return all(task_id in self.results for task_id in self.task_groups[group_id])
    
    def _aggregate_group_results(self, group_id):
        """
        Aggregate results for a completed task group.
        
        Args:
            group_id: ID of the task group
        """
        if not self._is_group_complete(group_id):
            return
        
        task_ids = self.task_groups[group_id]
        if not task_ids:
            return
        
        # Get the first task to determine the type
        first_task_id = next(iter(task_ids))
        task_type = self.results[first_task_id]['type']
        
        # Use different aggregation strategies based on task type
        if task_type == 'layer_computation':
            self._aggregate_layer_computation(group_id, task_ids)
        elif task_type == 'token_generation':
            self._aggregate_token_generation(group_id, task_ids)
        elif task_type == 'embedding':
            self._aggregate_embedding(group_id, task_ids)
        else:
            # Default aggregation: just collect all results in a list
            self.group_results[group_id] = {
                'type': task_type,
                'results': [self.results[task_id]['result'] for task_id in task_ids]
            }
        
        # Update group status
        self.group_status[group_id] = {
            'total_tasks': len(task_ids),
            'completed_tasks': len(task_ids),
            'percent_complete': 100.0,
            'is_complete': True,
            'aggregated': True
        }
        
        logger.info(f"Aggregated results for task group {group_id}")
    
    def _aggregate_layer_computation(self, group_id, task_ids):
        """
        Aggregate layer computation results.
        
        Args:
            group_id: ID of the task group
            task_ids: List of task IDs in the group
        """
        # Sort tasks by layer index
        tasks_by_layer = {}
        for task_id in task_ids:
            layer_index = self.results[task_id]['parameters'].get('layer_index', 0)
            tasks_by_layer[layer_index] = task_id
        
        # Collect outputs in order
        layer_outputs = []
        for layer_idx in sorted(tasks_by_layer.keys()):
            task_id = tasks_by_layer[layer_idx]
            layer_outputs.append({
                'layer_index': layer_idx,
                'output': self.results[task_id]['result']['output_data']
            })
        
        # Store aggregated result
        self.group_results[group_id] = {
            'type': 'layer_computation',
            'layer_outputs': layer_outputs,
            'aggregation_method': 'layer_sequence'
        }
    
    def _aggregate_token_generation(self, group_id, task_ids):
        """
        Aggregate token generation results.
        
        Args:
            group_id: ID of the task group
            task_ids: List of task IDs in the group
        """
        # For token generation, we need to concatenate in the right order
        # Get sequence indices
        tasks_by_sequence = {}
        for task_id in task_ids:
            seq_index = self.results[task_id]['parameters'].get('sequence_index', 0)
            tasks_by_sequence[seq_index] = task_id
        
        # Concatenate token sequences
        all_tokens = []
        for seq_idx in sorted(tasks_by_sequence.keys()):
            task_id = tasks_by_sequence[seq_idx]
            output_ids = self.results[task_id]['result']['output_ids']
            all_tokens.extend(output_ids)
        
        # Calculate total generation time
        total_time = sum(
            self.results[task_id]['result'].get('generation_time', 0)
            for task_id in task_ids
        )
        
        # Store aggregated result
        self.group_results[group_id] = {
            'type': 'token_generation',
            'output_ids': all_tokens,
            'total_generation_time': total_time,
            'aggregation_method': 'sequence_concatenation'
        }
    
    def _aggregate_embedding(self, group_id, task_ids):
        """
        Aggregate embedding results.
        
        Args:
            group_id: ID of the task group
            task_ids: List of task IDs in the group
        """
        # For embeddings, we typically average or concatenate
        aggregation_method = self.results[next(iter(task_ids))]['parameters'].get('aggregation_method', 'average')
        
        all_embeddings = [
            self.results[task_id]['result']['embeddings']
            for task_id in task_ids
        ]
        
        if aggregation_method == 'average':
            # Convert to numpy arrays and average
            try:
                np_embeddings = [np.array(emb) for emb in all_embeddings]
                aggregated_embedding = np.mean(np_embeddings, axis=0).tolist()
            except:
                # Fallback if numpy conversion fails
                aggregated_embedding = all_embeddings[0]
                logger.warning(f"Failed to average embeddings for group {group_id}")
        
        elif aggregation_method == 'concatenate':
            # Flatten and concatenate all embeddings
            aggregated_embedding = []
            for emb in all_embeddings:
                if isinstance(emb, list):
                    aggregated_embedding.extend(emb)
                else:
                    aggregated_embedding.append(emb)
        
        else:
            # Default to returning all embeddings
            aggregated_embedding = all_embeddings
        
        # Store aggregated result
        self.group_results[group_id] = {
            'type': 'embedding',
            'embeddings': aggregated_embedding,
            'aggregation_method': aggregation_method
        }
    
    def clean_old_results(self, max_age_seconds=3600):
        """
        Clean up old results to prevent memory leaks.
        
        Args:
            max_age_seconds: Maximum age of results to keep
        """
        with self.lock:
            current_time = time.time()
            
            # Clean individual task results
            task_ids_to_remove = []
            for task_id, result in self.results.items():
                if current_time - result.get('completed_at', 0) > max_age_seconds:
                    task_ids_to_remove.append(task_id)
            
            for task_id in task_ids_to_remove:
                del self.results[task_id]
            
            # Clean group results and status
            group_ids_to_remove = []
            for group_id in self.group_results:
                if not any(group_id in task_group for task_group in self.task_groups.values()):
                    group_ids_to_remove.append(group_id)
            
            for group_id in group_ids_to_remove:
                if group_id in self.group_results:
                    del self.group_results[group_id]
                if group_id in self.group_status:
                    del self.group_status[group_id]
            
            # Clean empty task groups
            group_ids_to_remove = []
            for group_id, task_ids in self.task_groups.items():
                if not task_ids:
                    group_ids_to_remove.append(group_id)
            
            for group_id in group_ids_to_remove:
                del self.task_groups[group_id]
            
            logger.info(f"Cleaned {len(task_ids_to_remove)} old results")
```

# src/coordinator/scheduler.py

```py
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
```

# src/main.py

```py
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
```

# src/model/__init__.py

```py
"""
Model components for DistributedLLM.

This package contains implementations of model architectures, tokenizers,
and inference utilities for large language models.
"""

from src.model.layers import ShardedModel, ShardedModelLoader
from src.model.tokenizer import Tokenizer
from src.model.inference import ModelInference

__all__ = [
    "ShardedModel",
    "ShardedModelLoader",
    "Tokenizer",
    "ModelInference",
]
```

# src/model/inference.py

```py
"""
Inference utilities for DistributedLLM.

Provides functionality for running inference on language models
in a distributed manner across multiple devices and machines.
"""

import logging
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
import numpy as np

from src.model.layers import ShardedModel, ShardedModelLoader
from src.model.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class ModelInference:
    """
    Manages distributed inference for large language models.
    
    This class handles:
    1. Token generation across distributed model components
    2. Batching and sequencing of requests
    3. Memory management and caching
    4. Coordination of multi-device inference
    """
    
    def __init__(
        self,
        model_id: str,
        device_map: Optional[Union[str, Dict[str, str]]] = "auto",
        cache_dir: Optional[str] = None,
        max_batch_size: int = 8,
        max_sequence_length: int = 2048,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the inference manager.
        
        Args:
            model_id: ID of the model to load
            device_map: How to distribute model components across devices
            cache_dir: Directory for caching model weights
            max_batch_size: Maximum batch size for inference
            max_sequence_length: Maximum sequence length for generation
            dtype: Data type to use for model weights and activation
        """
        self.model_id = model_id
        self.device_map = device_map
        self.cache_dir = cache_dir
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        
        # Load model
        logger.info(f"Loading model {model_id} for inference...")
        self.model = self._load_model()
        
        # Load tokenizer
        self.tokenizer = Tokenizer(model_id, cache_dir=cache_dir)
        
        # Inference settings
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "max_new_tokens": 256,
        }
        
        # KV-cache for efficient generation
        self.kv_cache = {}
        
        # Statistics and monitoring
        self.inference_stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_inference_time": 0,
            "tokens_per_second": 0,
        }
    
    def _load_model(self) -> ShardedModel:
        """Load the model with the specified device mapping."""
        # Get model configuration and structure
        model_info = ShardedModelLoader.load_model(self.model_id, self.device_map)
        
        # Create the actual ShardedModel instance
        model = ShardedModel(model_info["config"], model_info["device_map"])
        
        # Set model to eval mode
        model.eval()
        
        return model
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate text completions based on the input prompt.
        
        Args:
            prompt: Input prompt text or list of prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for token repetition
            do_sample: Whether to use sampling or greedy decoding
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated text completions
        """
        # Update generation config with any provided parameters
        generation_config = self.generation_config.copy()
        if max_new_tokens is not None:
            generation_config["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            generation_config["temperature"] = temperature
        if top_p is not None:
            generation_config["top_p"] = top_p
        if top_k is not None:
            generation_config["top_k"] = top_k
        if repetition_penalty is not None:
            generation_config["repetition_penalty"] = repetition_penalty
        if do_sample is not None:
            generation_config["do_sample"] = do_sample
        
        # Convert single prompt to list
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        
        # Tokenize prompts
        input_tokens = self.tokenizer.encode_batch(prompts)
        
        # Run generation
        start_time = time.time()
        
        # Generate token IDs
        generated_ids = self._generate_tokens(input_tokens, generation_config)
        
        # Decode generated tokens
        generated_texts = self.tokenizer.decode_batch(generated_ids)
        
        # Update statistics
        generation_time = time.time() - start_time
        num_tokens_generated = sum(len(ids) - len(input_ids) for ids, input_ids in zip(generated_ids, input_tokens))
        
        self.inference_stats["total_requests"] += len(prompts)
        self.inference_stats["total_tokens_generated"] += num_tokens_generated
        self.inference_stats["total_inference_time"] += generation_time
        self.inference_stats["tokens_per_second"] = num_tokens_generated / generation_time
        
        logger.info(f"Generated {num_tokens_generated} tokens in {generation_time:.2f}s ({num_tokens_generated / generation_time:.2f} tokens/sec)")
        
        return generated_texts
    
    def _generate_tokens(
        self,
        input_tokens: List[List[int]],
        generation_config: Dict[str, Any]
    ) -> List[List[int]]:
        """
        Generate token IDs based on input tokens.
        
        Args:
            input_tokens: Batch of input token IDs
            generation_config: Generation parameters
        
        Returns:
            List of generated token ID sequences
        """
        # Convert to tensors
        max_len = max(len(tokens) for tokens in input_tokens)
        padded_tokens = [tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens)) for tokens in input_tokens]
        input_ids = torch.tensor(padded_tokens, dtype=torch.long)
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        for i, tokens in enumerate(input_tokens):
            attention_mask[i, len(tokens):] = 0
        
        # Move to the first device in the model
        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)
        attention_mask = attention_mask.to(first_device)
        
        # Handle batching if needed
        batch_size = input_ids.shape[0]
        if batch_size > self.max_batch_size:
            logger.warning(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}. Splitting into multiple batches.")
            
            # Process batches and concatenate results
            all_generated_ids = []
            for i in range(0, batch_size, self.max_batch_size):
                batch_input_ids = input_ids[i:i+self.max_batch_size]
                batch_attention_mask = attention_mask[i:i+self.max_batch_size]
                batch_generated_ids = self._generate_tokens_single_batch(
                    batch_input_ids, batch_attention_mask, generation_config
                )
                all_generated_ids.extend(batch_generated_ids)
            return all_generated_ids
        else:
            # Process single batch
            return self._generate_tokens_single_batch(input_ids, attention_mask, generation_config)
    
    def _generate_tokens_single_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: Dict[str, Any]
    ) -> List[List[int]]:
        """
        Generate tokens for a single batch that fits in memory.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            generation_config: Generation parameters
        
        Returns:
            List of token ID sequences
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        max_length = seq_len + generation_config["max_new_tokens"]
        
        # Initialize storage for generated sequences
        generated_ids = input_ids.clone()
        
        # Initialize KV cache if enabled
        use_kv_cache = True  # Could be a config parameter
        
        # Generate tokens autoregressively
        for _ in range(generation_config["max_new_tokens"]):
            # Get predictions for next token
            with torch.no_grad():
                if use_kv_cache and generated_ids.shape[1] > 1:
                    # Use cached values for previous tokens
                    # In a real implementation, this would use a specific forward method
                    # that takes advantage of KV caching
                    logits = self.model(generated_ids[:, -1:], None)[:, -1, :]
                else:
                    # No cache or first token
                    logits = self.model(generated_ids, attention_mask)[:, -1, :]
            
            # Apply temperature
            if generation_config["temperature"] > 0:
                logits = logits / generation_config["temperature"]
            
            # Apply repetition penalty
            if generation_config["repetition_penalty"] > 1.0:
                # Create score tensor from logits
                scores = logits.clone()
                
                # Apply penalty to previously generated tokens
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        # If score > 0, reduce it; if score < 0, increase it
                        scores[i, token_id] /= generation_config["repetition_penalty"]
                
                logits = scores
            
            # Apply top-k filtering
            if generation_config["top_k"] > 0:
                indices_to_remove = logits < torch.topk(logits, generation_config["top_k"])[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if generation_config["top_p"] < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > generation_config["top_p"]
                
                # Shift indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted indices back to original order
                for i in range(batch_size):
                    indices_to_remove = sorted_indices_to_remove[i].scatter(
                        0, sorted_indices[i], sorted_indices_to_remove[i]
                    )
                    logits[i][indices_to_remove] = float('-inf')
            
            # Get next token
            if generation_config["do_sample"]:
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append new tokens
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
            ], dim=1)
            
            # Check for end of sequence tokens
            if all((next_tokens == self.tokenizer.eos_token_id).view(-1)):
                break
        
        # Convert to list of token IDs
        return [ids.tolist() for ids in generated_ids]
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the input text.
        
        Args:
            text: Input text or list of texts
        
        Returns:
            Embeddings as numpy array [batch_size, embedding_dim]
        """
        # Convert single text to list
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Tokenize text
        input_tokens = self.tokenizer.encode_batch(texts)
        
        # Convert to tensors
        max_len = max(len(tokens) for tokens in input_tokens)
        padded_tokens = [tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens)) for tokens in input_tokens]
        input_ids = torch.tensor(padded_tokens, dtype=torch.long)
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        for i, tokens in enumerate(input_tokens):
            attention_mask[i, len(tokens):] = 0
        
        # Move to the first device in the model
        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)
        attention_mask = attention_mask.to(first_device)
        
        # Forward pass through the model to get embeddings
        with torch.no_grad():
            # In a real implementation, we'd use a specific embedding method
            # Here we're just using the output of the final layer
            outputs = self.model(input_ids, attention_mask)
            
            # Get embeddings from the last token of each sequence
            embeddings = []
            for i, length in enumerate(map(len, input_tokens)):
                embeddings.append(outputs[i, length-1])
            
            # Stack embeddings
            embeddings = torch.stack(embeddings)
        
        # Convert to numpy
        return embeddings.cpu().numpy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return self.inference_stats
    
    def clear_cache(self):
        """Clear KV-cache and other temporary storage."""
        self.kv_cache = {}
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Cleared inference caches")
    
    def unload(self):
        """Unload the model from memory."""
        del self.model
        self.model = None
        
        # Clear caches
        self.clear_cache()
        
        # Run garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Unloaded model {self.model_id}")


class DistributedInference:
    """
    Manages inference across multiple worker nodes.
    
    This class coordinates token generation across multiple machines,
    each potentially running a shard of the model.
    """
    
    def __init__(
        self,
        coordinator_client,
        model_id: str,
        tokenizer = None,
        max_sequence_length: int = 2048,
    ):
        """
        Initialize the distributed inference manager.
        
        Args:
            coordinator_client: Client for communicating with coordinator
            model_id: ID of the model to use
            tokenizer: Optional tokenizer to use (will be loaded if not provided)
            max_sequence_length: Maximum sequence length for generation
        """
        self.coordinator_client = coordinator_client
        self.model_id = model_id
        self.max_sequence_length = max_sequence_length
        
        # Load tokenizer if not provided
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer(model_id)
        
        # Request queues
        self.request_queue = queue.Queue()
        self.result_queue = {}  # Map from request_id to result
        
        # Statistics and monitoring
        self.inference_stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_inference_time": 0,
            "tokens_per_second": 0,
        }
        
        # Start worker threads
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.worker_thread.start()
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text completions using distributed inference.
        
        Args:
            prompt: Input prompt text or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            top_k: Top-k sampling threshold
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text completion(s)
        """
        # Create a unique ID for this request
        request_id = f"req_{time.time()}_{hash(str(prompt))}"
        
        # Create result placeholder
        self.result_queue[request_id] = None
        
        # Prepare generation parameters
        params = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            **kwargs
        }
        
        # Add request to queue
        self.request_queue.put((request_id, params))
        
        # Wait for result
        while self.result_queue[request_id] is None:
            time.sleep(0.1)
            
            # Check if processing failed
            if request_id not in self.result_queue:
                raise RuntimeError("Request processing failed")
        
        # Get result
        result = self.result_queue[request_id]
        del self.result_queue[request_id]
        
        # Return single result if input was a single string
        if isinstance(prompt, str) and isinstance(result, list) and len(result) == 1:
            return result[0]
        
        return result
    
    def _process_requests(self):
        """Process requests from the queue and send to coordinator."""
        while self.running:
            try:
                # Get request from queue
                try:
                    request_id, params = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the request
                start_time = time.time()
                
                try:
                    # Tokenize input
                    prompt = params["prompt"]
                    is_batch = isinstance(prompt, list)
                    
                    if is_batch:
                        prompts = prompt
                    else:
                        prompts = [prompt]
                    
                    # Tokenize prompts
                    input_tokens = self.tokenizer.encode_batch(prompts)
                    
                    # Create generation tasks
                    tasks = []
                    for i, tokens in enumerate(input_tokens):
                        task = {
                            "type": "token_generation",
                            "input_ids": tokens,
                            "params": {
                                k: v for k, v in params.items() 
                                if k not in ["prompt"]
                            },
                            "sequence_index": i
                        }
                        tasks.append(task)
                    
                    # Submit tasks to coordinator
                    task_ids = self.coordinator_client.submit_tasks(tasks)
                    
                    # Wait for tasks to complete
                    results = self.coordinator_client.wait_for_tasks(task_ids)
                    
                    # Process results
                    output_ids = []
                    for result in sorted(results, key=lambda r: r["sequence_index"]):
                        output_ids.append(result["output_ids"])
                    
                    # Decode output tokens
                    generated_texts = self.tokenizer.decode_batch(output_ids)
                    
                    # Set result
                    if is_batch:
                        self.result_queue[request_id] = generated_texts
                    else:
                        self.result_queue[request_id] = generated_texts[0]
                    
                    # Update statistics
                    generation_time = time.time() - start_time
                    num_new_tokens = sum(len(out) - len(inp) for out, inp in zip(output_ids, input_tokens))
                    
                    self.inference_stats["total_requests"] += len(prompts)
                    self.inference_stats["total_tokens_generated"] += num_new_tokens
                    self.inference_stats["total_inference_time"] += generation_time
                    self.inference_stats["tokens_per_second"] = (
                        self.inference_stats["total_tokens_generated"] / 
                        self.inference_stats["total_inference_time"]
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing request {request_id}: {e}")
                    # Set error result
                    self.result_queue[request_id] = f"Error: {str(e)}"
                
                finally:
                    # Mark request as processed
                    self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in request processing thread: {e}")
                time.sleep(1)  # Avoid tight loop on persistent errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return self.inference_stats
    
    def stop(self):
        """Stop the inference manager."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
```

# src/model/layers.py

```py
"""
Model layer implementations for the DistributedLLM system.

Provides implementations for model layers that can be distributed and sharded
across multiple devices, inspired by HuggingFace's model sharding approach.
"""

import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ShardedModelLoader:
    """
    Utility for loading large language models in a sharded manner across devices.
    Implements model sharding strategies similar to HuggingFace's Accelerate.
    """
    
    @staticmethod
    def load_model(model_id: str, device_map: Union[str, Dict[str, str]]) -> Dict[str, Any]:
        """
        Load a model with the specified device mapping.
        
        Args:
            model_id: Identifier for the model to load
            device_map: How to distribute model across devices
                - "auto": Automatically place based on available memory
                - "balanced": Distribute evenly across available devices
                - Dict: Explicit mapping of layer names to devices
        
        Returns:
            Dictionary containing model components
        """
        logger.info(f"Loading model {model_id} with device_map={device_map}")
        
        # In a real implementation, this would use Transformers or a similar library
        # Here we'll just return a mock model structure
        
        # Simulate loading time
        time.sleep(2)
        
        # Create a mock model with different components
        model = {
            "config": {
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
            },
            "device_map": ShardedModelLoader._resolve_device_map(device_map),
            "components": {}
        }
        
        # Create mock components with their assigned devices
        components = [
            "embedding", "transformer.blocks.0", "transformer.blocks.1", 
            "transformer.blocks.2", "transformer.blocks.3", "transformer.blocks.4",
            "transformer.blocks.5", "transformer.blocks.6", "transformer.blocks.7",
            "transformer.blocks.8", "transformer.blocks.9", "lm_head"
        ]
        
        for component in components:
            device = ShardedModelLoader._get_component_device(component, model["device_map"])
            model["components"][component] = {
                "device": device,
                "loaded": True,
                "parameters": ShardedModelLoader._mock_parameters(component, model["config"])
            }
        
        logger.info(f"Model {model_id} loaded and sharded across devices")
        return model
    
    @staticmethod
    def _resolve_device_map(device_map: Union[str, Dict[str, str]]) -> Dict[str, str]:
        """
        Resolve the device map to a concrete mapping of components to devices.
        
        Args:
            device_map: Device mapping specification
        
        Returns:
            Concrete mapping of component names to device identifiers
        """
        # If already a dict, return as is
        if isinstance(device_map, dict):
            return device_map
        
        # Check available devices
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(device_count)]
        else:
            device_count = 0
            devices = []
        
        # Always add CPU as a fallback
        devices.append("cpu")
        
        # For "auto" strategy, distribute based on device memory
        if device_map == "auto":
            resolved_map = {}
            
            # In a real implementation, we would analyze memory requirements
            # and place components optimally
            
            # For this mock implementation, we'll place first half on GPU (if available)
            # and the rest on CPU
            if device_count > 0:
                components = ["embedding", "transformer.blocks.0", "transformer.blocks.1", 
                             "transformer.blocks.2", "transformer.blocks.3", "transformer.blocks.4",
                             "transformer.blocks.5", "transformer.blocks.6", "transformer.blocks.7",
                             "transformer.blocks.8", "transformer.blocks.9", "lm_head"]
                
                # Place first components on GPU
                gpu_components = components[:len(components) // 2]
                cpu_components = components[len(components) // 2:]
                
                for comp in gpu_components:
                    resolved_map[comp] = "cuda:0"
                
                for comp in cpu_components:
                    resolved_map[comp] = "cpu"
            else:
                # All on CPU
                resolved_map = {"*": "cpu"}
            
            return resolved_map
        
        # For "balanced" strategy, distribute evenly across devices
        elif device_map == "balanced":
            resolved_map = {}
            
            if device_count > 0:
                components = ["embedding", "transformer.blocks.0", "transformer.blocks.1", 
                             "transformer.blocks.2", "transformer.blocks.3", "transformer.blocks.4",
                             "transformer.blocks.5", "transformer.blocks.6", "transformer.blocks.7",
                             "transformer.blocks.8", "transformer.blocks.9", "lm_head"]
                
                # Distribute components evenly
                for i, comp in enumerate(components):
                    device_idx = i % device_count
                    resolved_map[comp] = f"cuda:{device_idx}"
            else:
                # All on CPU
                resolved_map = {"*": "cpu"}
            
            return resolved_map
        
        # Default to CPU for unknown strategies
        return {"*": "cpu"}
    
    @staticmethod
    def _get_component_device(component, device_map: Dict[str, str]) -> str:
        """
        Get the device for a specific component based on the device map.
        
        Args:
            component: Component name
            device_map: Device mapping
        
        Returns:
            Device identifier for the component
        """
        # Check for exact match
        if component in device_map:
            return device_map[component]
        
        # Check for prefix match
        for prefix, device in device_map.items():
            if prefix.endswith('.*') and component.startswith(prefix[:-2]):
                return device
        
        # Fall back to wildcard
        if "*" in device_map:
            return device_map["*"]
        
        # Default to CPU
        return "cpu"
    
    @staticmethod
    def _mock_parameters(component, config):
        """Create mock parameters for a component based on the config."""
        hidden_size = config["hidden_size"]
        
        if component == "embedding":
            return {
                "weight": torch.zeros(config["vocab_size"], hidden_size),
            }
        elif component.startswith("transformer.blocks"):
            return {
                "attention.q_proj.weight": torch.zeros(hidden_size, hidden_size),
                "attention.k_proj.weight": torch.zeros(hidden_size, hidden_size),
                "attention.v_proj.weight": torch.zeros(hidden_size, hidden_size),
                "attention.o_proj.weight": torch.zeros(hidden_size, hidden_size),
                "feed_forward.gate_proj.weight": torch.zeros(config["intermediate_size"], hidden_size),
                "feed_forward.up_proj.weight": torch.zeros(config["intermediate_size"], hidden_size),
                "feed_forward.down_proj.weight": torch.zeros(hidden_size, config["intermediate_size"]),
                "layer_norm.weight": torch.zeros(hidden_size),
            }
        elif component == "lm_head":
            return {
                "weight": torch.zeros(config["vocab_size"], hidden_size),
            }
        else:
            return {}


class ShardedAttention(nn.Module):
    """
    Attention mechanism that can be sharded across devices.
    Similar to multi-head attention but with support for device sharding.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        device_map: Optional[Dict[str, str]] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        
        # Initialize projection matrices
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # Apply device mapping if provided
        if device_map:
            self._apply_device_map(device_map)
    
    def _apply_device_map(self, device_map: Dict[str, str]):
        """Move projection matrices to specified devices."""
        for name, module in [
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("o_proj", self.o_proj)
        ]:
            device = ShardedModelLoader._get_component_device(f"attention.{name}", device_map)
            module.to(device)
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        """
        Forward pass for the attention mechanism.
        Handles cross-device communication if matrices are on different devices.
        """
        batch_size, seq_length = hidden_states.size()[:2]
        
        # Project queries, keys, and values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose for batched attention
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)  # [batch, heads, seq_len, head_dim]
        
        # Transpose and reshape
        context = context.transpose(1, 2).contiguous()  # [batch, seq_len, heads, head_dim]
        context = context.view(batch_size, seq_length, -1)  # [batch, seq_len, heads*head_dim]
        
        # Final projection
        output = self.o_proj(context)
        
        return output


class ShardedMLP(nn.Module):
    """
    Multi-layer perceptron that can be sharded across devices.
    Implements the feed-forward network in a transformer block.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device_map: Optional[Dict[str, str]] = None
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Apply device mapping if provided
        if device_map:
            self._apply_device_map(device_map)
    
    def _apply_device_map(self, device_map: Dict[str, str]):
        """Move matrices to specified devices."""
        for name, module in [
            ("gate_proj", self.gate_proj),
            ("up_proj", self.up_proj),
            ("down_proj", self.down_proj)
        ]:
            device = ShardedModelLoader._get_component_device(f"feed_forward.{name}", device_map)
            module.to(device)
    
    def forward(self, x):
        """Forward pass with support for cross-device computation."""
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        
        # Down projection
        output = self.down_proj(intermediate)
        
        return output


class ShardedTransformerBlock(nn.Module):
    """
    Transformer block that can have its components sharded across devices.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        layer_idx: int = 0,
        device_map: Optional[Dict[str, str]] = None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-attention layer norm
        self.input_layernorm = nn.LayerNorm(hidden_size)
        
        # Self-attention
        self.attention = ShardedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            device_map=device_map
        )
        
        # Post-attention layer norm
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.mlp = ShardedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device_map=device_map
        )
        
        # Apply device mapping for layer norms
        if device_map:
            self._apply_device_map(device_map)
    
    def _apply_device_map(self, device_map: Dict[str, str]):
        """Move layer norms to specified devices."""
        block_prefix = f"transformer.blocks.{self.layer_idx}"
        
        for name, module in [
            ("input_layernorm", self.input_layernorm),
            ("post_attention_layernorm", self.post_attention_layernorm)
        ]:
            device = ShardedModelLoader._get_component_device(f"{block_prefix}.{name}", device_map)
            module.to(device)
    
    def forward(self, hidden_states, attention_mask=None):
        """Forward pass with support for cross-device computation."""
        # Ensure inputs are on the correct device
        input_device = self.input_layernorm.weight.device
        if hidden_states.device != input_device:
            hidden_states = hidden_states.to(input_device)
        
        # Residual connection for attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Residual connection for MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class ShardedModel(nn.Module):
    """
    A complete model that can be sharded across multiple devices.
    Implements the core of a large language model.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device_map: Optional[Union[str, Dict[str, str]]] = None
    ):
        super().__init__()
        self.config = config
        
        # Resolve device map
        if isinstance(device_map, str):
            self.device_map = ShardedModelLoader._resolve_device_map(device_map)
        else:
            self.device_map = device_map or {"*": "cpu"}
        
        # Initialize model components
        self._init_components()
    
    def _init_components(self):
        """Initialize model components with device mapping."""
        config = self.config
        hidden_size = config["hidden_size"]
        
        # Token embedding
        self.embedding = nn.Embedding(config["vocab_size"], hidden_size)
        embedding_device = ShardedModelLoader._get_component_device("embedding", self.device_map)
        self.embedding.to(embedding_device)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ShardedTransformerBlock(
                hidden_size=hidden_size,
                num_heads=config["num_attention_heads"],
                intermediate_size=config["intermediate_size"],
                layer_idx=i,
                device_map=self.device_map
            )
            for i in range(config["num_hidden_layers"])
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size)
        norm_device = ShardedModelLoader._get_component_device("norm", self.device_map)
        self.norm.to(norm_device)
        
        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, config["vocab_size"], bias=False)
        lm_head_device = ShardedModelLoader._get_component_device("lm_head", self.device_map)
        self.lm_head.to(lm_head_device)
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the entire model, handling cross-device transfers."""
        # Get embeddings
        hidden_states = self.embedding(input_ids)
        
        # Process through transformer blocks
        for block in self.blocks:
            # Ensure hidden states are on the right device for this block
            block_device = block.input_layernorm.weight.device
            if hidden_states.device != block_device:
                hidden_states = hidden_states.to(block_device)
            
            hidden_states = block(hidden_states, attention_mask)
        
        # Final normalization
        norm_device = self.norm.weight.device
        if hidden_states.device != norm_device:
            hidden_states = hidden_states.to(norm_device)
        
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        lm_head_device = self.lm_head.weight.device
        if hidden_states.device != lm_head_device:
            hidden_states = hidden_states.to(lm_head_device)
        
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate(
        self,
        input_ids,
        max_length: int = 20,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ):
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
        
        Returns:
            Generated token IDs
        """
        # Start with the provided input ids
        current_ids = input_ids.clone()
        batch_size = current_ids.shape[0]
        
        # Generate tokens one by one
        for _ in range(max_length - current_ids.shape[1]):
            # Get attention mask for the current sequence
            attention_mask = torch.ones_like(current_ids)
            
            # Get predictions for the next token
            with torch.no_grad():
                logits = self.forward(current_ids, attention_mask)
                next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                # Get indices of top-k values
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                
                # Create a new distribution with only top-k logits
                next_token_logits = torch.zeros_like(next_token_logits)
                next_token_logits.scatter_(-1, top_k_indices, top_k_values)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                
                # Compute cumulative probabilities
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create a mask for indices to remove
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                
                # Apply the mask to the logits
                next_token_logits.masked_fill_(indices_to_remove, -float('inf'))
            
            # Get probabilities and sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the sequence
            current_ids = torch.cat([current_ids, next_token], dim=-1)
        
        return current_ids
```

# src/model/tokenizer.py

```py
"""
Tokenizer implementation for DistributedLLM.

Provides tokenization utilities for text processing in language models,
with support for various tokenization schemes.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

# Try to import the tokenizer libraries, with graceful fallbacks
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    logger.warning("SentencePiece not available. Some tokenizers may not work.")

try:
    import tokenizers
    from tokenizers import Tokenizer as HFTokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    logger.warning("Hugging Face tokenizers not available. Some tokenizers may not work.")


class Tokenizer:
    """
    Tokenizer for language models with support for various backends.
    
    This class provides a unified interface for different tokenization libraries,
    with support for batched operations and caching.
    """
    
    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        legacy_mode: bool = False,
    ):
        """
        Initialize the tokenizer for a specific model.
        
        Args:
            model_id: ID of the model to load the tokenizer for
            cache_dir: Directory to cache tokenizer files
            legacy_mode: Whether to use legacy tokenization mode
        """
        self.model_id = model_id
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "distributed_llm")
        self.legacy_mode = legacy_mode
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Default token IDs
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.unk_token_id = 3
        
        # Load tokenizer based on model ID
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the appropriate tokenizer based on model ID."""
        if "llama" in self.model_id.lower():
            self._load_llama_tokenizer()
        elif "t5" in self.model_id.lower():
            self._load_t5_tokenizer()
        elif "gpt" in self.model_id.lower():
            self._load_gpt_tokenizer()
        else:
            logger.warning(f"Unknown model type: {self.model_id}. Using fallback tokenizer.")
            self._load_fallback_tokenizer()
    
    def _load_llama_tokenizer(self):
        """Load LLaMA tokenizer."""
        if not TOKENIZERS_AVAILABLE:
            logger.error("LLaMA tokenizer requires 'tokenizers' library.")
            self._load_fallback_tokenizer()
            return
        
        try:
            # In a real implementation, we would download or locate the tokenizer files
            # For this mock implementation, we'll create a basic tokenizer
            self.tokenizer = self._create_mock_tokenizer()
            self.tokenizer_type = "llama"
            
            # Specific settings for LLaMA
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.pad_token_id = 0
            
            logger.info("Loaded LLaMA tokenizer")
        except Exception as e:
            logger.error(f"Error loading LLaMA tokenizer: {e}")
            self._load_fallback_tokenizer()
    
    def _load_t5_tokenizer(self):
        """Load T5 tokenizer."""
        if not SENTENCEPIECE_AVAILABLE:
            logger.error("T5 tokenizer requires 'sentencepiece' library.")
            self._load_fallback_tokenizer()
            return
        
        try:
            # In a real implementation, we would load the SentencePiece model
            # For this mock implementation, we'll create a basic tokenizer
            self.sp_model = None  # This would be a SentencePiece model
            self.tokenizer_type = "t5"
            
            # Specific settings for T5
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.unk_token_id = 2
            
            logger.info("Loaded T5 tokenizer")
        except Exception as e:
            logger.error(f"Error loading T5 tokenizer: {e}")
            self._load_fallback_tokenizer()
    
    def _load_gpt_tokenizer(self):
        """Load GPT tokenizer."""
        if not TOKENIZERS_AVAILABLE:
            logger.error("GPT tokenizer requires 'tokenizers' library.")
            self._load_fallback_tokenizer()
            return
        
        try:
            # In a real implementation, we would load the tokenizer files
            # For this mock implementation, we'll create a basic tokenizer
            self.tokenizer = self._create_mock_tokenizer()
            self.tokenizer_type = "gpt"
            
            # Specific settings for GPT
            self.bos_token_id = 50256
            self.eos_token_id = 50256
            self.pad_token_id = 50256
            
            logger.info("Loaded GPT tokenizer")
        except Exception as e:
            logger.error(f"Error loading GPT tokenizer: {e}")
            self._load_fallback_tokenizer()
    
    def _load_fallback_tokenizer(self):
        """Load a fallback tokenizer when the specific one is not available."""
        self.tokenizer_type = "fallback"
        logger.warning("Using fallback tokenizer with limited functionality")
    
    def _create_mock_tokenizer(self):
        """Create a mock tokenizer for demonstration purposes."""
        if TOKENIZERS_AVAILABLE:
            # Create a simple Hugging Face tokenizer
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import Whitespace
            
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            
            return tokenizer
        else:
            return None
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a single text string to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens like BOS/EOS
        
        Returns:
            List of token IDs
        """
        return self.encode_batch([text], add_special_tokens)[0]
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts to token IDs.
        
        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens like BOS/EOS
        
        Returns:
            List of token ID lists for each text
        """
        if self.tokenizer_type == "fallback":
            # Very basic fallback implementation
            return [[ord(c) % 30000 for c in text] for text in texts]
        
        elif self.tokenizer_type == "llama" or self.tokenizer_type == "gpt":
            if not TOKENIZERS_AVAILABLE:
                return self._fallback_encode_batch(texts, add_special_tokens)
            
            # Use Hugging Face tokenizers library
            encoding = self.tokenizer.encode_batch(texts)
            token_ids = [e.ids for e in encoding]
            
            # Add special tokens if requested
            if add_special_tokens:
                token_ids = [
                    [self.bos_token_id] + ids + [self.eos_token_id]
                    for ids in token_ids
                ]
            
            return token_ids
        
        elif self.tokenizer_type == "t5":
            if not SENTENCEPIECE_AVAILABLE or self.sp_model is None:
                return self._fallback_encode_batch(texts, add_special_tokens)
            
            # Use SentencePiece
            token_ids = []
            for text in texts:
                ids = self.sp_model.encode(text, out_type=int)
                if add_special_tokens:
                    ids = [self.bos_token_id] + ids + [self.eos_token_id]
                token_ids.append(ids)
            
            return token_ids
        
        else:
            return self._fallback_encode_batch(texts, add_special_tokens)
        
    def _fallback_encode_batch(self, texts: List[str], add_special_tokens: bool) -> List[List[int]]:
        """Fallback implementation for encoding when libraries aren't available."""
        token_ids = []
        for text in texts:
            # Simple character-based tokenization as fallback
            ids = [ord(c) % 30000 for c in text]
            if add_special_tokens:
                ids = [self.bos_token_id] + ids + [self.eos_token_id]
            token_ids.append(ids)
        
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens like BOS/EOS
        
        Returns:
            Decoded text string
        """
        return self.decode_batch([token_ids], skip_special_tokens)[0]

    def decode_batch(self, batch_token_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token ID lists back to texts.
        
        Args:
            batch_token_ids: List of token ID lists to decode
            skip_special_tokens: Whether to skip special tokens like BOS/EOS
        
        Returns:
            List of decoded text strings
        """
        if self.tokenizer_type == "fallback":
            # Very basic fallback implementation
            return ["".join(chr(min(id, 127)) for id in ids) for ids in batch_token_ids]
        
        elif self.tokenizer_type == "llama" or self.tokenizer_type == "gpt":
            if not TOKENIZERS_AVAILABLE:
                return self._fallback_decode_batch(batch_token_ids, skip_special_tokens)
            
            # Filter special tokens if requested
            if skip_special_tokens:
                special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
                filtered_ids = [
                    [id for id in ids if id not in special_tokens]
                    for ids in batch_token_ids
                ]
            else:
                filtered_ids = batch_token_ids
            
            # Use Hugging Face tokenizers
            texts = self.tokenizer.decode_batch(filtered_ids)
            return texts
        
        elif self.tokenizer_type == "t5":
            if not SENTENCEPIECE_AVAILABLE or self.sp_model is None:
                return self._fallback_decode_batch(batch_token_ids, skip_special_tokens)
            
            # Filter special tokens if requested
            texts = []
            for ids in batch_token_ids:
                if skip_special_tokens:
                    special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
                    ids = [id for id in ids if id not in special_tokens]
                
                text = self.sp_model.decode(ids)
                texts.append(text)
            
            return texts
        
        else:
            return self._fallback_decode_batch(batch_token_ids, skip_special_tokens)

    def _fallback_decode_batch(self, batch_token_ids: List[List[int]], skip_special_tokens: bool) -> List[str]:
        """Fallback implementation for decoding when libraries aren't available."""
        texts = []
        for ids in batch_token_ids:
            # Filter special tokens if requested
            if skip_special_tokens:
                special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
                ids = [id for id in ids if id not in special_tokens]
            
            # Simple character-based decoding as fallback
            text = "".join(chr(min(id, 127)) for id in ids)
            texts.append(text)
        
        return texts

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        if self.tokenizer_type == "llama" or self.tokenizer_type == "gpt":
            if hasattr(self.tokenizer, "get_vocab_size"):
                return self.tokenizer.get_vocab_size()
            return 32000  # Default approximate value
        
        elif self.tokenizer_type == "t5":
            if self.sp_model:
                return self.sp_model.vocab_size()
            return 32000  # Default approximate value
        
        return 30000  # Fallback approximate value
```

# src/utils/__init__.py

```py
"""
Utility functions for the DistributedLLM system.

This package contains utility functions for networking, serialization,
and other common operations used throughout the system.
"""

from src.utils.networking import (
    send_message, 
    receive_message, 
    discover_nodes, 
    register_node,
    get_local_ip,
    is_port_available,
    find_available_port
)
from src.utils.serialization import (
    serialize_tensor,
    deserialize_tensor,
    serialize_model_weights,
    deserialize_model_weights,
    save_model_weights,
    load_model_weights,
    serialize_cache,
    deserialize_cache
)

__all__ = [
    "send_message",
    "receive_message",
    "discover_nodes",
    "register_node",
    "get_local_ip",
    "is_port_available",
    "find_available_port",
    "serialize_tensor",
    "deserialize_tensor",
    "serialize_model_weights",
    "deserialize_model_weights",
    "save_model_weights",
    "load_model_weights",
    "serialize_cache",
    "deserialize_cache"
]
```

# src/utils/networking.py

```py
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
```

# src/utils/serialization.py

```py
"""
Serialization utilities for DistributedLLM.

Provides functions for efficient serialization and deserialization of
model weights, tensors, and other data structures.
"""

import logging
import io
import os
import pickle
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some serialization features will be limited.")


def serialize_tensor(tensor, compress: bool = True) -> bytes:
    """
    Serialize a tensor to bytes, with optional compression.
    
    Args:
        tensor: PyTorch tensor or NumPy array to serialize
        compress: Whether to apply compression
    
    Returns:
        Serialized tensor as bytes
    """
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        # Convert PyTorch tensor to NumPy array
        tensor = tensor.detach().cpu().numpy()
    
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")
    
    # Get tensor metadata
    dtype = str(tensor.dtype)
    shape = tensor.shape
    
    # Serialize metadata
    metadata = {
        'dtype': dtype,
        'shape': shape,
    }
    metadata_bytes = pickle.dumps(metadata)
    
    # Serialize tensor data
    tensor_bytes = tensor.tobytes()
    
    # Combine metadata and tensor data
    metadata_size = len(metadata_bytes)
    combined = struct.pack('>I', metadata_size) + metadata_bytes + tensor_bytes
    
    # Apply compression if requested
    if compress:
        compressed = zlib.compress(combined)
        # Add compression flag
        result = b'C' + compressed
    else:
        # Add no-compression flag
        result = b'N' + combined
    
    return result


def deserialize_tensor(data: bytes) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Deserialize a tensor from bytes.
    
    Args:
        data: Serialized tensor bytes
    
    Returns:
        Deserialized tensor (NumPy array or PyTorch tensor based on availability)
    """
    # Check compression flag
    compression_flag = data[0:1]
    data = data[1:]
    
    # Decompress if needed
    if compression_flag == b'C':
        data = zlib.decompress(data)
    
    # Extract metadata size
    metadata_size = struct.unpack('>I', data[:4])[0]
    
    # Extract metadata
    metadata_bytes = data[4:4+metadata_size]
    metadata = pickle.loads(metadata_bytes)
    
    # Extract tensor data
    tensor_bytes = data[4+metadata_size:]
    
    # Reconstruct tensor
    dtype = np.dtype(metadata['dtype'])
    shape = metadata['shape']
    
    array = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
    
    # Convert to PyTorch tensor if available
    if TORCH_AVAILABLE:
        return torch.from_numpy(array)
    else:
        return array


def serialize_model_weights(weights: Dict[str, Any], compress: bool = True) -> bytes:
    """
    Serialize model weights to bytes.
    
    Args:
        weights: Dictionary of weight tensors
        compress: Whether to apply compression
    
    Returns:
        Serialized weights as bytes
    """
    # Serialize each tensor in the weights dictionary
    serialized_weights = {}
    for key, tensor in weights.items():
        serialized_weights[key] = serialize_tensor(tensor, compress=False)
    
    # Serialize the dictionary
    result = pickle.dumps(serialized_weights)
    
    # Apply compression if requested
    if compress:
        result = zlib.compress(result)
    
    return result


def deserialize_model_weights(data: bytes) -> Dict[str, Any]:
    """
    Deserialize model weights from bytes.
    
    Args:
        data: Serialized weights bytes
    
    Returns:
        Dictionary of weight tensors
    """
    try:
        # Try to decompress first (in case it's compressed)
        try:
            data = zlib.decompress(data)
        except zlib.error:
            # Not compressed, continue with original data
            pass
        
        # Deserialize the dictionary
        serialized_weights = pickle.loads(data)
        
        # Deserialize each tensor in the dictionary
        weights = {}
        for key, tensor_bytes in serialized_weights.items():
            weights[key] = deserialize_tensor(tensor_bytes)
        
        return weights
    
    except Exception as e:
        logger.error(f"Error deserializing model weights: {e}")
        raise


def save_model_weights(weights: Dict[str, Any], file_path: str, compress: bool = True):
    """
    Save model weights to a file.
    
    Args:
        weights: Dictionary of weight tensors
        file_path: Path to save the weights to
        compress: Whether to apply compression
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Serialize weights
    serialized = serialize_model_weights(weights, compress=compress)
    
    # Save to file
    with open(file_path, 'wb') as f:
        f.write(serialized)
    
    logger.info(f"Saved model weights to {file_path}")


def load_model_weights(file_path: str) -> Dict[str, Any]:
    """
    Load model weights from a file.
    
    Args:
        file_path: Path to load the weights from
    
    Returns:
        Dictionary of weight tensors
    """
    # Load from file
    with open(file_path, 'rb') as f:
        serialized = f.read()
    
    # Deserialize weights
    weights = deserialize_model_weights(serialized)
    
    logger.info(f"Loaded model weights from {file_path}")
    return weights


def serialize_cache(cache_data: Dict[str, Any]) -> bytes:
    """
    Serialize a key-value cache for efficient storage.
    
    Args:
        cache_data: Dictionary of cache data
    
    Returns:
        Serialized cache as bytes
    """
    # Special handling for tensors in the cache
    serialized_cache = {}
    for key, value in cache_data.items():
        if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            serialized_cache[key] = ('tensor', serialize_tensor(value))
        elif isinstance(value, np.ndarray):
            serialized_cache[key] = ('array', serialize_tensor(value))
        else:
            serialized_cache[key] = ('pickle', pickle.dumps(value))
    
    # Serialize the dictionary
    result = pickle.dumps(serialized_cache)
    
    # Apply compression
    return zlib.compress(result)


def deserialize_cache(data: bytes) -> Dict[str, Any]:
    """
    Deserialize a key-value cache.
    
    Args:
        data: Serialized cache bytes
    
    Returns:
        Dictionary of cache data
    """
    try:
        # Decompress the data
        decompressed = zlib.decompress(data)
        
        # Deserialize the dictionary
        serialized_cache = pickle.loads(decompressed)
        
        # Deserialize each value in the dictionary
        cache_data = {}
        for key, (value_type, value_bytes) in serialized_cache.items():
            if value_type in ('tensor', 'array'):
                cache_data[key] = deserialize_tensor(value_bytes)
            elif value_type == 'pickle':
                cache_data[key] = pickle.loads(value_bytes)
            else:
                logger.warning(f"Unknown value type: {value_type}")
                cache_data[key] = None
        
        return cache_data
    
    except Exception as e:
        logger.error(f"Error deserializing cache: {e}")
        raise


def chunk_large_tensor(tensor, chunk_size_mb: int = 100) -> List[bytes]:
    """
    Split a large tensor into smaller chunks for efficient transmission.
    
    Args:
        tensor: Tensor to chunk
        chunk_size_mb: Maximum chunk size in megabytes
    
    Returns:
        List of serialized tensor chunks
    """
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")
    
    # Calculate chunk size in elements
    bytes_per_element = tensor.itemsize
    elements_per_mb = 1024 * 1024 // bytes_per_element
    chunk_size_elements = chunk_size_mb * elements_per_mb
    
    # Flatten tensor
    flat_tensor = tensor.reshape(-1)
    total_elements = flat_tensor.size
    
    # Split into chunks
    chunks = []
    for i in range(0, total_elements, chunk_size_elements):
        chunk = flat_tensor[i:i+chunk_size_elements]
        chunk_tensor = chunk.reshape((-1,) + tensor.shape[1:])
        chunks.append(serialize_tensor(chunk_tensor))
    
    return chunks


def reassemble_tensor_chunks(chunks: List[bytes], original_shape: Optional[Tuple[int, ...]] = None) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Reassemble tensor chunks back into a single tensor.
    
    Args:
        chunks: List of serialized tensor chunks
        original_shape: Optional original tensor shape
    
    Returns:
        Reassembled tensor
    """
    # Deserialize chunks
    deserialized_chunks = [deserialize_tensor(chunk) for chunk in chunks]
    
    if not deserialized_chunks:
        raise ValueError("No chunks provided")
    
    # Concatenate chunks
    if TORCH_AVAILABLE and isinstance(deserialized_chunks[0], torch.Tensor):
        concatenated = torch.cat([chunk.flatten() for chunk in deserialized_chunks])
        if original_shape:
            concatenated = concatenated.reshape(original_shape)
    else:
        concatenated = np.concatenate([chunk.flatten() for chunk in deserialized_chunks])
        if original_shape:
            concatenated = concatenated.reshape(original_shape)
    
    return concatenated
```

# src/worker/__init__.py

```py
"""
Worker components for the DistributedLLM system.

This package contains components for worker nodes in the distributed system,
including compute engines, resource monitors, and communication utilities.
"""

from src.worker.compute_engine import ComputeEngine
from src.worker.resource_monitor import ResourceMonitor
from src.worker.communication import CoordinatorClient, TaskProcessor

__all__ = [
    "ComputeEngine",
    "ResourceMonitor",
    "CoordinatorClient",
    "TaskProcessor"
]
```

# src/worker/communication.py

```py
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
```

# src/worker/compute_engine.py

```py
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
```

# src/worker/resource_monitor.py

```py
"""
Resource Monitor for DistributedLLM worker nodes.

Monitors system resources like CPU, RAM, and GPU, providing utilization metrics
to the coordinator for workload balancing decisions.
"""

import logging
import os
import platform
import time
import psutil
from typing import Dict, Any, Optional

# Try importing GPU monitoring libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gpustat
    GPUSTAT_AVAILABLE = True
except ImportError:
    GPUSTAT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Monitors system resources and provides metrics for performance optimization.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize the resource monitor."""
        self.update_interval = update_interval
        self.last_update = 0
        self.metrics = {}
        
        # Initialize metrics
        self.update()
        
        logger.info(f"Resource monitor initialized. GPU monitoring: {TORCH_AVAILABLE or GPUSTAT_AVAILABLE}")
    
    def update(self) -> Dict[str, Any]:
        """Update resource metrics."""
        current_time = time.time()
        
        # Only update if interval has passed
        if current_time - self.last_update < self.update_interval:
            return self.metrics
        
        self.last_update = current_time
        
        # Update CPU metrics
        self.metrics.update(self._get_cpu_metrics())
        
        # Update memory metrics
        self.metrics.update(self._get_memory_metrics())
        
        # Update disk metrics
        self.metrics.update(self._get_disk_metrics())
        
        # Update GPU metrics if available
        gpu_metrics = self._get_gpu_metrics()
        if gpu_metrics:
            self.metrics.update(gpu_metrics)
        
        return self.metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current resource metrics."""
        self.update()
        return self.metrics
    
    def get_cpu_percent(self) -> float:
        """Get the current CPU utilization percentage."""
        self.update()
        return self.metrics.get("cpu_percent", 0.0)
    
    def get_memory_usage(self) -> float:
        """Get the current memory usage in MB."""
        self.update()
        return self.metrics.get("memory_used_mb", 0.0)
    
    def get_available_memory(self) -> float:
        """Get the available memory in MB."""
        self.update()
        return self.metrics.get("memory_available_mb", 0.0)
    
    def get_gpu_memory_usage(self) -> Optional[float]:
        """Get the current GPU memory usage in MB, if available."""
        self.update()
        return self.metrics.get("gpu_memory_used_mb")
    
    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU-related metrics."""
        cpu_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "cpu_load_1min": os.getloadavg()[0] if hasattr(os, 'getloadavg') else None,
            "cpu_load_5min": os.getloadavg()[1] if hasattr(os, 'getloadavg') else None,
            "cpu_load_15min": os.getloadavg()[2] if hasattr(os, 'getloadavg') else None,
        }
        
        return cpu_metrics
    
    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory-related metrics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_metrics = {
            "memory_total_mb": mem.total / (1024 * 1024),
            "memory_available_mb": mem.available / (1024 * 1024),
            "memory_used_mb": mem.used / (1024 * 1024),
            "memory_percent": mem.percent,
            "swap_total_mb": swap.total / (1024 * 1024),
            "swap_used_mb": swap.used / (1024 * 1024),
            "swap_percent": swap.percent
        }
        
        return memory_metrics
    
    def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk-related metrics."""
        disk = psutil.disk_usage('/')
        
        disk_metrics = {
            "disk_total_gb": disk.total / (1024 * 1024 * 1024),
            "disk_free_gb": disk.free / (1024 * 1024 * 1024),
            "disk_used_gb": disk.used / (1024 * 1024 * 1024),
            "disk_percent": disk.percent
        }
        
        return disk_metrics
    
    def _get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU-related metrics if available."""
        if not TORCH_AVAILABLE and not GPUSTAT_AVAILABLE:
            return None
        
        gpu_metrics = {}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device_count = torch.cuda.device_count()
                gpu_metrics["gpu_count"] = device_count
                
                if device_count > 0:
                    # Get metrics for the first GPU
                    gpu_metrics["gpu_name"] = torch.cuda.get_device_name(0)
                    
                    # Try to get memory usage
                    if hasattr(torch.cuda, 'memory_allocated') and hasattr(torch.cuda, 'memory_reserved'):
                        allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                        reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
                        gpu_metrics["gpu_memory_used_mb"] = allocated
                        gpu_metrics["gpu_memory_reserved_mb"] = reserved
                        
                        # Get total memory
                        if hasattr(torch.cuda, 'get_device_properties'):
                            props = torch.cuda.get_device_properties(0)
                            if hasattr(props, 'total_memory'):
                                total = props.total_memory / (1024 * 1024)
                                gpu_metrics["gpu_memory_total_mb"] = total
                                gpu_metrics["gpu_memory_percent"] = (allocated / total) * 100
            
            except Exception as e:
                logger.warning(f"Error getting PyTorch GPU metrics: {e}")
        
        elif GPUSTAT_AVAILABLE:
            try:
                gpu_stats = gpustat.GPUStatCollection.new_query()
                if gpu_stats.gpus:
                    gpu = gpu_stats.gpus[0]  # First GPU
                    gpu_metrics["gpu_count"] = len(gpu_stats.gpus)
                    gpu_metrics["gpu_name"] = gpu.name
                    gpu_metrics["gpu_temperature"] = gpu.temperature
                    gpu_metrics["gpu_utilization"] = gpu.utilization
                    gpu_metrics["gpu_memory_used_mb"] = gpu.memory_used
                    gpu_metrics["gpu_memory_total_mb"] = gpu.memory_total
                    gpu_metrics["gpu_memory_percent"] = (gpu.memory_used / gpu.memory_total) * 100
            
            except Exception as e:
                logger.warning(f"Error getting gpustat metrics: {e}")
        
        return gpu_metrics


# Simple test to verify functionality
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    monitor = ResourceMonitor()
    
    # Print initial metrics
    print("Initial metrics:")
    for key, value in monitor.get_metrics().items():
        print(f"  {key}: {value}")
    
    # Update and print again
    time.sleep(2)
    print("\nUpdated metrics:")
    for key, value in monitor.get_metrics().items():
        print(f"  {key}: {value}")
    
    # Print specific metrics
    print(f"\nCPU: {monitor.get_cpu_percent()}%")
    print(f"Memory: {monitor.get_memory_usage()} MB")
    print(f"GPU Memory: {monitor.get_gpu_memory_usage()} MB")
```

# tests/test_communication.py

```py

```

# tests/test_inference.py

```py

```

# tests/test_load_balancing.py

```py

```

