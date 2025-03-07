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