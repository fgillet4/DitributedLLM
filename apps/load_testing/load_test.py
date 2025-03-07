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