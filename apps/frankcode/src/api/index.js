/**
 * API module entry point
 * 
 * Exports the client creation function and other utilities
 */

const { createClient } = require('./client');
const { createQueue } = require('./queue');
const { createWebSocketClient } = require('./websocket');
const { createOllamaClient } = require('./ollama-api-integration');

module.exports = {
  createClient,
  createQueue,
  createWebSocketClient,
  createOllamaClient
};