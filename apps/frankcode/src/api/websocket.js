/**
 * WebSocket Client for Distributed LLM Communication
 * 
 * Manages WebSocket connections to the distributed LLM network
 */

const WebSocket = require('ws');
const { logger } = require('../utils');

/**
 * Create a WebSocket client
 * 
 * @param {Object} options Configuration options
 * @param {string} options.host Host for WebSocket connection
 * @param {number} options.port Port for WebSocket connection
 * @param {Function} options.onConnect Connection callback
 * @param {Function} options.onDisconnect Disconnect callback
 * @param {Function} options.onError Error callback
 * @returns {Object} The WebSocket client interface
 */
function createWebSocketClient(options) {
  const {
    host,
    port,
    model,
    onConnect = () => {},
    onDisconnect = () => {},
    onError = () => {}
  } = options;
  
  // WebSocket instance
  let ws = null;
  
  // Request handling
  let nextRequestId = 1;
  const pendingRequests = new Map();
  
  /**
   * Connect to the WebSocket server
   * 
   * @returns {Promise<void>}
   */
  function connect() {
    return new Promise((resolve, reject) => {
      try {
        // Close existing connection if any
        if (ws) {
          ws.terminate();
        }
        
        // Create new WebSocket
        ws = new WebSocket(`ws://${host}:${port}`);
        
        // Set up event handlers
        ws.on('open', () => {
          logger.info(`WebSocket connected to ${host}:${port}`);
          onConnect();
          resolve();
        });
        
        ws.on('close', () => {
          logger.info('WebSocket connection closed');
          onDisconnect();
        });
        
        ws.on('error', (error) => {
          logger.error('WebSocket error', { error });
          onError(error);
          reject(error);
        });
        
        ws.on('message', (data) => {
          try {
            const message = JSON.parse(data);
            handleMessage(message);
          } catch (error) {
            logger.error('Error parsing WebSocket message', { error });
          }
        });
      } catch (error) {
        logger.error('Error connecting to WebSocket', { error });
        reject(error);
      }
    });
  }
  
  /**
   * Handle incoming messages
   * 
   * @param {Object} message Message object
   */
  function handleMessage(message) {
    // Check for response ID
    if (message.id && pendingRequests.has(message.id)) {
      const { resolve, reject, streamHandlers } = pendingRequests.get(message.id);
      
      // Handle streaming responses
      if (message.type === 'stream_token' && streamHandlers) {
        streamHandlers.onToken(message.token);
        return;
      }
      
      // Handle stream completion
      if (message.type === 'stream_end' && streamHandlers) {
        streamHandlers.onComplete(message.payload);
        pendingRequests.delete(message.id);
        return;
      }
      
      // Handle errors
      if (message.error) {
        reject(new Error(message.error));
        pendingRequests.delete(message.id);
        return;
      }
      
      // Handle normal responses
      resolve(message.payload);
      pendingRequests.delete(message.id);
    }
  }
  
  /**
   * Send a request to the server
   * 
   * @param {Object} request Request object
   * @returns {Promise<Object>} Response object
   */
  function sendRequest(request) {
    return new Promise((resolve, reject) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket not connected'));
        return;
      }
      
      // Generate request ID
      const id = nextRequestId++;
      
      // Store the promise callbacks
      pendingRequests.set(id, { resolve, reject });
      
      // Send the request
      ws.send(JSON.stringify({
        id,
        ...request
      }));
      
      // Set timeout
      setTimeout(() => {
        if (pendingRequests.has(id)) {
          pendingRequests.delete(id);
          reject(new Error('Request timed out'));
        }
      }, 60000); // 60 second timeout
    });
  }
  
  /**
   * Stream a request to the server
   * 
   * @param {Object} request Request object
   * @param {Function} onToken Token callback
   * @param {Function} onComplete Completion callback
   * @returns {Promise<void>}
   */
  function streamRequest({ type, payload, onToken, onComplete }) {
    return new Promise((resolve, reject) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket not connected'));
        return;
      }
      
      // Generate request ID
      const id = nextRequestId++;
      
      // Store callbacks
      pendingRequests.set(id, {
        resolve,
        reject,
        streamHandlers: { onToken, onComplete }
      });
      
      // Send the request
      ws.send(JSON.stringify({
        id,
        type,
        payload,
        stream: true
      }));
      
      // Request is resolved when we start receiving the stream
      resolve();
      
      // Set timeout
      setTimeout(() => {
        if (pendingRequests.has(id)) {
          pendingRequests.delete(id);
          reject(new Error('Stream request timed out'));
        }
      }, 60000); // 60 second timeout
    });
  }
  
  /**
   * Close the WebSocket connection
   */
  function close() {
    if (ws) {
      ws.close();
      ws = null;
    }
  }
  
  /**
   * Check if connected
   * 
   * @returns {boolean} Connection status
   */
  function isConnected() {
    return ws && ws.readyState === WebSocket.OPEN;
  }
  
  // Return the client interface
  return {
    connect,
    sendRequest,
    streamRequest,
    close,
    isConnected
  };
}

module.exports = {
  createWebSocketClient
};