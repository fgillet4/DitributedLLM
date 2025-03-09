/**
 * API Client for LLM Interaction
 * 
 * Supports both distributed LLM networks and Ollama
 */

const fetch = require('node-fetch');
const WebSocket = require('ws');
const { logger } = require('../utils/logger');
const { createQueue } = require('./queue');
const { createWebSocketClient } = require('./websocket');

/**
 * Create an API client
 * 
 * @param {Object} options Configuration options
 * @param {string} options.host Host for the LLM service
 * @param {number} options.port Port for the LLM service
 * @param {string} options.model Model to use
 * @param {number} options.temperature Temperature setting (0-1)
 * @returns {Object} The API client interface
 */
function createClient(options) {
  const {
    host,
    port,
    model,
    temperature = 0.7,
    api = 'ollama' // 'ollama' or 'distributed'
  } = options;
  
  // State tracking
  let isConnected = false;
  let reconnectAttempts = 0;
  const MAX_RECONNECT_ATTEMPTS = 5;
  
  // Create request queue
  const requestQueue = createQueue({
    concurrency: 1, // Process one request at a time
    timeout: 60000   // 60 second timeout
  });
  
  // WebSocket client for distributed LLM
  let wsClient = null;
  
    /**
   * Initialize the client
   * 
   * @returns {Promise<void>}
   */
  async function initialize() {
    try {
      if (api === 'distributed') {
        // Initialize WebSocket connection for distributed mode
        wsClient = createWebSocketClient({
          host,
          port,
          model,
          onConnect: () => {
            isConnected = true;
            reconnectAttempts = 0;
            logger.info('Connected to distributed LLM network');
          },
          onDisconnect: () => {
            isConnected = false;
            logger.warn('Disconnected from distributed LLM network');
            
            // Try to reconnect
            if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
              reconnectAttempts++;
              const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
              
              logger.info(`Attempting to reconnect in ${delay / 1000} seconds...`);
              setTimeout(() => wsClient.connect(), delay);
            } else {
              logger.error('Max reconnection attempts reached');
            }
          },
          onError: (error) => {
            logger.error('WebSocket error', { error });
          }
        });
        
        await wsClient.connect();
      } else {
        // For Ollama, just test the connection
        try {
          const response = await fetch(`http://${host}:${port}/api/tags`);
          if (response.ok) {
            const data = await response.json();
            isConnected = true;
            
            // Log available models
            const models = data.models || [];
            logger.info(`Connected to Ollama with ${models.length} models available`);
            
            // Check if our model is available
            const modelAvailable = models.some(m => m.name === model);
            if (!modelAvailable) {
              logger.warn(`Model '${model}' not found in Ollama`);
            }
          } else {
            throw new Error(`Ollama API returned status ${response.status}`);
          }
        } catch (error) {
          logger.error('Failed to connect to Ollama API', { error });
          isConnected = false;
        }
      }
    } catch (error) {
      logger.error('Failed to initialize API client connection', { error });
      isConnected = false;
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
      
      // Handle error from API client
      if (response.error) {
        // Add a special response for network errors
        conversationHistory.push({ role: 'assistant', content: response.text });
        
        return {
          text: response.text,
          error: true,
          tokenUsage: tokenMonitor.getCurrentUsage()
        };
      }
      
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
      
      // Add error to conversation history
      conversationHistory.push({ 
        role: 'assistant', 
        content: `Error processing message: ${error.message}` 
      });
      
      return {
        text: `Error processing message: ${error.message}`,
        error: true,
        tokenUsage: tokenMonitor.getCurrentUsage()
      };
    }
  }
  
  /**
   * Generate a response from the LLM
   * 
   * @param {string} prompt Prompt to send to the LLM
   * @param {Object} options Request options
   * @returns {Promise<Object>} Response object
   */
  async function generateResponse(prompt, options = {}) {
    const requestOptions = {
      temperature: options.temperature || temperature,
      maxTokens: options.maxTokens || 2048,
      stop: options.stop || ['\n\nUSER:', '\n\nASSISTANT:']
    };
    
    // Add to request queue
    return requestQueue.add(async () => {
      if (api === 'distributed') {
        return generateDistributedResponse(prompt, requestOptions);
      } else {
        return generateOllamaResponse(prompt, requestOptions);
      }
    });
  }
  
  /**
   * Generate a response using the distributed LLM network
   * 
   * @param {string} prompt Prompt to send
   * @param {Object} options Request options
   * @returns {Promise<Object>} Response object
   */
  async function generateDistributedResponse(prompt, options) {
    if (!isConnected || !wsClient) {
      throw new Error('Not connected to distributed LLM network');
    }
    
    try {
      const response = await wsClient.sendRequest({
        type: 'generate',
        payload: {
          prompt,
          model,
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stop_sequences: options.stop
        }
      });
      
      return {
        text: response.text,
        tokens: response.usage?.total_tokens || 0,
        model: response.model || model
      };
    } catch (error) {
      logger.error('Failed to generate response from distributed LLM', { error });
      throw error;
    }
  }
  
  /**
   * Generate a response using Ollama
   * 
   * @param {string} prompt Prompt to send
   * @param {Object} options Request options
   * @returns {Promise<Object>} Response object
   */
  async function generateOllamaResponse(prompt, options) {
    try {
      const response = await fetch(`http://${host}:${port}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model,
          prompt,
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stop: options.stop
        })
      });
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      
      return {
        text: data.response,
        tokens: data.eval_count || 0,
        model: data.model || model
      };
    } catch (error) {
      // Improved error handling for better TUI display
      logger.error('Failed to generate response from Ollama', { error });
      
      // Return a user-friendly message instead of throwing
      return {
        text: `Error: Connection to Ollama failed. Try running with --offline flag or make sure Ollama is running with 'ollama serve'\n\nTechnical details: ${error.message}`,
        tokens: 0,
        error: true
      };
    }
  }
  
  /**
   * Stream a response from the LLM
   * 
   * @param {string} prompt Prompt to send
   * @param {Function} onToken Callback for each token
   * @param {Function} onComplete Callback when complete
   * @param {Object} options Request options
   * @returns {Promise<void>}
   */
  async function streamResponse(prompt, onToken, onComplete, options = {}) {
    const requestOptions = {
      temperature: options.temperature || temperature,
      maxTokens: options.maxTokens || 2048,
      stop: options.stop || ['\n\nUSER:', '\n\nASSISTANT:']
    };
    
    // Add to request queue
    return requestQueue.add(async () => {
      if (api === 'distributed') {
        return streamDistributedResponse(prompt, onToken, onComplete, requestOptions);
      } else {
        return streamOllamaResponse(prompt, onToken, onComplete, requestOptions);
      }
    });
  }
  
  /**
   * Stream a response from the distributed LLM network
   * 
   * @param {string} prompt Prompt to send
   * @param {Function} onToken Callback for each token
   * @param {Function} onComplete Callback when complete
   * @param {Object} options Request options
   * @returns {Promise<void>}
   */
  async function streamDistributedResponse(prompt, onToken, onComplete, options) {
    if (!isConnected || !wsClient) {
      throw new Error('Not connected to distributed LLM network');
    }
    
    try {
      await wsClient.streamRequest({
        type: 'generate_stream',
        payload: {
          prompt,
          model,
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stop_sequences: options.stop
        },
        onToken,
        onComplete
      });
    } catch (error) {
      logger.error('Failed to stream response from distributed LLM', { error });
      throw error;
    }
  }
  
  /**
   * Stream a response from Ollama
   * 
   * @param {string} prompt Prompt to send
   * @param {Function} onToken Callback for each token
   * @param {Function} onComplete Callback when complete
   * @param {Object} options Request options
   * @returns {Promise<void>}
   */
  async function streamOllamaResponse(prompt, onToken, onComplete, options) {
    try {
      const response = await fetch(`http://${host}:${port}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model,
          prompt,
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stop: options.stop,
          stream: true
        })
      });
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      // Parse the stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let totalTokens = 0;
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        // Decode the chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Process complete JSON objects from the buffer
        let newlineIndex;
        while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + 1);
          
          if (line.trim() === '') continue;
          
          try {
            const data = JSON.parse(line);
            
            if (data.response) {
              onToken(data.response);
              totalTokens++;
            }
            
            // Check if done
            if (data.done) {
              onComplete({
                model: data.model || model,
                tokens: totalTokens
              });
              return;
            }
          } catch (error) {
            logger.error('Error parsing Ollama stream', { error, line });
          }
        }
      }
      
      // Final completion if we haven't already called it
      onComplete({
        model,
        tokens: totalTokens
      });
    } catch (error) {
      logger.error('Failed to stream response from Ollama', { error });
      throw error;
    }
  }
  
  /**
   * Execute a shell command using the LLM
   * 
   * @param {string} command Command to execute
   * @returns {Promise<Object>} Execution result
   */
  async function executeCommand(command) {
    // This is handled by the agent, not the LLM directly
    throw new Error('Command execution should be handled by the agent');
  }
  
  /**
   * Check if the client is connected
   * 
   * @returns {boolean} Connection status
   */
  function getConnectionStatus() {
    return isConnected;
  }
  
  /**
   * Get information about the current model
   * 
   * @returns {Promise<Object>} Model information
   */
  async function getModelInfo() {
    try {
      if (api === 'ollama') {
        const response = await fetch(`http://${host}:${port}/api/show`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            name: model
          })
        });
        
        if (!response.ok) {
          throw new Error(`Ollama API returned status ${response.status}`);
        }
        
        const data = await response.json();
        
        return {
          name: data.model || model,
          parameters: data.parameters || {},
          size: data.size || 'unknown',
          family: data.family || 'unknown'
        };
      } else {
        // For distributed LLM
        if (!wsClient) {
          throw new Error('Not connected to distributed LLM network');
        }
        
        const response = await wsClient.sendRequest({
          type: 'model_info',
          payload: { model }
        });
        
        return {
          name: response.name || model,
          parameters: response.parameters || {},
          size: response.size || 'unknown',
          family: response.family || 'unknown'
        };
      }
    } catch (error) {
      logger.error('Failed to get model info', { error });
      throw error;
    }
  }
  
  // Initialize on creation
  initialize().catch(error => {
    logger.error('Failed to initialize API client', { error });
  });
  
  // Return the client interface
  return {
    generateResponse,
    streamResponse,
    executeCommand,
    getConnectionStatus,
    getModelInfo,
    reconnect: initialize
  };
}

// Export the client creation function
module.exports = {
  createClient
};