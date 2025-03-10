/**
 * Ollama API Integration for FrankCode
 * 
 * This module provides an API client for connecting to Ollama LLM service.
 * It's designed to be a drop-in replacement for the distributed LLM client.
 */

const fetch = require('node-fetch');
const { logger } = require('../utils/logger');

/**
 * Create an Ollama API client
 * 
 * @param {Object} options Configuration options
 * @param {string} options.host Host for the Ollama service
 * @param {number} options.port Port for the Ollama service
 * @param {string} options.model Model to use
 * @param {number} options.temperature Temperature setting (0-1)
 * @returns {Object} The Ollama API client interface
 */
function createOllamaClient(options) {
  const {
    host = 'localhost',
    port = 11434,
    model = 'llama2',
    temperature = 0.7
  } = options;
  
  const baseUrl = `http://${host}:${port}`;
  let isConnected = false;
  
  /**
   * Initialize the client and test connection
   * 
   * @returns {Promise<boolean>} Connection success
   */
  async function initialize() {
    try {
      logger.debug('Testing Ollama connection...');
      
      // Check if Ollama is running
      const response = await fetch(`${baseUrl}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      const models = data.models || [];
      
      // Log available models
      logger.info(`Connected to Ollama. Available models: ${models.map(m => m.name).join(', ') || 'none'}`);
      
      // Check if our model is available
      const modelAvailable = models.some(m => m.name === model);
      if (!modelAvailable && models.length > 0) {
        // Automatically use the first available model
        model = models[0].name;
        logger.info(`Model switched to available model: ${model}`);
      } else if (!modelAvailable) {
        logger.warn(`Model '${model}' not found in Ollama. Available models: ${models.map(m => m.name).join(', ')}`);
      }
      
      await testOllamaEndpoint();
      
      isConnected = true;
      return true;
    } catch (error) {
      logger.error(`Failed to connect to Ollama at ${baseUrl}: ${error.message}`);
      isConnected = false;
      return false;
    }
  }
  // Add this function near the top of the ollama-api-integration.js file, just inside the createOllamaClient function
async function testOllamaEndpoint() {
    try {
      console.log(`Testing Ollama API at ${baseUrl}...`);
      
      // Test /api/tags endpoint
      console.log("Testing /api/tags endpoint...");
      const tagsResponse = await fetch(`${baseUrl}/api/tags`);
      console.log(`/api/tags status: ${tagsResponse.status}`);
      
      if (tagsResponse.ok) {
        const tagsData = await tagsResponse.json();
        console.log(`Available models: ${JSON.stringify(tagsData)}`);
      }
      
      // Test minimal generate request
      console.log("Testing /api/generate endpoint with minimal request...");
      const minimalRequest = { model, prompt: "Hello" };
      console.log(`Request body: ${JSON.stringify(minimalRequest)}`);
      
      const genResponse = await fetch(`${baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(minimalRequest)
      });
      
      console.log(`/api/generate status: ${genResponse.status}`);
      console.log(`/api/generate status text: ${genResponse.statusText}`);
      
      // Log full request and response details
      console.log(`Full request URL: ${baseUrl}/api/generate`);
      console.log(`Host: ${host}, Port: ${port}`);
      
      // Try different endpoint variations
      console.log("Testing alternative endpoints...");
      const endpoints = [
        "/api/generate",
        "/api/completion",
        "/api/completions",
        "/api/chat/completions"
      ];
      
      for (const endpoint of endpoints) {
        try {
          const testResp = await fetch(`${baseUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(minimalRequest)
          });
          console.log(`${endpoint} status: ${testResp.status}`);
        } catch (err) {
          console.log(`${endpoint} error: ${err.message}`);
        }
      }
    } catch (error) {
      console.error(`Test failed: ${error.message}`);
    }
  }
  
  // Then add this line at the end of the initialize function:
  
  /**
   * Generate a response from Ollama
   * 
   * @param {string} prompt The prompt to send
   * @param {Object} options Additional options
   * @returns {Promise<Object>} The response
   */
  async function generateResponse(prompt, options = {}) {
    try {
      logger.debug(`Sending request to Ollama (${model})`);
      
      // Use a minimal request body matching the successful curl command
      const requestBody = {
        model: model,
        prompt: prompt
      };
      
      const response = await fetch(`${baseUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      // Read response as text first
      const responseText = await response.text();
      
      // Handle different response formats
      try {
        // Try parsing as single JSON object
        const data = JSON.parse(responseText);
        return {
          text: data.response || "No response text",
          tokens: data.eval_count || 0,
          model: data.model || model
        };
      } catch (parseError) {
        // If not single JSON, try parsing as multiple JSON lines
        try {
          const lines = responseText.split('\n').filter(line => line.trim());
          // Get the last complete JSON object
          const lastLine = lines[lines.length - 1];
          const lastData = JSON.parse(lastLine);
          
          // Combine all responses
          const fullText = lines.map(line => {
            try {
              const obj = JSON.parse(line);
              return obj.response || '';
            } catch (e) {
              return '';
            }
          }).join('');
          
          return {
            text: fullText,
            tokens: lastData.eval_count || lines.length,
            model: model
          };
        } catch (e) {
          // If all parsing fails, return raw text
          return {
            text: responseText,
            tokens: responseText.length / 4, // rough estimate
            model: model
          };
        }
      }
    } catch (error) {
      logger.error(`Error generating response from Ollama: ${error.message}`);
      return {
        text: `Error: Failed to generate response from Ollama. ${error.message}`,
        tokens: 0,
        error: true
      };
    }
  }
  
  /**
   * Stream a response from Ollama
   * 
   * @param {string} prompt The prompt to send
   * @param {Function} onToken Callback for each token
   * @param {Function} onComplete Callback when complete
   * @param {Object} options Additional options
   */
  async function streamResponse(prompt, onToken, onComplete, options = {}) {
    try {
      const requestOptions = {
        temperature: options.temperature || temperature,
        max_tokens: options.maxTokens || 2048,
        stop: options.stop || ['\n\nUSER:', '\n\nASSISTANT:']
      };
      
      logger.debug(`Streaming request to Ollama (${model})`);
      
      const response = await fetch(`${baseUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model,
          prompt,
          temperature: requestOptions.temperature,
          max_tokens: requestOptions.max_tokens,
          stop: requestOptions.stop,
          stream: true
        })
      });
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
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
            logger.error(`Error parsing Ollama stream: ${error.message}`);
          }
        }
      }
      
      // Final completion if we haven't already called it
      onComplete({
        model,
        tokens: totalTokens
      });
    } catch (error) {
      logger.error(`Error streaming response from Ollama: ${error.message}`);
      onToken(`Error: Failed to stream response from Ollama. ${error.message}`);
      onComplete({ tokens: 0, error: true });
    }
  }
  
  /**
   * Get model information
   * 
   * @returns {Promise<Object>} Model information
   */
  async function getModelInfo() {
    try {
      const response = await fetch(`${baseUrl}/api/show`, {
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
        family: data.modelfile?.parameter || 'unknown'
      };
    } catch (error) {
      logger.error(`Error getting model info: ${error.message}`);
      return {
        name: model,
        parameters: {},
        size: 'unknown',
        family: 'unknown'
      };
    }
  }
  
  /**
   * Get connection status
   * 
   * @returns {boolean} Connection status
   */
  function getConnectionStatus() {
    return isConnected;
  }
  
  // Initialize on creation
  initialize().catch(error => {
    logger.error(`Failed to initialize Ollama client: ${error.message}`);
  });
  
  // Return the client interface
  return {
    generateResponse,
    streamResponse,
    getModelInfo,
    getConnectionStatus,
    reconnect: initialize
  };
}

module.exports = {
  createOllamaClient
};