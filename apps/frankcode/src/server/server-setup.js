/**
 * FrankCode Server Setup
 * 
 * This script sets up an API server for FrankCode and creates a public URL using ngrok.
 * It allows remote access to the LLM service and provides a proxy for Ollama.
 */

const express = require('express');
const cors = require('cors');
const ngrok = require('ngrok');
const fetch = require('node-fetch');
const fs = require('fs').promises;
const path = require('path');
const dotenv = require('dotenv');
const { logger } = require('../utils/logger');

// Load environment variables from parent directories
function loadEnvFromParentDirs() {
  let currentDir = process.cwd();
  let rootDir = path.parse(currentDir).root;
  
  // Try to find .env in current and parent directories
  while (currentDir !== rootDir) {
    const envPath = path.join(currentDir, '.env');
    try {
      if (fs.existsSync(envPath)) {
        logger.info(`Found .env file at ${envPath}`);
        dotenv.config({ path: envPath });
        return true;
      }
    } catch (error) {
      // Ignore errors and continue searching
    }
    
    // Move up one directory
    currentDir = path.dirname(currentDir);
  }
  
  // Try root directory as a last resort
  const rootEnvPath = path.join(rootDir, '.env');
  try {
    if (fs.existsSync(rootEnvPath)) {
      logger.info(`Found .env file at ${rootEnvPath}`);
      dotenv.config({ path: rootEnvPath });
      return true;
    }
  } catch (error) {
    // Ignore errors
  }
  
  logger.warn('No .env file found in any parent directory');
  return false;
}

// Load environment variables
loadEnvFromParentDirs();

// Ollama API configuration
const ollamaHost = process.env.OLLAMA_HOST || 'localhost';
const ollamaPort = process.env.OLLAMA_PORT || 11434;
const ollamaUrl = `http://${ollamaHost}:${ollamaPort}`;

/**
 * Start the FrankCode server
 * 
 * @param {Object} options Configuration options
 * @param {number} options.port Port for the server
 * @param {boolean} options.enableNgrok Whether to enable ngrok
 * @returns {Promise<Object>} Server info
 */
async function startServer(options = {}) {
  const {
    port = 3000,
    enableNgrok = true
  } = options;
  
  // Create Express app
  const app = express();
  
  // Configure middleware
  app.use(cors());
  app.use(express.json());
  
  // Health check endpoint
  app.get('/health', (req, res) => {
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString()
    });
  });
  
  // Ollama proxy endpoint
  app.post('/api/generate', async (req, res) => {
    try {
      const response = await fetch(`${ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(req.body)
      });
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      logger.error(`Error proxying to Ollama: ${error.message}`);
      res.status(500).json({
        error: 'Failed to proxy request to Ollama',
        message: error.message
      });
    }
  });
  
  // Ollama streaming endpoint
  app.post('/api/generate/stream', async (req, res) => {
    try {
      // Set up streaming response
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      // Forward request to Ollama
      const ollamaReq = await fetch(`${ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...req.body,
          stream: true
        })
      });
      
      if (!ollamaReq.ok) {
        throw new Error(`Ollama API returned status ${ollamaReq.status}`);
      }
      
      // Stream response from Ollama to client
      const reader = ollamaReq.body.getReader();
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        // Forward chunk to client
        res.write(value);
        res.flush();
      }
      
      res.end();
    } catch (error) {
      logger.error(`Error streaming from Ollama: ${error.message}`);
      res.status(500).json({
        error: 'Failed to stream from Ollama',
        message: error.message
      });
    }
  });
  
  // Ollama model list endpoint
  app.get('/api/models', async (req, res) => {
    try {
      const response = await fetch(`${ollamaUrl}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      logger.error(`Error getting models from Ollama: ${error.message}`);
      res.status(500).json({
        error: 'Failed to get models from Ollama',
        message: error.message
      });
    }
  });
  
  // Start the server
  return new Promise((resolve, reject) => {
    try {
      const server = app.listen(port, async () => {
        logger.info(`FrankCode server running on http://localhost:${port}`);
        
        let ngrokUrl = null;
        
        // Set up ngrok if enabled
        if (enableNgrok) {
          try {
            // Check for ngrok auth token
            const ngrokToken = process.env.NGROK_AUTH_TOKEN;
            
            if (!ngrokToken) {
              logger.warn('NGROK_AUTH_TOKEN not found in environment variables');
              logger.warn('Ngrok may not work correctly without authentication');
              logger.warn('Try specifying the token directly when starting the server:');
              logger.warn('  NGROK_AUTH_TOKEN=your_token node bin/start-server.js');
            } else {
              // Configure ngrok with auth token
              logger.info('Found NGROK_AUTH_TOKEN, authenticating...');
              await ngrok.authtoken(ngrokToken);
              logger.info('Ngrok authenticated successfully');
            }
            
            // Start ngrok tunnel
            ngrokUrl = await ngrok.connect({
              addr: port,
              region: 'us'
            });
            
            logger.info(`Ngrok URL: ${ngrokUrl}`);
            
            // Save ngrok URL to file for future reference
            const ngrokFilePath = path.join(process.cwd(), 'ngrok_url.txt');
            await fs.writeFile(ngrokFilePath, ngrokUrl);
            logger.info(`Ngrok URL saved to ${ngrokFilePath}`);
          } catch (ngrokError) {
            logger.error(`Failed to start ngrok: ${ngrokError.message}`);
            logger.info('Continuing without ngrok...');
          }
        }
        
        resolve({
          server,
          localUrl: `http://localhost:${port}`,
          ngrokUrl
        });
      });
      
      // Handle server errors
      server.on('error', (error) => {
        logger.error(`Server error: ${error.message}`);
        reject(error);
      });
    } catch (error) {
      logger.error(`Failed to start server: ${error.message}`);
      reject(error);
    }
  });
}

/**
 * Stop the server and clean up resources
 * 
 * @param {Object} serverInfo Server info object
 */
async function stopServer(serverInfo) {
  try {
    // Close the server
    if (serverInfo && serverInfo.server) {
      await new Promise((resolve) => {
        serverInfo.server.close(resolve);
      });
      logger.info('Server closed');
    }
    
    // Disconnect ngrok
    if (serverInfo && serverInfo.ngrokUrl) {
      await ngrok.disconnect();
      await ngrok.kill();
      logger.info('Ngrok disconnected');
    }
  } catch (error) {
    logger.error(`Error stopping server: ${error.message}`);
  }
}

module.exports = {
  startServer,
  stopServer
};