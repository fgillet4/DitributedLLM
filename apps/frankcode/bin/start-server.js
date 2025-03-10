#!/usr/bin/env node

/**
 * FrankCode Server Start Script
 * 
 * This script starts the FrankCode server and sets up Ngrok for remote access.
 */

const { startServer, stopServer } = require('../src/server/server-setup');
const { logger } = require('../src/utils/logger');

// Configure server options
const serverOptions = {
  port: process.env.SERVER_PORT || 3000,
  enableNgrok: true
};

/**
 * Main function to start the server
 */
async function main() {
  try {
    logger.info('Starting FrankCode server...');
    
    // Start the server
    const serverInfo = await startServer(serverOptions);
    
    logger.info(`Server running at ${serverInfo.localUrl}`);
    
    if (serverInfo.ngrokUrl) {
      logger.info(`Ngrok URL: ${serverInfo.ngrokUrl}`);
      logger.info('You can use this URL to access your Ollama instance remotely');
    }
    
    // Handle shutdown
    process.on('SIGINT', async () => {
      logger.info('Shutting down server...');
      await stopServer(serverInfo);
      process.exit(0);
    });
    
    process.on('SIGTERM', async () => {
      logger.info('Shutting down server...');
      await stopServer(serverInfo);
      process.exit(0);
    });
    
    // Log startup completion
    logger.info('Server startup complete');
  } catch (error) {
    logger.error(`Failed to start server: ${error.message}`);
    process.exit(1);
  }
}

// Run the main function
main();