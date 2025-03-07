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
      logger.info('Shutting down FrankCode...');
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
    logger.error('Failed to start FrankCode', { error });
    console.error('Failed to start FrankCode:', error.message);
    process.exit(1);
  }
}

module.exports = {
  startApp
};