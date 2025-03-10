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
    // Setup logging based on configuration - Specify TUI mode to disable console logging
    setupLogging(config.logging, true); // Pass 'true' to indicate we're in TUI mode
    
    logger.info('Starting SynthBot...');
    logger.debug('Configuration loaded', { config });
    
    // Initial project scan
    logger.info('Scanning project files...');
    const projectFiles = await scanProjectFiles(config.projectRoot, {
      exclude: config.context.excludeDirs.concat(config.context.excludeFiles),
      prioritize: config.context.priorityFiles
    });
    
    logger.info(`Found ${projectFiles.length} files in project`);
    
    // Create API client
    let apiClient;
    
    // Check if offline mode is enabled
    // Check if offline mode is enabled
    if (config.offline) {
      logger.info('Running in offline mode - no LLM connection will be attempted');
      
      // Create a dummy client for offline mode
      apiClient = {
        generateResponse: async (prompt) => ({ 
          text: "Running in offline mode. LLM services are not available.\n\nYour prompt was:\n" + prompt,
          tokens: 0 
        }),
        streamResponse: async (prompt, onToken, onComplete) => {
          onToken("Running in offline mode. LLM services are not available.");
          onComplete({ tokens: 0 });
        },
        getConnectionStatus: () => false,
        getModelInfo: async () => ({ name: 'offline', parameters: {} })
      };
    } else {
      // Normal mode - try to connect to LLM service
      // Check if we should use Ollama
      const useOllama = config.llm.api === 'ollama';
      
      logger.info(`Connecting to ${useOllama ? 'Ollama' : 'distributed LLM'} service...`);
      
      try {
        if (useOllama) {
          // Import the Ollama client
          const { createOllamaClient } = require('./api');
          
          apiClient = createOllamaClient({
            host: '127.0.0.1',
            port: config.llm.coordinatorPort,
            model: 'deepseek-r1:1.5b',  // Use the model you have installed
            temperature: config.llm.temperature
          });
          
          logger.info(`Attempting to connect to Ollama at ${config.llm.coordinatorHost}:${config.llm.coordinatorPort}`);
        } else {
          apiClient = createClient({
            host: config.llm.coordinatorHost,
            port: config.llm.coordinatorPort,
            model: config.llm.model,
            temperature: config.llm.temperature,
            api: config.llm.api
          });
          
          logger.info(`Attempting to connect to ${config.llm.api} at ${config.llm.coordinatorHost}:${config.llm.coordinatorPort}`);
        }
      } catch (error) {
        logger.error('Failed to create API client', { error });
        
        // Create a dummy client that returns error messages
        apiClient = {
          generateResponse: async () => ({ 
            text: "ERROR: Could not connect to LLM service. Please check if Ollama is running with 'ollama serve'\n\nTry running with --offline flag if you want to explore the UI without LLM connection.",
            tokens: 0 
          }),
          streamResponse: async (prompt, onToken, onComplete) => {
            onToken("ERROR: Could not connect to LLM service. Please check if Ollama is running with 'ollama serve'");
            onComplete({ tokens: 0 });
          },
          getConnectionStatus: () => false,
          getModelInfo: async () => ({ name: 'disconnected', parameters: {} })
        };
      }
    }
    
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
    
    // Start the TUI
    app.start();

    // Save statusBar reference for other components to use
    app.screen.statusBar = app.statusBar;
    
    // Handle application shutdown
    function shutdown() {
      logger.info('Shutting down FrankCode...');
      app.destroy();
      process.exit(0);
    }
    
    // Handle signals
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
    
    logger.info('SynthBot started successfully');
    
  } catch (error) {
    logger.error('Failed to start FrankCode', { error });
    console.error('Failed to start FrankCode:', error.message);
    process.exit(1);
  }
}

// IMPORTANT: Export the startApp function
module.exports = {
  startApp
};