/**
 * Configuration Utilities
 * 
 * Loads and manages application configuration
 */

const path = require('path');
const fs = require('fs');
const { cosmiconfig } = require('cosmiconfig');
const dotenv = require('dotenv');
const { logger } = require('./logger');

// Default configuration
const defaultConfig = {
  // LLM connection settings
  llm: {
    api: 'ollama',           // 'ollama' or 'distributed'
    coordinatorHost: 'localhost',
    coordinatorPort: 11434,  // Default Ollama port
    model: 'llama2',
    temperature: 0.7
  },
  
  // Project context settings
  context: {
    maxTokens: 8192,
    excludeDirs: ['node_modules', 'dist', '.git', 'build', 'vendor'],
    excludeFiles: ['*.lock', '*.log', '*.min.js', '*.min.css', '*.bin', '*.exe'],
    priorityFiles: ['README.md', 'package.json', 'composer.json', 'requirements.txt', 'Makefile']
  },
  
  // UI preferences
  ui: {
    theme: 'dark',
    layout: 'default',
    fontSize: 14
  },
  
  // Logging settings
  logging: {
    level: 'info',
    console: true,
    file: true
  }
};

/**
 * Load configuration
 * 
 * @param {Object} options Configuration options
 * @param {string} options.configPath Path to configuration file
 * @returns {Promise<Object>} Loaded configuration
 */
async function loadConfig(options = {}) {
  try {
    // Load .env file if it exists
    dotenv.config();
    
    // Load configuration using cosmiconfig
    const explorer = cosmiconfig('frankcode', {
      searchPlaces: [
        '.frankcoderc',
        '.frankcoderc.json',
        '.frankcoderc.yaml',
        '.frankcoderc.yml',
        '.frankcoderc.js',
        'frankcode.config.js',
        'package.json'
      ]
    });
    
    // Try to load the configuration
    let result;
    
    if (options.configPath) {
      // Load from specified path
      result = await explorer.load(options.configPath);
    } else {
      // Search for configuration
      result = await explorer.search();
    }
    
    // Get configuration object
    const config = result ? result.config : {};
    
    // Merge with default configuration
    const mergedConfig = mergeConfigs(defaultConfig, config);
    
    // Override with environment variables
    applyEnvironmentVariables(mergedConfig);
    
    // Override with command line options
    applyCommandLineOptions(mergedConfig, options);
    
    // Set project root
    mergedConfig.projectRoot = options.projectRoot || process.cwd();
    
    return mergedConfig;
  } catch (error) {
    logger.error('Failed to load configuration', { error });
    
    // Return default configuration
    return {
      ...defaultConfig,
      projectRoot: options.projectRoot || process.cwd()
    };
  }
}

/**
 * Merge configurations
 * 
 * @param {Object} target Target configuration
 * @param {Object} source Source configuration
 * @returns {Object} Merged configuration
 */
function mergeConfigs(target, source) {
  const result = { ...target };
  
  for (const key in source) {
    if (source[key] === null || source[key] === undefined) {
      continue;
    }
    
    if (typeof source[key] === 'object' && !Array.isArray(source[key])) {
      result[key] = mergeConfigs(result[key] || {}, source[key]);
    } else {
      result[key] = source[key];
    }
  }
  
  return result;
}

/**
 * Apply environment variables to configuration
 * 
 * @param {Object} config Configuration object
 */
function applyEnvironmentVariables(config) {
  // LLM configuration
  if (process.env.FRANKCODE_LLM_API) {
    config.llm.api = process.env.FRANKCODE_LLM_API;
  }
  
  if (process.env.FRANKCODE_LLM_HOST) {
    config.llm.coordinatorHost = process.env.FRANKCODE_LLM_HOST;
  }
  
  if (process.env.FRANKCODE_LLM_PORT) {
    config.llm.coordinatorPort = parseInt(process.env.FRANKCODE_LLM_PORT, 10);
  }
  
  if (process.env.FRANKCODE_LLM_MODEL) {
    config.llm.model = process.env.FRANKCODE_LLM_MODEL;
  }
  
  if (process.env.FRANKCODE_LLM_TEMPERATURE) {
    config.llm.temperature = parseFloat(process.env.FRANKCODE_LLM_TEMPERATURE);
  }
  
  // Context configuration
  if (process.env.FRANKCODE_CONTEXT_MAX_TOKENS) {
    config.context.maxTokens = parseInt(process.env.FRANKCODE_CONTEXT_MAX_TOKENS, 10);
  }
  
  // UI configuration
  if (process.env.FRANKCODE_UI_THEME) {
    config.ui.theme = process.env.FRANKCODE_UI_THEME;
  }
  
  // Logging configuration
  if (process.env.FRANKCODE_LOG_LEVEL) {
    config.logging.level = process.env.FRANKCODE_LOG_LEVEL;
  }
}

/**
 * Apply command line options to configuration
 * 
 * @param {Object} config Configuration object
 * @param {Object} options Command line options
 */
function applyCommandLineOptions(config, options) {
  // LLM configuration
  if (options.api) {
    config.llm.api = options.api;
  }
  
  if (options.host) {
    config.llm.coordinatorHost = options.host;
  }
  
  if (options.port) {
    config.llm.coordinatorPort = parseInt(options.port, 10);
  }
  
  if (options.model) {
    config.llm.model = options.model;
  }
  
  if (options.temperature !== undefined) {
    config.llm.temperature = parseFloat(options.temperature);
  }
  
  // Context configuration
  if (options.maxTokens) {
    config.context.maxTokens = parseInt(options.maxTokens, 10);
  }
  
  // UI configuration
  if (options.theme) {
    config.ui.theme = options.theme;
  }
  
  // Logging configuration
  if (options.verbose) {
    config.logging.level = 'debug';
  } else if (options.quiet) {
    config.logging.level = 'error';
  } else if (options.logLevel) {
    config.logging.level = options.logLevel;
  }
}

/**
 * Save configuration to file
 * 
 * @param {Object} config Configuration object
 * @param {string} filePath File path
 * @returns {Promise<boolean>} Success indicator
 */
async function saveConfig(config, filePath) {
  try {
    // Determine file format based on extension
    const ext = path.extname(filePath).toLowerCase();
    let content;
    
    if (ext === '.json') {
      content = JSON.stringify(config, null, 2);
    } else if (ext === '.js') {
      content = `module.exports = ${JSON.stringify(config, null, 2)};`;
    } else {
      // Default to JSON
      content = JSON.stringify(config, null, 2);
    }
    
    // Write to file
    await fs.promises.writeFile(filePath, content, 'utf8');
    
    logger.info(`Configuration saved to ${filePath}`);
    return true;
  } catch (error) {
    logger.error(`Failed to save configuration to ${filePath}`, { error });
    return false;
  }
}

module.exports = {
  loadConfig,
  saveConfig,
  defaultConfig
};