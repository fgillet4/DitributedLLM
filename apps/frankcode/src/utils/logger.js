// src/utils/logger.js - Complete rewrite to disable ALL console output

/**
 * Logger module
 * 
 * Provides logging capabilities for the application
 */

const winston = require('winston');
const fs = require('fs');
const path = require('path');

// Create logs directory if it doesn't exist
const logsDir = path.join(process.cwd(), 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir);
}

// Custom log levels
const levels = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3
};

// Create winston logger - NO CONSOLE TRANSPORT BY DEFAULT
const logger = winston.createLogger({
  levels,
  format: winston.format.combine(
    winston.format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    winston.format.errors({ stack: true }),
    winston.format.splat(),
    winston.format.json()
  ),
  defaultMeta: { service: 'frankcode' },
  transports: [
    // Write logs with level 'error' and below to error.log
    new winston.transports.File({
      filename: path.join(logsDir, 'error.log'),
      level: 'error'
    }),
    
    // Write logs with level 'info' and below to combined.log
    new winston.transports.File({
      filename: path.join(logsDir, 'combined.log')
    })
  ]
});

// Store original console methods
const originalConsole = {
  log: console.log,
  warn: console.warn,
  error: console.error,
  info: console.info,
  debug: console.debug
};

/**
 * Set up logging based on configuration
 * 
 * @param {Object} config Logging configuration
 * @param {boolean} isTUI Whether running in TUI mode
 */
function setupLogging(config = {}, isTUI = false) {
  // Default configuration
  const {
    level = 'info',
    console = !isTUI, // Disable console in TUI mode
    file = true,
    filePath,
    maxSize = '10m',
    maxFiles = 5
  } = config;
  
  // Set log level
  logger.level = level;
  
  // Configure file transports
  if (file) {
    // Custom log path or default
    const logPath = filePath || logsDir;
    
    // Ensure log directory exists
    if (!fs.existsSync(logPath)) {
      fs.mkdirSync(logPath, { recursive: true });
    }
  }
  
  // Override console methods in TUI mode
  if (isTUI) {
    // Redirect ALL console output to files in TUI mode
    console.log = (...args) => {
      logger.info(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
    };
    
    console.warn = (...args) => {
      logger.warn(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
    };
    
    console.error = (...args) => {
      logger.error(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
    };
    
    console.info = (...args) => {
      logger.info(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
    };
    
    console.debug = (...args) => {
      logger.debug(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
    };
    
    // Remove all console transports from Winston
    logger.transports = logger.transports.filter(
      t => t.name !== 'console'
    );
  } else {
    // Restore console methods if not in TUI mode
    console.log = originalConsole.log;
    console.warn = originalConsole.warn;
    console.error = originalConsole.error;
    console.info = originalConsole.info;
    console.debug = originalConsole.debug;
    
    // Add console transport to Winston if enabled
    if (console) {
      // Remove existing console transports
      logger.transports = logger.transports.filter(
        t => t.name !== 'console'
      );
      
      // Add new console transport
      logger.add(new winston.transports.Console({
        format: winston.format.combine(
          winston.format.colorize(),
          winston.format.simple()
        )
      }));
    }
  }
  
  // Log initialization (to file only in TUI mode)
  logger.info('Logging initialized', { level, isTUI });
}

// Create a special startup function to immediately disable console output
function quietStartup() {
  // Immediately redirect console to file
  console.log = (...args) => {
    logger.info(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
  };
  
  console.warn = (...args) => {
    logger.warn(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
  };
  
  console.error = (...args) => {
    logger.error(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
  };
  
  console.info = (...args) => {
    logger.info(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
  };
  
  console.debug = (...args) => {
    logger.debug(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
  };
  
  // Ensure no console transports
  logger.transports = logger.transports.filter(
    t => t.name !== 'console'
  );
  
  logger.info('Console output redirected to file for TUI mode');
}

module.exports = {
  logger,
  setupLogging,
  quietStartup
};