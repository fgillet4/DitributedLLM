/**
 * Logger module
 * 
 * Provides logging capabilities for the application
 */

const winston = require('winston');
const fs = require('fs');
const path = require('path');

// Custom log levels
const levels = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3
};

// Create logs directory if it doesn't exist
const logsDir = path.join(process.cwd(), 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir);
}

// Create winston logger
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

/**
 * Set up logging based on configuration
 * 
 * @param {Object} config Logging configuration
 */
function setupLogging(config = {}) {
  // Default configuration
  const {
    level = 'info',
    console = true,
    file = true,
    filePath,
    maxSize = '10m',
    maxFiles = 5
  } = config;
  
  // Set log level
  logger.level = level;
  
  // Add console transport if enabled
  if (console) {
    logger.add(new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }));
  }
  
  // Configure file transports if enabled
  if (file) {
    // Remove existing file transports
    logger.transports = logger.transports.filter(
      t => t.name !== 'file'
    );
    
    // Custom log path or default
    const logPath = filePath || logsDir;
    
    // Ensure log directory exists
    if (!fs.existsSync(logPath)) {
      fs.mkdirSync(logPath, { recursive: true });
    }
    
    // Add configured file transports
    logger.add(new winston.transports.File({
      filename: path.join(logPath, 'error.log'),
      level: 'error',
      maxsize: maxSize,
      maxFiles
    }));
    
    logger.add(new winston.transports.File({
      filename: path.join(logPath, 'combined.log'),
      maxsize: maxSize,
      maxFiles
    }));
  }
  
  logger.info('Logging initialized', { level });
}