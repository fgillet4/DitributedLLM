/**
 * Utilities module entry point
 * 
 * Exports utility functions and objects
 */

// Note: Order of imports is important to avoid circular dependencies
const { logger, setupLogging } = require('./logger');
const { formatTimestamp, formatFileSize, formatDuration, formatPercentage, truncate } = require('./formatters');
const { loadConfig, saveConfig } = require('./config');
const { scanProjectFiles, ensureDir, pathExists, getStats, getFileType } = require('./fileSystem');
const { createGitUtils } = require('./git');
const { createTokenMonitor } = require('./tokenMonitor');

module.exports = {
  logger,
  setupLogging,
  loadConfig,
  saveConfig,
  scanProjectFiles,
  ensureDir,
  pathExists,
  getStats,
  getFileType,
  createGitUtils,
  formatTimestamp,
  formatFileSize,
  formatDuration,
  formatPercentage,
  truncate,
  createTokenMonitor,
  fileOperations: require('./fileOperations')

};