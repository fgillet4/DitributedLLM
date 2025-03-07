/**
 * Utilities module entry point
 * 
 * Exports utility functions and objects
 */

const { logger, setupLogging } = require('./logger');
const { loadConfig, saveConfig } = require('./config');
const { scanProjectFiles, ensureDir, pathExists, getStats, getFileType } = require('./fileSystem');
const { createGitUtils } = require('./git');
const { formatTimestamp, formatFileSize, formatDuration, formatPercentage, truncate } = require('./formatters');
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
  createTokenMonitor
};