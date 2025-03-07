/**
 * Token Monitor
 * 
 * Tracks token usage for context management
 */

const { logger } = require('./logger');
const { EventEmitter } = require('events');

/**
 * Create a token monitor
 * 
 * @param {Object} options Configuration options
 * @param {number} options.maxTokens Maximum tokens
 * @param {number} options.warningThreshold Warning threshold (0-1)
 * @returns {Object} The token monitor interface
 */
function createTokenMonitor(options = {}) {
  const {
    maxTokens = 8192,
    warningThreshold = 0.8
  } = options;
  
  // Current token usage
  let currentTokens = 0;
  
  // Create event emitter
  const events = new EventEmitter();
  
  /**
   * Update token usage
   * 
   * @param {number} tokens Number of tokens
   */
  function updateUsage(tokens) {
    // Calculate new total
    const newTotal = currentTokens + tokens;
    
    // Check if we're going over the limit
    if (newTotal > maxTokens) {
      logger.warn(`Token limit exceeded: ${newTotal}/${maxTokens}`);
      events.emit('limit-exceeded', { current: newTotal, max: maxTokens });
    } else if (newTotal / maxTokens >= warningThreshold && currentTokens / maxTokens < warningThreshold) {
      // Emit warning if we cross the threshold
      logger.warn(`Token usage approaching limit: ${newTotal}/${maxTokens}`);
      events.emit('warning-threshold', { current: newTotal, max: maxTokens });
    }
    
    // Update current tokens
    currentTokens = newTotal;
    
    // Emit update event
    events.emit('usage-updated', { current: currentTokens, max: maxTokens });
  }
  
  /**
   * Set token usage directly
   * 
   * @param {number} tokens Number of tokens
   */
  function setUsage(tokens) {
    // Check if we're going over the limit
    if (tokens > maxTokens) {
      logger.warn(`Token limit exceeded: ${tokens}/${maxTokens}`);
      events.emit('limit-exceeded', { current: tokens, max: maxTokens });
    } else if (tokens / maxTokens >= warningThreshold && currentTokens / maxTokens < warningThreshold) {
      // Emit warning if we cross the threshold
      logger.warn(`Token usage approaching limit: ${tokens}/${maxTokens}`);
      events.emit('warning-threshold', { current: tokens, max: maxTokens });
    }
    
    // Update current tokens
    currentTokens = tokens;
    
    // Emit update event
    events.emit('usage-updated', { current: currentTokens, max: maxTokens });
  }
  
  /**
   * Reset token usage
   */
  function reset() {
    currentTokens = 0;
    events.emit('usage-updated', { current: currentTokens, max: maxTokens });
    logger.debug('Token usage reset');
  }
  
  /**
   * Get current token usage
   * 
   * @returns {number} Current token usage
   */
  function getCurrentUsage() {
    return currentTokens;
  }
  
  /**
   * Get maximum tokens
   * 
   * @returns {number} Maximum tokens
   */
  function getMaxTokens() {
    return maxTokens;
  }
  
  /**
   * Get usage ratio
   * 
   * @returns {number} Usage ratio (0-1)
   */
  function getUsageRatio() {
    return currentTokens / maxTokens;
  }
  
  /**
   * Check if nearing limit
   * 
   * @returns {boolean} Whether nearing limit
   */
  function isNearingLimit() {
    return getUsageRatio() >= warningThreshold;
  }
  
  /**
   * Add an event listener
   * 
   * @param {string} event Event name
   * @param {Function} listener Event listener
   */
  function on(event, listener) {
    events.on(event, listener);
  }
  
  /**
   * Remove an event listener
   * 
   * @param {string} event Event name
   * @param {Function} listener Event listener
   */
  function off(event, listener) {
    events.off(event, listener);
  }
  
  // Return the token monitor interface
  return {
    updateUsage,
    setUsage,
    reset,
    getCurrentUsage,
    getMaxTokens,
    getUsageRatio,
    isNearingLimit,
    on,
    off
  };
}

module.exports = {
  createTokenMonitor
};