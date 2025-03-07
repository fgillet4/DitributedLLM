/**
 * Status Bar Component
 * 
 * Displays information about the current state of the application
 */

const chalk = require('chalk');
const { logger } = require('../utils');

/**
 * Create a status bar component
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.widget The status bar widget
 * @param {Object} options.apiClient API client
 * @param {Object} options.tokenMonitor Token monitoring utility
 * @returns {Object} The status bar interface
 */
function createStatusBar({ widget, apiClient, tokenMonitor }) {
  // Status message
  let statusMessage = 'Ready';
  
  // Update interval handle
  let updateInterval = null;
  
  /**
   * Initialize the status bar
   */
  function init() {
    // Update immediately
    updateStatus();
    
    // Set up update interval (every 5 seconds)
    updateInterval = setInterval(() => {
      updateStatus();
    }, 5000);
  }
  
  /**
   * Update the status bar
   */
  function updateStatus() {
    try {
      // Get connection status
      const connected = apiClient.getConnectionStatus();
      
      // Get token usage
      const tokenUsage = tokenMonitor.getCurrentUsage();
      const tokenMax = tokenMonitor.getMaxTokens();
      const tokenPercentage = Math.floor((tokenUsage / tokenMax) * 100);
      
      // Format the token meter
      const tokenMeter = formatTokenMeter(tokenUsage, tokenMax);
      
      // Create status bar content
      const content = [
        // Left side
        `${chalk.bold('Status:')} ${statusMessage}`,
        
        // Center
        `Model: ${apiClient.getConnectionStatus() ? chalk.green('Connected') : chalk.red('Disconnected')}`,
        
        // Right side
        `Tokens: ${tokenMeter} ${tokenUsage}/${tokenMax} (${tokenPercentage}%)`
      ].join('    ');
      
      // Set widget content
      widget.setContent(content);
      
      // Check if token usage is nearing limit
      if (tokenPercentage > 80) {
        widget.style.bg = 'red';
      } else if (tokenPercentage > 60) {
        widget.style.bg = 'yellow';
      } else {
        widget.style.bg = 'blue';
      }
      
      // Render the screen
      widget.screen.render();
    } catch (error) {
      logger.error('Failed to update status bar', { error });
    }
  }
  
  /**
   * Format a token usage meter
   * 
   * @param {number} used Tokens used
   * @param {number} max Maximum tokens
   * @returns {string} Formatted meter
   */
  function formatTokenMeter(used, max) {
    const width = 10;
    const filled = Math.floor((used / max) * width);
    const empty = width - filled;
    
    let color = chalk.green;
    if (used / max > 0.8) {
      color = chalk.red;
    } else if (used / max > 0.6) {
      color = chalk.yellow;
    }
    
    const meter = color('[' + '='.repeat(filled) + ' '.repeat(empty) + ']');
    
    return meter;
  }
  
  /**
   * Update the status message
   * 
   * @param {string} message New status message
   */
  function update(message) {
    statusMessage = message;
    updateStatus();
  }
  
  /**
   * Destroy the status bar component
   */
  function destroy() {
    if (updateInterval) {
      clearInterval(updateInterval);
      updateInterval = null;
    }
  }
  
  // Initialize immediately
  init();
  
  // Return the status bar interface
  return {
    update,
    updateStatus,
    destroy
  };
}

module.exports = {
  createStatusBar
};