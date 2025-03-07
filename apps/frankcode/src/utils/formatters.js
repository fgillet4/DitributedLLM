/**
 * Formatting utilities
 * 
 * Functions for formatting data in the UI
 */

/**
 * Format a timestamp
 * 
 * @param {Date} date Date object
 * @returns {string} Formatted timestamp
 */
function formatTimestamp(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  
  /**
   * Format a file size
   * 
   * @param {number} bytes Size in bytes
   * @param {number} decimals Decimal places
   * @returns {string} Formatted size
   */
  function formatFileSize(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }
  
  /**
   * Format a duration
   * 
   * @param {number} ms Duration in milliseconds
   * @returns {string} Formatted duration
   */
  function formatDuration(ms) {
    if (ms < 1000) {
      return `${ms}ms`;
    }
    
    const seconds = Math.floor(ms / 1000);
    
    if (seconds < 60) {
      return `${seconds}s`;
    }
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes < 60) {
      return `${minutes}m ${remainingSeconds}s`;
    }
    
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    
    return `${hours}h ${remainingMinutes}m`;
  }
  
  /**
   * Format a percentage
   * 
   * @param {number} value Value
   * @param {number} total Total
   * @param {number} decimals Decimal places
   * @returns {string} Formatted percentage
   */
  function formatPercentage(value, total, decimals = 1) {
    if (total === 0) return '0%';
    
    const percentage = (value / total) * 100;
    return percentage.toFixed(decimals) + '%';
  }
  
  /**
   * Truncate a string
   * 
   * @param {string} str String to truncate
   * @param {number} length Maximum length
   * @param {string} suffix Suffix to add
   * @returns {string} Truncated string
   */
  function truncate(str, length = 50, suffix = '...') {
    if (!str) return '';
    if (str.length <= length) return str;
    
    return str.substring(0, length - suffix.length) + suffix;
  }
  
  /**
   * Format a code snippet for display
   * 
   * @param {string} code Code snippet
   * @param {number} maxLines Maximum lines
   * @returns {string} Formatted code
   */
  function formatCodeSnippet(code, maxLines = 10) {
    if (!code) return '';
    
    const lines = code.split('\n');
    
    if (lines.length <= maxLines) {
      return code;
    }
    
    return lines.slice(0, maxLines).join('\n') + `\n... (${lines.length - maxLines} more lines)`;
  }
  
  module.exports = {
    formatTimestamp,
    formatFileSize,
    formatDuration,
    formatPercentage,
    truncate,
    formatCodeSnippet
  };