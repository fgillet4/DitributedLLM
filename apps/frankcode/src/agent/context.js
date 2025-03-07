/**
 * Context Management for the Agent
 * 
 * Handles the active context window, ensuring token limits are respected
 * and the most relevant information is included.
 */

const { logger } = require('../utils');
const { countTokens } = require('./tokenizer');

/**
 * Create a context manager
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.tokenMonitor Token usage monitor
 * @param {number} options.maxSize Maximum context size in tokens
 * @returns {Object} The context manager interface
 */
function manageContext({ tokenMonitor, maxSize }) {
  // Context storage
  const fileContexts = new Map();
  
  // Track total tokens used by context
  let totalContextTokens = 0;
  
  // LRU tracking for context eviction
  const fileAccessTimes = new Map();
  
  /**
   * Add a file to the context
   * 
   * @param {string} filePath Path to the file
   * @param {string} content File content
   * @returns {boolean} Success indicator
   */
  function addFileContext(filePath, content) {
    try {
      // Calculate tokens for this file content
      const tokens = countTokens(content);
      
      // Check if this will exceed our max context
      if (totalContextTokens + tokens > maxSize) {
        // Need to make room
        makeRoomInContext(tokens);
      }
      
      // Generate a summary for the file
      const summary = summarizeFile(filePath, content);
      
      // Add to context
      fileContexts.set(filePath, {
        content,
        tokens,
        summary
      });
      
      // Update total tokens
      totalContextTokens += tokens;
      
      // Update access time for LRU tracking
      fileAccessTimes.set(filePath, Date.now());
      
      logger.debug(`Added file to context: ${filePath} (${tokens} tokens)`);
      return true;
    } catch (error) {
      logger.error(`Failed to add file to context: ${filePath}`, { error });
      return false;
    }
  }
  
  /**
   * Update an existing file in context
   * 
   * @param {string} filePath Path to the file
   * @param {string} newContent New file content
   * @returns {boolean} Success indicator
   */
  function updateFileContext(filePath, newContent) {
    try {
      // Check if file exists in context
      if (!fileContexts.has(filePath)) {
        return addFileContext(filePath, newContent);
      }
      
      // Get existing context
      const existing = fileContexts.get(filePath);
      
      // Calculate new tokens
      const newTokens = countTokens(newContent);
      
      // Update total token count
      totalContextTokens = totalContextTokens - existing.tokens + newTokens;
      
      // If we exceed max, need to make room
      if (totalContextTokens > maxSize) {
        makeRoomInContext(0, [filePath]);  // Don't evict the file we're updating
      }
      
      // Generate a new summary
      const summary = summarizeFile(filePath, newContent);
      
      // Update context
      fileContexts.set(filePath, {
        content: newContent,
        tokens: newTokens,
        summary
      });
      
      // Update access time
      fileAccessTimes.set(filePath, Date.now());
      
      logger.debug(`Updated file in context: ${filePath} (${newTokens} tokens)`);
      return true;
    } catch (error) {
      logger.error(`Failed to update file in context: ${filePath}`, { error });
      return false;
    }
  }
  
  /**
   * Remove a file from context
   * 
   * @param {string} filePath Path to the file
   * @returns {boolean} Success indicator
   */
  function removeFileContext(filePath) {
    try {
      // Check if file exists in context
      if (!fileContexts.has(filePath)) {
        return false;
      }
      
      // Get existing context
      const existing = fileContexts.get(filePath);
      
      // Update total token count
      totalContextTokens -= existing.tokens;
      
      // Remove from context
      fileContexts.delete(filePath);
      
      // Remove from access times
      fileAccessTimes.delete(filePath);
      
      logger.debug(`Removed file from context: ${filePath}`);
      return true;
    } catch (error) {
      logger.error(`Failed to remove file from context: ${filePath}`, { error });
      return false;
    }
  }
  
  /**
   * Make room in the context by evicting least recently used files
   * 
   * @param {number} neededTokens Tokens needed for new content
   * @param {Array<string>} excludeFiles Files to exclude from eviction
   */
  function makeRoomInContext(neededTokens, excludeFiles = []) {
    // Continue evicting until we have enough room
    while (totalContextTokens + neededTokens > maxSize && fileContexts.size > 0) {
      // Create array of [filePath, lastAccessTime] for sorting
      const accessEntries = Array.from(fileAccessTimes.entries())
        .filter(([filePath]) => !excludeFiles.includes(filePath));
      
      // Sort by access time (oldest first)
      accessEntries.sort((a, b) => a[1] - b[1]);
      
      // If no files can be evicted, break
      if (accessEntries.length === 0) break;
      
      // Evict the oldest accessed file
      const [oldestFilePath] = accessEntries[0];
      
      // Remove it from context
      const evictedContext = fileContexts.get(oldestFilePath);
      fileContexts.delete(oldestFilePath);
      fileAccessTimes.delete(oldestFilePath);
      
      // Update token count
      totalContextTokens -= evictedContext.tokens;
      
      logger.debug(`Evicted file from context: ${oldestFilePath} (${evictedContext.tokens} tokens)`);
    }
    
    // If we still don't have enough room, log a warning
    if (totalContextTokens + neededTokens > maxSize) {
      logger.warn(`Context window capacity exceeded - some context may be lost`);
    }
  }
  
  /**
   * Get the current context for use in prompts
   * 
   * @returns {Object} The current context
   */
  function getCurrentContext() {
    return {
      fileContexts: Array.from(fileContexts.entries()).map(([filePath, context]) => ({
        filePath,
        summary: context.summary,
        content: context.content
      })),
      totalTokens: totalContextTokens
    };
  }
  
  /**
   * Reset the context manager, clearing all context
   */
  function reset() {
    fileContexts.clear();
    fileAccessTimes.clear();
    totalContextTokens = 0;
    logger.debug('Context manager reset');
  }
  
  /**
   * Generate a brief summary of a file
   * 
   * @param {string} filePath Path to the file
   * @param {string} content File content
   * @returns {string} File summary
   */
  function summarizeFile(filePath, content) {
    // Extract file extension
    const ext = filePath.split('.').pop().toLowerCase();
    
    // Simple summary based on file type
    if (['js', 'ts', 'jsx', 'tsx'].includes(ext)) {
      // JavaScript/TypeScript files: count functions, classes, imports
      const functionCount = (content.match(/function\s+\w+/g) || []).length;
      const arrowFunctionCount = (content.match(/const\s+\w+\s*=\s*(\(.*?\)|[^=]*?)\s*=>/g) || []).length;
      const classCount = (content.match(/class\s+\w+/g) || []).length;
      const importCount = (content.match(/import\s+/g) || []).length;
      
      return `JS file with ${functionCount + arrowFunctionCount} functions, ${classCount} classes, ${importCount} imports`;
    } else if (['json'].includes(ext)) {
      // JSON files: try to identify type from content or keys
      if (content.includes('"dependencies"') && content.includes('"name"')) {
        return 'Package configuration file';
      } else {
        return 'JSON configuration file';
      }
    } else if (['md', 'markdown'].includes(ext)) {
      // Markdown files: count headers
      const h1Count = (content.match(/^#\s+.+$/gm) || []).length;
      const h2Count = (content.match(/^##\s+.+$/gm) || []).length;
      
      return `Markdown document with ${h1Count} main sections, ${h2Count} subsections`;
    } else if (['html', 'htm'].includes(ext)) {
      return 'HTML document';
    } else if (['css', 'scss', 'sass', 'less'].includes(ext)) {
      return 'Stylesheet file';
    } else {
      // Default
      const lines = content.split('\n').length;
      return `File with ${lines} lines`;
    }
  }
  
  // Return the context manager interface
  return {
    addFileContext,
    updateFileContext,
    removeFileContext,
    getCurrentContext,
    reset,
    getTotalTokens: () => totalContextTokens
  };
}

module.exports = {
  manageContext
};