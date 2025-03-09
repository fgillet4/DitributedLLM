/**
 * Tokenizer module for token counting
 * 
 * Provides utilities for counting tokens in text using tiktoken
 * This is important for managing context windows with LLMs
 */

let tiktoken;
try {
  tiktoken = require('tiktoken');
} catch (error) {
  // Silent fail - don't log anything here
}

// Cache encoding to avoid recreating it on each call
let encoding;
let logger;

/**
 * Initialize the tokenizer
 * 
 * @param {string} model Model name for tokenization (default is cl100k, used by many models)
 * @returns {void}
 */
function initTokenizer(model = 'cl100k_base') {
  try {
    // Only import logger after we're initialized to avoid circular dependencies
    if (!logger) {
      logger = require('../utils/logger').logger;
    }
    
    if (tiktoken) {
      encoding = tiktoken.getEncoding(model);
      logger.debug(`Tokenizer initialized with model: ${model}`);
    } else {
      // Fallback when tiktoken is not available - don't log warning
      logger.debug('Using fallback tokenization method - tiktoken not available');
      encoding = null;
    }
  } catch (error) {
    // Log error to file but don't throw - silently use fallback
    if (logger) {
      logger.debug('Failed to initialize tokenizer, using fallback', { error });
    }
    
    encoding = null;
  }
}

/**
 * Count tokens in a text string
 * 
 * @param {string} text Text to count tokens for
 * @returns {number} Number of tokens
 */
function countTokens(text) {
  if (!text) return 0;
  
  try {
    // If encoding is not initialized, initialize it
    if (!encoding) {
      initTokenizer();
    }
    
    // If we have tiktoken encoding, use it
    if (encoding) {
      return encoding.encode(text).length;
    }
    
    // Fallback tokenization (approximate) - don't log warning
    // This is a simplistic approximation - in practice, real tokenization is more complex
    return Math.ceil(text.length / 4);
  } catch (error) {
    if (logger) {
      logger.debug('Error counting tokens, using fallback', { error });
    }
    
    // Return a conservative estimate
    return Math.ceil(text.length / 3);
  }
}

/**
 * Truncate text to a maximum number of tokens
 * 
 * @param {string} text Text to truncate
 * @param {number} maxTokens Maximum number of tokens
 * @returns {string} Truncated text
 */
function truncateToTokens(text, maxTokens) {
  if (!text) return '';
  
  try {
    // If encoding is not initialized, initialize it
    if (!encoding) {
      initTokenizer();
    }
    
    // If we have tiktoken encoding, use it
    if (encoding) {
      const tokens = encoding.encode(text);
      
      if (tokens.length <= maxTokens) {
        return text;
      }
      
      const truncatedTokens = tokens.slice(0, maxTokens);
      return encoding.decode(truncatedTokens);
    }
    
    // Fallback truncation (approximate) - no warning log
    const approxTokens = countTokens(text);
    
    if (approxTokens <= maxTokens) {
      return text;
    }
    
    const ratio = maxTokens / approxTokens;
    const charLimit = Math.floor(text.length * ratio);
    
    // Try to cut at a sentence or paragraph boundary
    const truncated = text.substring(0, charLimit);
    
    // Look for a good breakpoint near the end
    const breakpoints = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', '];
    
    for (const breakpoint of breakpoints) {
      const lastIdx = truncated.lastIndexOf(breakpoint);
      if (lastIdx !== -1 && lastIdx > charLimit * 0.8) {
        return truncated.substring(0, lastIdx + breakpoint.length) + '...';
      }
    }
    
    // If no good breakpoint, just cut and add ellipsis
    return truncated + '...';
  } catch (error) {
    if (logger) {
      logger.debug('Error truncating text to tokens, using simple fallback', { error });
    }
    
    // Return a conservatively truncated text
    const approxCharPerToken = 4;
    const charLimit = maxTokens * approxCharPerToken;
    return text.substring(0, charLimit) + '...';
  }
}

// Lazy initialization - initialize when first used

module.exports = {
  countTokens,
  truncateToTokens,
  initTokenizer
};