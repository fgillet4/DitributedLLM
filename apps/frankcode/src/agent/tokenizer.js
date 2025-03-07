/**
 * Tokenizer module for token counting
 * 
 * Provides utilities for counting tokens in text using tiktoken
 * This is important for managing context windows with LLMs
 */

const tiktoken = require('tiktoken');
const logger = require('../utils/logger');

// Cache encoding to avoid recreating it on each call
let encoding;

/**
 * Initialize the tokenizer
 * 
 * @param {string} model Model name for tokenization (default is cl100k, used by many models)
 * @returns {void}
 */
function initTokenizer(model = 'cl100k_base') {
  try {
    encoding = tiktoken.getEncoding(model);
    logger.debug(`Tokenizer initialized with model: ${model}`);
  } catch (error) {
    logger.error('Failed to initialize tokenizer', { error });
    
    // Fallback to a simple tokenization method
    logger.warn('Using fallback tokenization method');
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
    
    // Fallback tokenization (approximate)
    // This is a simplistic approximation - in practice, real tokenization is more complex
    return Math.ceil(text.length / 4);
  } catch (error) {
    logger.error('Error counting tokens', { error });
    
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
    
    // Fallback truncation (approximate)
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
    logger.error('Error truncating text to tokens', { error });
    
    // Return a conservatively truncated text
    const approxCharPerToken = 4;
    const charLimit = maxTokens * approxCharPerToken;
    return text.substring(0, charLimit) + '...';
  }
}

// Initialize tokenizer on module load
initTokenizer();

module.exports = {
  countTokens,
  truncateToTokens,
  initTokenizer
};