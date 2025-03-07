/**
 * Agent module entry point
 * 
 * Exports the agent creation function and other utilities
 */

const { createAgent } = require('./agent');
const { manageContext } = require('./context');
const { countTokens, truncateToTokens } = require('./tokenizer');
const fileManager = require('./fileManager');
const { logger } = require('../utils');

/**
 * Initialize an agent instance
 * 
 * @param {Object} options Configuration options
 * @returns {Object} The agent instance
 */
function initAgent(options) {
  logger.info('Initializing agent...');
  
  const agent = createAgent(options);
  
  return agent;
}

module.exports = {
  initAgent,
  createAgent,
  manageContext,
  countTokens,
  truncateToTokens,
  fileManager
};