/**
 * Improved Conversation Management Module for FrankCode
 * 
 * Provides utilities for managing conversation history,
 * including clearing, compacting, and summarizing.
 */

const { logger } = require('../utils/logger');
const fs = require('fs').promises;
const path = require('path');

/**
 * Create a conversation manager
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.agent Agent instance
 * @param {Object} options.outputRenderer Output renderer
 * @param {Object} options.llm LLM client
 * @returns {Object} The conversation manager interface
 */
function createConversationManager({ agent, outputRenderer, llm }) {
  /**
   * Clear the conversation history
   * 
   * @returns {Promise<void>}
   */
  async function clearConversation() {
    try {
      // Reset the agent's conversation history
      agent.reset();
      
      // Clear the output renderer
      outputRenderer.clear();
      
      // Display confirmation message
      outputRenderer.addSystemMessage('Conversation history cleared.');
      
      logger.info('Conversation history cleared by user');
    } catch (error) {
      logger.error('Failed to clear conversation', { error });
      outputRenderer.addErrorMessage(`Error clearing conversation: ${error.message}`);
    }
  }
  
  /**
   * Compact the conversation by summarizing and clearing history
   * 
   * @returns {Promise<void>}
   */
  async function compactConversation() {
    try {
      // Get the current conversation history
      const history = agent.getConversationHistory();
      
      if (!history || history.length === 0) {
        outputRenderer.addSystemMessage('No conversation history to compact.');
        return;
      }
      
      outputRenderer.addSystemMessage('Compacting conversation...');
      
      // Generate a summary of the conversation
      const summary = await generateConversationSummary(history);
      
      // Reset the agent's conversation history but preserve system messages
      agent.reset(true);
      
      // Add the summary as system context
      agent.addSystemContext(summary);
      
      // Clear the output display
      outputRenderer.clear();
      
      // Display the summary with proper formatting
      outputRenderer.addSystemMessage('***Session Summary***');
      
      // Split the summary into lines and add with proper formatting
      const summaryLines = summary.split('\n');
      for (const line of summaryLines) {
        outputRenderer.addSystemMessage(line);
      }
      
      outputRenderer.addSystemMessage('\nConversation history has been compacted. The summary above has been retained in context.');
      
      logger.info('Conversation compacted with summary');
    } catch (error) {
      logger.error('Failed to compact conversation', { error });
      outputRenderer.addErrorMessage(`Error compacting conversation: ${error.message}`);
    }
  }
  
  /**
   * Generate a summary of the conversation history
   * 
   * @param {Array<Object>} history Conversation history
   * @returns {Promise<string>} Summary text
   */
  async function generateConversationSummary(history) {
    try {
      if (!history || history.length === 0) {
        return "No conversation history to summarize.";
      }
      
      // Filter out system messages and thinking tags
      const filteredHistory = history.filter(msg => 
        msg.role !== 'system'
      ).map(msg => ({
        role: msg.role,
        content: msg.content.replace(/<think>[\s\S]*?<\/think>/g, '') // Remove thinking tags
      }));
      
      if (filteredHistory.length === 0) {
        return "No user-assistant interactions to summarize.";
      }
      
      // Format the conversation history
      const formattedHistory = filteredHistory.map(msg => 
        `${msg.role.toUpperCase()}: ${msg.content.substring(0, 500)}${msg.content.length > 500 ? '...' : ''}`
      ).join('\n\n');
      
      // Create a prompt for the LLM to generate a summary
      const prompt = `I need you to create a structured summary of the following conversation between a user and FrankCode, an AI coding assistant. Format the summary like this example:

We've been working on a chemical process simulation application, focusing on fixing display and calculation issues in the heat exchanger components. Here's what we've accomplished:

**Main Issues Fixed:**

1. **Heat Exchanger Duty Display**
  - Fixed an issue where the heat exchanger duty wasn't updating correctly
  - Added forced numeric conversion to ensure duty values display correctly
  - Enhanced logging to trace calculation steps and verify values

2. **Property Display Improvements**
  - Added "Show Properties on Flowsheet" toggle in component configuration modals
  - Reverted flow rate formatting to maintain readability for large values
  - Fixed precision issues in numeric displays

**Main Files Modified:**

1. **/src/lib/simulation.ts**
  - Enhanced calculateHeatDuty() with better logging
  - Fixed the HeatExchangerNode.calculate() method
  - Added deep object cloning to avoid reference issues

**Next Steps:**

1. Test heat exchanger calculations with different flow rates
2. Verify connection configuration modal works as expected
3. Look into other components that may need similar display improvements

The most critical fixes were ensuring that changing boundary inflow properly propagates through the simulation.

Here's the conversation to summarize:

${formattedHistory}

Create a similar summary that:
1. Identifies the main topics and tasks discussed
2. Lists specific problems solved or features implemented 
3. Mentions key files or components that were modified or discussed
4. Suggests logical next steps based on the conversation
5. Highlights the most important achievements or insights

Format with clear sections using **bold** for headers and bullet points for details. Keep the summary concise but comprehensive.`;
      
      // Generate the summary using the LLM
      const response = await llm.generateResponse(prompt, {
        temperature: 0.3,  // Lower temperature for more factual summaries
        maxTokens: 1024    // Allow enough tokens for a detailed summary
      });
      
      return response.text;
    } catch (error) {
      logger.error('Failed to generate conversation summary', { error });
      return 'Failed to generate summary. The conversation history has still been cleared.';
    }
  }
  
  /**
   * Save the conversation to a file
   * 
   * @param {string} filePath File path
   * @returns {Promise<boolean>} Success indicator
   */
  async function saveConversation(filePath) {
    try {
      // Get the conversation history
      const history = agent.getConversationHistory();
      
      if (!history || history.length === 0) {
        outputRenderer.addSystemMessage('No conversation history to save.');
        return false;
      }
      
      // Format the conversation for saving
      const formattedHistory = history.map(msg => {
        // Remove thinking tags for cleaner export
        const cleanContent = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
        return `${msg.role.toUpperCase()}:\n${cleanContent}\n\n---\n\n`;
      }).join('');
      
      // Ensure directory exists
      const dirPath = path.dirname(filePath);
      await fs.mkdir(dirPath, { recursive: true }).catch(() => {});
      
      // Write to file
      await fs.writeFile(filePath, formattedHistory, 'utf8');
      
      outputRenderer.addSystemMessage(`Conversation saved to ${filePath}`);
      logger.info(`Conversation saved to ${filePath}`);
      
      return true;
    } catch (error) {
      logger.error('Failed to save conversation', { error });
      outputRenderer.addErrorMessage(`Error saving conversation: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Export conversation with summary
   * 
   * @param {string} filePath File path
   * @returns {Promise<boolean>} Success indicator
   */
  async function exportConversationWithSummary(filePath) {
    try {
      // Get the conversation history
      const history = agent.getConversationHistory();
      
      if (!history || history.length === 0) {
        outputRenderer.addSystemMessage('No conversation history to export.');
        return false;
      }
      
      // Generate a summary
      const summary = await generateConversationSummary(history);
      
      // Format the conversation for saving (remove thinking tags)
      const formattedHistory = history.map(msg => {
        const cleanContent = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
        return `${msg.role.toUpperCase()}:\n${cleanContent}\n\n---\n\n`;
      }).join('');
      
      // Combine summary and full history
      const content = `# Conversation Summary\n\n${summary}\n\n# Full Conversation\n\n${formattedHistory}`;
      
      // Ensure directory exists
      const dirPath = path.dirname(filePath);
      await fs.mkdir(dirPath, { recursive: true }).catch(() => {});
      
      // Write to file
      await fs.writeFile(filePath, content, 'utf8');
      
      outputRenderer.addSystemMessage(`Conversation with summary exported to ${filePath}`);
      logger.info(`Conversation exported to ${filePath}`);
      
      return true;
    } catch (error) {
      logger.error('Failed to export conversation', { error });
      outputRenderer.addErrorMessage(`Error exporting conversation: ${error.message}`);
      return false;
    }
  }
  
  // Return the conversation manager interface
  return {
    clearConversation,
    compactConversation,
    saveConversation,
    exportConversationWithSummary,
    generateConversationSummary
  };
}

module.exports = {
  createConversationManager
};