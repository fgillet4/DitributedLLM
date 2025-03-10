/**
 * Core Agent Implementation
 * 
 * Manages interactions with the LLM, processes project context,
 * and handles file modifications.
 */

const path = require('path');
const fs = require('fs').promises;
const { logger } = require('../utils');
const { manageContext } = require('./context');
const { countTokens } = require('./tokenizer');
const { 
  readFile, 
  writeFile, 
  listDirectoryFiles,
  checkFileExists
} = require('./fileManager');

/**
 * Create the agent instance
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.apiClient API client for LLM communication
 * @param {Object} options.tokenMonitor Token usage monitor
 * @param {Array} options.projectFiles Initial list of project files
 * @param {string} options.projectRoot Project root directory
 * @returns {Object} The agent interface
 */
function createAgent({ apiClient, tokenMonitor, projectFiles, projectRoot }) {
  // Initialize conversation history
  let conversationHistory = [];
  
  // Initialize file context
  let fileContexts = new Map();
  
  // Initialize the context manager
  const contextManager = manageContext({
    tokenMonitor,
    maxSize: tokenMonitor.getMaxTokens()
  });
  /**
   * Add system context to the conversation
   * 
   * @param {string} context System context to add
   */
  function addSystemContext(context) {
    try {
      // Add a system message to the conversation history
      conversationHistory.push({ 
        role: 'system', 
        content: context 
      });
      
      logger.debug('Added system context to conversation');
    } catch (error) {
      logger.error('Failed to add system context', { error });
    }
  }
  // Update the reset function to optionally preserve system context
  /**
   * Reset the agent state
   * 
   * @param {boolean} preserveSystemContext Whether to preserve system context messages
   */
  function reset(preserveSystemContext = false) {
    try {
      if (preserveSystemContext) {
        // Filter out non-system messages
        const systemMessages = conversationHistory.filter(msg => msg.role === 'system');
        conversationHistory = systemMessages;
      } else {
        conversationHistory = [];
      }
      
      contextManager.reset();
      logger.debug('Agent state reset');
    } catch (error) {
      logger.error('Failed to reset agent state', { error });
      // Ensure we at least clear the conversation if there's an error
      conversationHistory = [];
      contextManager.reset();
    }
  }
  
  /**
   * Load content of a file and add to context
   * 
   * @param {string} filePath Path to the file
   * @returns {Promise<string>} The file content
   */
  async function loadFileContext(filePath) {
    try {
      // Check if file exists
      const exists = await checkFileExists(filePath);
      if (!exists) {
        throw new Error(`File not found: ${filePath}`);
      }
      
      // Load file content
      const content = await readFile(filePath);
      
      // Calculate tokens
      const tokens = countTokens(content);
      
      // Store in context cache
      fileContexts.set(filePath, {
        content,
        tokens,
        lastModified: new Date()
      });
      
      // Add to context manager
      contextManager.addFileContext(filePath, content);
      
      return content;
    } catch (error) {
      logger.error(`Failed to load file context: ${filePath}`, { error });
      throw error;
    }
  }
  
  /**
   * Modify a file in the project
   * 
   * @param {string} filePath Path to the file
   * @param {string} newContent New file content
   * @returns {Promise<boolean>} Success indicator
   */
  async function modifyFile(filePath, newContent) {
    try {
      const fullPath = path.resolve(projectRoot, filePath);
      
      // Check if file exists
      const exists = await checkFileExists(fullPath);
      if (!exists) {
        throw new Error(`File not found: ${fullPath}`);
      }
      
      // Get original content for diff
      const originalContent = await readFile(fullPath);
      
      // Write new content
      await writeFile(fullPath, newContent);
      
      // Update context
      if (fileContexts.has(filePath)) {
        // Update cache
        fileContexts.set(filePath, {
          content: newContent,
          tokens: countTokens(newContent),
          lastModified: new Date()
        });
        
        // Update context manager
        contextManager.updateFileContext(filePath, newContent);
      }
      
      logger.info(`Modified file: ${filePath}`);
      return true;
    } catch (error) {
      logger.error(`Failed to modify file: ${filePath}`, { error });
      return false;
    }
  }
  
  /**
   * Scan for specific files in the project based on patterns
   * 
   * @param {string} pattern Glob pattern to match files
   * @returns {Promise<Array<string>>} Matching file paths
   */
  async function findFiles(pattern) {
    try {
      // Implementation would use something like fast-glob
      return [];
    } catch (error) {
      logger.error(`Failed to find files: ${pattern}`, { error });
      return [];
    }
  }
  
  /**
   * Process a user message and generate a response
   * 
   * @param {string} message User's message
   * @returns {Promise<Object>} Response object
   */
  async function processMessage(message) {
    try {
      // Add user message to conversation history
      conversationHistory.push({ role: 'user', content: message });
      
      // Get current context
      const context = contextManager.getCurrentContext();
      
      // Generate LLM prompt
      const prompt = generatePrompt(message, context);
      
      // Get response from LLM
      logger.debug('Sending prompt to LLM', { messageLength: message.length });
      const response = await apiClient.generateResponse(prompt);
      
      // Add assistant response to conversation history
      conversationHistory.push({ role: 'assistant', content: response.text });
      
      // Update token usage
      tokenMonitor.updateUsage(countTokens(response.text));
      
      // Check for file modifications in the response
      const fileModifications = parseFileModifications(response.text);
      
      // Return the processed response
      return {
        text: response.text,
        fileModifications,
        tokenUsage: tokenMonitor.getCurrentUsage()
      };
    } catch (error) {
      logger.error('Failed to process message', { error });
      throw error;
    }
  }
  
  /**
   * Generate a prompt for the LLM
   * 
   * @param {string} message User message
   * @param {Object} context Current context
   * @returns {string} Generated prompt
   */
  function generatePrompt(message, context) {
    // Basic prompt with system instruction and context
    const systemPrompt = `You are FrankCode, an AI coding assistant that helps developers understand and modify their codebase. 
You can read files and make changes to them directly. Always be concise and focus on providing practical solutions.
When modifying files, use the format:
\`\`\`file:path/to/file.js
// New content here
\`\`\`

Current project context:
${context.fileContexts.map(fc => `- ${fc.filePath} (${fc.summary})`).join('\n')}

Current conversation:
${conversationHistory.map(msg => `${msg.role.toUpperCase()}: ${msg.content}`).join('\n\n')}`;

    return `${systemPrompt}\n\nUSER: ${message}\n\nASSISTANT:`;
  }
  
  /**
   * Parse file modifications from LLM response
   * 
   * @param {string} response LLM response text
   * @returns {Array<Object>} Array of file modifications
   */
  function parseFileModifications(response) {
    const modifications = [];
    const fileBlockRegex = /```file:(.*?)\n([\s\S]*?)```/g;
    
    let match;
    while ((match = fileBlockRegex.exec(response)) !== null) {
      const filePath = match[1].trim();
      const content = match[2];
      
      modifications.push({
        filePath,
        content
      });
    }
    
    return modifications;
  }
  
  /**
   * Initialize agent context with project files
   * 
   * @returns {Promise<void>}
   */
  async function initialize() {
    logger.info('Initializing agent context...');
    
    // Load key project files
    for (const file of projectFiles.slice(0, 10)) {  // Start with first 10 for performance
      try {
        await loadFileContext(file);
      } catch (error) {
        // Continue with other files on error
        logger.warn(`Failed to load file: ${file}`, { error });
      }
    }
    
    logger.info('Agent context initialized');
  }
  
  // Initialize immediately
  initialize();
  
  // Return the agent interface
  return {
    processMessage,
    loadFileContext,
    modifyFile,
    findFiles,
    getContextManager: () => contextManager,
    getFileContexts: () => fileContexts,
    getConversationHistory: () => conversationHistory,
    addSystemContext,  // Add this
    reset,  // This should replace the existing reset function
    setApiClient: (newClient) => {
      apiClient = newClient;
    }
  };
}

module.exports = {
  createAgent
};