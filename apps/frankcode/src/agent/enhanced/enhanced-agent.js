/**
 * Enhanced Agent for FrankCode
 * 
 * This module integrates various AI capabilities (RAG, GraphRAG, Memory)
 * to create a more capable autonomous code editing agent.
 */

const path = require('path');
const fs = require('fs').promises;
const { createCodeRAG } = require('./rag');
const { createGraphRAG } = require('./graphrag');
const { createMemoryController } = require('./memory-controller');
const { createVectorStore } = require('./embeddings');
const { logger } = require('../utils/logger');
const { countTokens } = require('./tokenizer');

/**
 * Create an enhanced agent for code editing
 * 
 * @param {Object} options Configuration options
 * @param {string} options.projectRoot Project root directory
 * @param {Object} options.apiClient API client for LLM
 * @param {Object} options.tokenMonitor Token usage monitor
 * @returns {Object} The enhanced agent interface
 */
function createEnhancedAgent({ projectRoot, apiClient, tokenMonitor }) {
  // Set up RAG systems
  const storageDir = path.join(projectRoot, '.frankcode');
  
  // Create RAG systems
  const codeVectorStore = createVectorStore({
    storePath: path.join(storageDir, 'code-vectors.json')
  });
  
  const codeRAG = createCodeRAG({
    projectRoot,
    vectorStore: codeVectorStore
  });
  
  const graphRAG = createGraphRAG({
    projectRoot,
    vectorStore: codeVectorStore
  });
  
  // Create memory controller
  const memoryController = createMemoryController({
    memoryPath: path.join(storageDir, 'memory.json'),
    maxMemories: 1000,
    relevanceThreshold: 0.7
  });
  
  // Track indexing status
  let isIndexed = false;
  let isIndexing = false;
  
  /**
   * Initialize the enhanced agent
   * 
   * @returns {Promise<void>}
   */
  async function initialize() {
    try {
      // Create storage directory
      await fs.mkdir(storageDir, { recursive: true });
      
      // Try to load existing GraphRAG data
      const graphLoaded = await graphRAG.loadGraph(path.join(storageDir, 'code-graph.json'));
      
      if (graphLoaded) {
        isIndexed = true;
        logger.info('GraphRAG data loaded from disk');
      }
    } catch (error) {
      logger.error(`Failed to initialize enhanced agent: ${error.message}`, { error });
    }
  }
  
  /**
   * Index the project for RAG and GraphRAG
   * 
   * @param {Object} options Indexing options
   * @returns {Promise<boolean>} Success indicator
   */
  async function indexProject(options = {}) {
    if (isIndexing) {
      logger.warn('Project indexing already in progress');
      return false;
    }
    
    try {
      isIndexing = true;
      
      // Index with CodeRAG
      logger.info('Indexing project with CodeRAG...');
      const codeRAGCount = await codeRAG.indexProject(options);
      
      // Index with GraphRAG
      logger.info('Indexing project with GraphRAG...');
      const graphRAGCount = await graphRAG.indexProject(options);
      
      isIndexed = true;
      isIndexing = false;
      
      logger.info(`Indexing complete. Indexed ${codeRAGCount} files with CodeRAG and ${graphRAGCount} files with GraphRAG`);
      return true;
    } catch (error) {
      isIndexing = false;
      logger.error(`Failed to index project: ${error.message}`, { error });
      return false;
    }
  }
  
  /**
   * Process a user message with context augmentation
   * 
   * @param {string} message User's message
   * @returns {Promise<Object>} Response object
   */
  async function processMessage(message) {
    try {
      // Check if we need to index the project
      if (!isIndexed && !isIndexing && !message.startsWith('/')) {
        logger.info('Project not indexed. Starting background indexing...');
        // Start indexing in the background
        indexProject().catch(error => {
          logger.error(`Background indexing failed: ${error.message}`, { error });
        });
      }
      
      // Extract the task from the message
      const task = extractTask(message);
      
      // Skip context augmentation for commands
      if (message.startsWith('/')) {
        return { 
          text: message,
          augmented: false
        };
      }
      
      // Get relevant memories
      logger.debug('Retrieving relevant memories...');
      const relevantMemories = await memoryController.getRelevantMemories(task);
      
      // Get relevant code context
      logger.debug('Retrieving relevant code context...');
      const codeContext = isIndexed ? 
        await codeRAG.retrieveContext(task, { maxResults: 3 }) : 
        [];
      
      // Get relevant graph context if task involves code relationships
      logger.debug('Retrieving relevant graph context...');
      const graphContext = isIndexed ? 
        await graphRAG.search(task, { maxResults: 3 }) : 
        [];
      
      // Format contexts
      const memoriesText = memoryController.formatMemoriesForPrompt(relevantMemories);
      const codeContextText = codeRAG.formatContextForPrompt(codeContext);
      const graphContextText = graphRAG.formatSearchResultsForPrompt(graphContext);
      
      // Combine contexts
      let augmentedContext = '';
      
      if (memoriesText) augmentedContext += memoriesText + '\n\n';
      if (codeContextText) augmentedContext += codeContextText + '\n\n';
      if (graphContextText) augmentedContext += graphContextText + '\n\n';
      
      // Truncate to avoid token limits
      if (augmentedContext.length > 0) {
        const maxContextTokens = 3000; // Should be adjusted based on model
        const tokenCount = countTokens(augmentedContext);
        
        if (tokenCount > maxContextTokens) {
          logger.debug(`Truncating context from ${tokenCount} to ${maxContextTokens} tokens`);
          augmentedContext = truncateToTokens(augmentedContext, maxContextTokens);
        }
        
        // Add the augmented context to the message
        augmentedContext += `\n\nUser message: ${message}`;
        
        logger.debug('Message augmented with context');
        return {
          text: augmentedContext,
          originalMessage: message,
          augmented: true
        };
      }
      
      // No context to add
      return {
        text: message,
        augmented: false
      };
    } catch (error) {
      logger.error(`Failed to process message: ${error.message}`, { error });
      return {
        text: message,
        error: error.message,
        augmented: false
      };
    }
  }
  
  /**
   * Extract a task from a message
   * 
   * @param {string} message User message
   * @returns {string} Extracted task
   */
  function extractTask(message) {
    // For now, just return the message as is
    // In a more sophisticated implementation, this could extract
    // the core task or intent from the message
    return message;
  }
  
  /**
   * Learn from user feedback
   * 
   * @param {string} context The context or task
   * @param {string} feedback User's feedback
   * @param {Object} metadata Additional metadata
   * @returns {Promise<boolean>} Success indicator
   */
  async function learnFromFeedback(context, feedback, metadata = {}) {
    try {
      // Determine memory type based on context
      let memoryType = '';
      
      if (context.toLowerCase().includes('code pattern') || 
          context.toLowerCase().includes('structure') ||
          context.toLowerCase().includes('format')) {
        await memoryController.rememberCodePattern(context, feedback, metadata);
      } else if (context.toLowerCase().includes('prefer') ||
                 context.toLowerCase().includes('like') ||
                 context.toLowerCase().includes('want')) {
        await memoryController.rememberUserPreference(context, feedback, metadata);
      } else {
        await memoryController.rememberProblemSolution(context, feedback, metadata);
      }
      
      logger.info('Learned from user feedback');
      return true;
    } catch (error) {
      logger.error(`Failed to learn from feedback: ${error.message}`, { error });
      return false;
    }
  }
  
  /**
   * Generate an LLM prompt with appropriate context
   * 
   * @param {string} task The task description
   * @param {Object} options Prompt options
   * @returns {Promise<string>} Generated prompt
   */
  async function generatePrompt(task, options = {}) {
    try {
      const { 
        includeMemories = true,
        includeCodeContext = true,
        includeGraphContext = true,
        systemInstruction = null
      } = options;
      
      // Start with system instruction if provided
      let prompt = '';
      
      if (systemInstruction) {
        prompt += systemInstruction + '\n\n';
      } else {
        prompt += 'You are FrankCode, an autonomous code editing agent. You can help users write, modify, and understand code. Please follow these guidelines:\n\n';
        prompt += '1. When modifying files, clearly show the changes and ask for confirmation\n';
        prompt += '2. When creating new files, show the full content and ask for confirmation\n';
        prompt += '3. Use consistent code patterns that match the existing codebase\n';
        prompt += '4. Provide explanations for your changes\n';
        prompt += '5. Remember user preferences for future interactions\n\n';
      }
      
      // Add context
      let contextAdded = false;
      
      // Add memories if available and requested
      if (includeMemories) {
        const relevantMemories = await memoryController.getRelevantMemories(task);
        if (relevantMemories.length > 0) {
          prompt += memoryController.formatMemoriesForPrompt(relevantMemories) + '\n\n';
          contextAdded = true;
        }
      }
      
      // Add code context if available and requested
      if (includeCodeContext && isIndexed) {
        const codeContext = await codeRAG.retrieveContext(task, { maxResults: 3 });
        if (codeContext.length > 0) {
          prompt += codeRAG.formatContextForPrompt(codeContext) + '\n\n';
          contextAdded = true;
        }
      }
      
      // Add graph context if available and requested
      if (includeGraphContext && isIndexed) {
        const graphContext = await graphRAG.search(task, { maxResults: 3 });
        if (graphContext.length > 0) {
          prompt += graphRAG.formatSearchResultsForPrompt(graphContext) + '\n\n';
          contextAdded = true;
        }
      }
      
      // Add the task
      if (contextAdded) {
        prompt += 'Based on the context above, please address the following:\n\n';
      }
      
      prompt += task;
      
      return prompt;
    } catch (error) {
      logger.error(`Failed to generate prompt: ${error.message}`, { error });
      return task; // Fallback to just the task
    }
  }
  
  /**
   * Process a file modification request
   * 
   * @param {Object} modification File modification data
   * @param {string} modification.filePath Path to the file
   * @param {string} modification.content New content
   * @returns {Promise<Object>} Processing result
   */
  async function processFileModification(modification) {
    try {
      // Get original file content if it exists
      let originalContent = '';
      try {
        originalContent = await fs.readFile(path.join(projectRoot, modification.filePath), 'utf8');
      } catch (error) {
        // File doesn't exist, treat as a new file
        originalContent = '';
      }
      
      // Check if this is a pattern we should remember
      if (originalContent) {
        // This is an existing file being modified
        await memoryController.rememberCodePattern(
          `Code pattern for file: ${modification.filePath}`,
          originalContent,
          { filePath: modification.filePath, type: 'file_structure' }
        );
      }
      
      // Return the modification with additional context
      return {
        ...modification,
        isNewFile: !originalContent,
        originalContent
      };
    } catch (error) {
      logger.error(`Failed to process file modification: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * Truncate text to fit within token limit
   * 
   * @param {string} text Text to truncate
   * @param {number} maxTokens Maximum number of tokens
   * @returns {string} Truncated text
   */
  function truncateToTokens(text, maxTokens) {
    // Simple truncation by removing content from the middle
    const tokens = countTokens(text);
    
    if (tokens <= maxTokens) {
      return text;
    }
    
    // Split into lines
    const lines = text.split('\n');
    
    // If very few lines, just return the first part
    if (lines.length <= 5) {
      // Estimate tokens per character and truncate
      const tokensPerChar = tokens / text.length;
      const maxChars = Math.floor(maxTokens / tokensPerChar);
      return text.substring(0, maxChars) + '\n\n[Context truncated due to length]';
    }
    
    // Keep the beginning and end parts
    const keepRatio = maxTokens / tokens;
    const linesToKeepStart = Math.floor(lines.length * keepRatio * 0.6); // More from beginning
    const linesToKeepEnd = Math.floor(lines.length * keepRatio * 0.4); // Less from end
    
    const beginningLines = lines.slice(0, linesToKeepStart);
    const endLines = lines.slice(-linesToKeepEnd);
    
    return beginningLines.join('\n') + 
           '\n\n[... Context truncated due to length ...]\n\n' + 
           endLines.join('\n');
  }
  
  // Initialize
  initialize();
  
  // Return the enhanced agent interface
  return {
    processMessage,
    indexProject,
    learnFromFeedback,
    generatePrompt,
    processFileModification,
    codeRAG,
    graphRAG,
    memoryController
  };
}

module.exports = {
  createEnhancedAgent
};