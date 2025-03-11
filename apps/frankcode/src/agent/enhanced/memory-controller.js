/**
 * Memory Controller for FrankCode
 * 
 * This module provides a way for FrankCode to learn from past experiences
 * and remember code patterns, solving approaches, and user preferences.
 */

const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const { createEmbedding, cosineSimilarity } = require('./embeddings');
const { logger } = require('../utils/logger');

/**
 * Create a Memory Controller
 * 
 * @param {Object} options Configuration options
 * @param {string} options.memoryPath Path to store memories
 * @param {number} options.maxMemories Maximum number of memories to keep
 * @param {number} options.relevanceThreshold Threshold for considering memories relevant
 * @returns {Object} The memory controller interface
 */
function createMemoryController({ memoryPath, maxMemories = 1000, relevanceThreshold = 0.7 }) {
  // Memory types
  const MemoryType = {
    CODE_PATTERN: 'code_pattern',     // Remembered code structures and patterns
    USER_PREFERENCE: 'user_preference', // User's preferred way of doing things
    PROBLEM_SOLUTION: 'problem_solution', // How a problem was solved
    FILE_STRUCTURE: 'file_structure',   // Structure of a particular file
    TASK_INSIGHT: 'task_insight',      // Insight for a particular task
    ERROR_RESOLUTION: 'error_resolution' // How an error was fixed
  };
  
  // In-memory store
  let memories = [];
  
  /**
   * Initialize the memory controller
   * 
   * @returns {Promise<void>}
   */
  async function initialize() {
    try {
      // Ensure directory exists
      await fs.mkdir(path.dirname(memoryPath), { recursive: true });
      
      // Check if memory file exists
      try {
        await fs.access(memoryPath);
        
        // Load memories
        const data = await fs.readFile(memoryPath, 'utf8');
        memories = JSON.parse(data);
        logger.info(`Loaded ${memories.length} memories`);
      } catch (error) {
        // File doesn't exist, start with empty memories
        memories = [];
        logger.info('Starting with empty memories');
      }
    } catch (error) {
      logger.error(`Failed to initialize memory controller: ${error.message}`, { error });
      memories = [];
    }
  }
  
  /**
   * Save the current memories to disk
   * 
   * @returns {Promise<boolean>} Success indicator
   */
  async function saveMemories() {
    try {
      // Ensure directory exists
      await fs.mkdir(path.dirname(memoryPath), { recursive: true });
      
      // Write to file
      await fs.writeFile(memoryPath, JSON.stringify(memories, null, 2), 'utf8');
      logger.debug(`Saved ${memories.length} memories to ${memoryPath}`);
      return true;
    } catch (error) {
      logger.error(`Failed to save memories: ${error.message}`, { error });
      return false;
    }
  }
  
  /**
   * Add a new memory
   * 
   * @param {Object} memory Memory object
   * @param {string} memory.type Memory type
   * @param {string} memory.task Description of the task or context
   * @param {string} memory.insight The insight or information to remember
   * @param {Object} memory.metadata Additional metadata
   * @returns {Promise<string>} Memory ID
   */
  async function addMemory(memory) {
    try {
      // Generate an ID for the memory
      const id = crypto.randomUUID();
      
      // Generate embeddings
      const taskEmbedding = await createEmbedding(memory.task);
      const insightEmbedding = await createEmbedding(memory.insight);
      
      // Create full memory object
      const newMemory = {
        id,
        ...memory,
        created: Date.now(),
        lastAccessed: Date.now(),
        accessCount: 0,
        taskEmbedding,
        insightEmbedding
      };
      
      // Add to memories
      memories.push(newMemory);
      
      // Limit size if needed
      if (memories.length > maxMemories) {
        // Remove oldest memory
        memories.sort((a, b) => a.lastAccessed - b.lastAccessed);
        memories.shift();
      }
      
      // Save memories
      await saveMemories();
      
      logger.info(`Added new ${memory.type} memory: ${id}`);
      return id;
    } catch (error) {
      logger.error(`Failed to add memory: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * Retrieve memories relevant to a task
   * 
   * @param {string} task Task description
   * @param {Object} options Query options
   * @param {Array<string>} options.types Memory types to include
   * @param {number} options.limit Maximum number of results
   * @param {number} options.threshold Relevance threshold
   * @returns {Promise<Array<Object>>} Relevant memories
   */
  async function getRelevantMemories(task, options = {}) {
    try {
      const {
        types = null,
        limit = 5,
        threshold = relevanceThreshold
      } = options;
      
      // Generate embedding for the task
      const taskEmbedding = await createEmbedding(task);
      
      // Find relevant memories
      const relevantMemories = [];
      
      for (const memory of memories) {
        // Filter by type if specified
        if (types && !types.includes(memory.type)) {
          continue;
        }
        
        // Calculate relevance score
        const relevance = cosineSimilarity(taskEmbedding, memory.taskEmbedding);
        
        // Add if relevant
        if (relevance >= threshold) {
          relevantMemories.push({
            ...memory,
            relevance,
            taskEmbedding: undefined, // Don't include embedding in results
            insightEmbedding: undefined
          });
        }
      }
      
      // Sort by relevance
      relevantMemories.sort((a, b) => b.relevance - a.relevance);
      
      // Update access info for retrieved memories
      for (const memory of relevantMemories.slice(0, limit)) {
        const original = memories.find(m => m.id === memory.id);
        if (original) {
          original.lastAccessed = Date.now();
          original.accessCount++;
        }
      }
      
      // Save memory access info
      await saveMemories();
      
      logger.debug(`Retrieved ${relevantMemories.length} relevant memories for task`);
      return relevantMemories.slice(0, limit);
    } catch (error) {
      logger.error(`Failed to retrieve relevant memories: ${error.message}`, { error });
      return [];
    }
  }
  
  /**
   * Remove a memory by ID
   * 
   * @param {string} id Memory ID
   * @returns {Promise<boolean>} Success indicator
   */
  async function removeMemory(id) {
    try {
      // Find memory index
      const index = memories.findIndex(m => m.id === id);
      
      if (index === -1) {
        logger.warn(`Memory not found: ${id}`);
        return false;
      }
      
      // Remove memory
      memories.splice(index, 1);
      
      // Save memories
      await saveMemories();
      
      logger.info(`Removed memory: ${id}`);
      return true;
    } catch (error) {
      logger.error(`Failed to remove memory: ${error.message}`, { error });
      return false;
    }
  }
  
  /**
   * Clear all memories
   * 
   * @returns {Promise<boolean>} Success indicator
   */
  async function clearMemories() {
    try {
      // Clear memories
      memories = [];
      
      // Save empty memories
      await saveMemories();
      
      logger.info('Cleared all memories');
      return true;
    } catch (error) {
      logger.error(`Failed to clear memories: ${error.message}`, { error });
      return false;
    }
  }
  
  /**
   * Format memories for use in a prompt
   * 
   * @param {Array<Object>} memories Memories to format
   * @returns {string} Formatted memories
   */
  function formatMemoriesForPrompt(memories) {
    if (!memories || memories.length === 0) {
      return '';
    }
    
    let formatted = '--- RELEVANT MEMORIES ---\n\n';
    
    for (const memory of memories) {
      formatted += `Type: ${memory.type}\n`;
      formatted += `Relevance: ${memory.relevance ? memory.relevance.toFixed(2) : 'N/A'}\n`;
      formatted += `Context: ${memory.task}\n`;
      formatted += `Insight: ${memory.insight}\n`;
      
      if (memory.metadata) {
        formatted += 'Metadata:\n';
        for (const [key, value] of Object.entries(memory.metadata)) {
          formatted += `- ${key}: ${value}\n`;
        }
      }
      
      formatted += '\n';
    }
    
    return formatted;
  }
  
  /**
   * Add a memory about a code pattern
   * 
   * @param {string} pattern Description of the pattern
   * @param {string} example Example of the pattern
   * @param {Object} metadata Additional metadata
   * @returns {Promise<string>} Memory ID
   */
  async function rememberCodePattern(pattern, example, metadata = {}) {
    return await addMemory({
      type: MemoryType.CODE_PATTERN,
      task: pattern,
      insight: example,
      metadata
    });
  }
  
  /**
   * Add a memory about a user preference
   * 
   * @param {string} context The context of the preference
   * @param {string} preference The user's preference
   * @param {Object} metadata Additional metadata
   * @returns {Promise<string>} Memory ID
   */
  async function rememberUserPreference(context, preference, metadata = {}) {
    return await addMemory({
      type: MemoryType.USER_PREFERENCE,
      task: context,
      insight: preference,
      metadata
    });
  }
  
  /**
   * Add a memory about a problem solution
   * 
   * @param {string} problem Description of the problem
   * @param {string} solution The solution
   * @param {Object} metadata Additional metadata
   * @returns {Promise<string>} Memory ID
   */
  async function rememberProblemSolution(problem, solution, metadata = {}) {
    return await addMemory({
      type: MemoryType.PROBLEM_SOLUTION,
      task: problem,
      insight: solution,
      metadata
    });
  }
  
  /**
   * Get memories about code patterns
   * 
   * @param {string} context The context to find patterns for
   * @param {Object} options Query options
   * @returns {Promise<Array<Object>>} Relevant code patterns
   */
  async function getCodePatterns(context, options = {}) {
    return await getRelevantMemories(context, {
      ...options,
      types: [MemoryType.CODE_PATTERN]
    });
  }
  
  /**
   * Get memories about user preferences
   * 
   * @param {string} context The context to find preferences for
   * @param {Object} options Query options
   * @returns {Promise<Array<Object>>} Relevant user preferences
   */
  async function getUserPreferences(context, options = {}) {
    return await getRelevantMemories(context, {
      ...options,
      types: [MemoryType.USER_PREFERENCE]
    });
  }
  
  /**
   * Get memories about problem solutions
   * 
   * @param {string} problem The problem to find solutions for
   * @param {Object} options Query options
   * @returns {Promise<Array<Object>>} Relevant problem solutions
   */
  async function getProblemSolutions(problem, options = {}) {
    return await getRelevantMemories(problem, {
      ...options,
      types: [MemoryType.PROBLEM_SOLUTION]
    });
  }
  
  // Initialize the memory controller
  initialize();
  
  // Return the memory controller interface
  return {
    addMemory,
    getRelevantMemories,
    removeMemory,
    clearMemories,
    formatMemoriesForPrompt,
    rememberCodePattern,
    rememberUserPreference,
    rememberProblemSolution,
    getCodePatterns,
    getUserPreferences,
    getProblemSolutions,
    MemoryType
  };
}

module.exports = {
  createMemoryController
};