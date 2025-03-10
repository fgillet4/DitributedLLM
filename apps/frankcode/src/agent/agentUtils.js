/**
 * Agent Utilities
 * 
 * Helper functions for integrating agent capabilities with the FrankCode TUI
 */

const { logger } = require('../utils/logger');
const { createAgentRunner } = require('./runner');

/**
 * Create an agent command processor
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.agent Agent instance
 * @param {Object} options.llm LLM client
 * @param {Object} options.screen Blessed screen object
 * @param {Object} options.outputRenderer Output renderer
 * @returns {Object} Command processor interface
 */
function createAgentCommandProcessor({ agent, llm, screen, outputRenderer }) {
  // Create agent runner
  const runner = createAgentRunner({ 
    llm, 
    screen, 
    outputRenderer,
    agent
  });

  // Command patterns
  const taskCommandPattern = /^(implement|create|build|add|refactor|fix|optimize)\s+(.+)$/i;
  const analyzeCommandPattern = /^(analyze|examine|check|review)\s+(.+)$/i;
  const explainCommandPattern = /^(explain|how|why)\s+(.+)$/i;
  
  /**
   * Process a user command and detect if it's an agent task
   * 
   * @param {string} command User command
   * @returns {Promise<boolean>} Whether the command was handled
   */
  async function processCommand(command) {
    try {
      // Normalize command
      const normalizedCommand = command.trim();
      
      // Check if this is a task command
      const taskMatch = taskCommandPattern.exec(normalizedCommand);
      if (taskMatch) {
        const verb = taskMatch[1].toLowerCase();
        const task = taskMatch[2];
        
        // Execute the task
        await runner.executeTask(`${verb} ${task}`);
        return true;
      }
      
      // Check if this is an analyze command
      const analyzeMatch = analyzeCommandPattern.exec(normalizedCommand);
      if (analyzeMatch) {
        const verb = analyzeMatch[1].toLowerCase();
        const target = analyzeMatch[2];
        
        // Execute analysis action
        await runner.executeAction({
          type: runner.ActionType.ANALYZE,
          description: `${verb} ${target}`,
          reasoning: 'User requested analysis'
        });
        return true;
      }
      
      // Check if this is an explain command
      const explainMatch = explainCommandPattern.exec(normalizedCommand);
      if (explainMatch) {
        const verb = explainMatch[1].toLowerCase();
        const question = explainMatch[2];
        
        // Execute question answering action
        await runner.executeAction({
          type: runner.ActionType.ANSWER_QUESTION,
          description: `${verb} ${question}`,
          reasoning: 'User asked a question'
        });
        return true;
      }
      
      // Check for explicit file operations
      if (normalizedCommand.startsWith('read ')) {
        const filePath = normalizedCommand.substring(5).trim();
        await runner.executeAction({
          type: runner.ActionType.READ_FILE,
          description: `Read the file ${filePath}`,
          files: [filePath]
        });
        return true;
      }
      
      if (normalizedCommand.startsWith('find ')) {
        const pattern = normalizedCommand.substring(5).trim();
        await runner.executeAction({
          type: runner.ActionType.SEARCH_FILES,
          description: `Find files matching ${pattern}`,
          files: [pattern]
        });
        return true;
      }
      
      // Not an agent command
      return false;
    } catch (error) {
      logger.error('Error processing agent command', { error });
      outputRenderer.addErrorMessage(`Error: ${error.message}`);
      return true; // Mark as handled to prevent further processing
    }
  }
  
  /**
   * Check if a command looks like it might be a task for the agent
   * 
   * @param {string} command User command
   * @returns {boolean} Whether it looks like an agent task
   */
  function isPotentialAgentTask(command) {
    const normalizedCommand = command.trim().toLowerCase();
    
    // Check against command patterns
    if (taskCommandPattern.test(normalizedCommand)) return true;
    if (analyzeCommandPattern.test(normalizedCommand)) return true;
    if (explainCommandPattern.test(normalizedCommand)) return true;
    
    // Check for explicit file operations
    if (normalizedCommand.startsWith('read ')) return true;
    if (normalizedCommand.startsWith('find ')) return true;
    
    // Look for programming-related terms that might indicate a task
    const programmingTerms = [
      'function', 'component', 'class', 'method',
      'module', 'interface', 'feature', 'bug',
      'performance', 'memory', 'leak', 'api',
      'endpoint', 'database', 'schema', 'model',
      'route', 'controller', 'view', 'template',
      'test', 'unit test', 'integration test',
      'authentication', 'authorization', 'security',
      'deploy', 'pipeline', 'build', 'docker',
      'caching', 'logging', 'error handling'
    ];
    
    return programmingTerms.some(term => normalizedCommand.includes(term));
  }
  
  /**
   * Suggest using agent capabilities
   * 
   * @param {string} command User command
   * @returns {string|null} Suggestion or null if none
   */
  function suggestAgentCapability(command) {
    const normalizedCommand = command.trim().toLowerCase();
    
    // Detect questions
    if (normalizedCommand.includes('how do i') || 
        normalizedCommand.includes('how to') ||
        normalizedCommand.includes('what is') ||
        normalizedCommand.includes('why does')) {
      return "It looks like you're asking a question. Try starting with 'explain' or 'how' for detailed explanations.";
    }
    
    // Detect potential tasks
    if (normalizedCommand.length > 15 && isPotentialAgentTask(normalizedCommand)) {
      return "This seems like a task I could help with. Try using task verbs like 'implement', 'create', 'fix', or 'refactor'.";
    }
    
    return null;
  }
  
  /**
   * Get a list of example agent commands
   * 
   * @returns {Array<string>} Example commands
   */
  function getExampleCommands() {
    return [
      "implement a function to sort users by creation date",
      "create a new React component for displaying user profiles",
      "fix the memory leak in the connection manager",
      "refactor the authentication module to use JWT",
      "analyze the performance bottlenecks in the data processing pipeline",
      "explain how the routing system works in this codebase",
      "read src/utils/formatters.js",
      "find all files related to authentication"
    ];
  }
  
  /**
   * Execute a specific agent task
   * 
   * @param {string} task Task description
   * @returns {Promise<Object>} Task result
   */
  async function executeTask(task) {
    return await runner.executeTask(task);
  }
  
  // Return the command processor interface
  return {
    processCommand,
    isPotentialAgentTask,
    suggestAgentCapability,
    getExampleCommands,
    executeTask,
    runner
  };
}

module.exports = {
  createAgentCommandProcessor
};