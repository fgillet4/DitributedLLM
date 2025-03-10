/**
 * Agent Runner for FrankCode
 * 
 * This module is responsible for executing agent actions,
 * coordinating between the LLM, tools, and user interface.
 */

const { logger } = require('../utils/logger');
const fileOperations = require('./tools/fileOperations');
const { confirmOperation } = require('../tui/confirmation');
const { createTaskPlanner } = require('./taskPlanner');

/**
 * Agent Action Types
 */
const ActionType = {
  READ_FILE: 'read_file',
  SEARCH_FILES: 'search_files',
  UPDATE_FILE: 'update_file',
  CREATE_FILE: 'create_file',
  DELETE_FILE: 'delete_file',
  GENERATE_CODE: 'generate_code',
  ANALYZE: 'analyze',
  ANSWER_QUESTION: 'answer_question'
};

/**
 * Create an agent runner
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.llm LLM client
 * @param {Object} options.screen Blessed screen object
 * @param {Object} options.outputRenderer Output renderer
 * @param {Object} options.agent Agent instance
 * @returns {Object} Agent runner interface
 */
function createAgentRunner({ llm, screen, outputRenderer, agent }) {
  // Store preferences
  let autoConfirm = false;
  
  // Create task planner
  const taskPlanner = createTaskPlanner({ llm, agent });
  
  // Track the current task execution
  let currentTask = null;
  let currentPlan = [];
  let currentStepIndex = 0;
  
  /**
   * Execute a complex task by breaking it down and sequentially processing steps
   * 
   * @param {string} taskDescription Description of the task
   * @returns {Promise<Object>} Task result
   */
  async function executeTask(taskDescription) {
    try {
      // Initialize task tracking
      currentTask = taskDescription;
      currentStepIndex = 0;
      
      // Add system message
      outputRenderer.addSystemMessage(`🧠 Analyzing task: ${taskDescription}`);
      
      // Get the current context from the agent
      const context = agent.getContextManager().getCurrentContext();
      
      // Create a plan for the task
      currentPlan = await taskPlanner.planTask(taskDescription, context);
      
      // If planning failed, return error
      if (currentPlan.length === 0 || (currentPlan.length === 1 && currentPlan[0].type === 'error')) {
        outputRenderer.addErrorMessage('Failed to create a plan for this task. Please try being more specific.');
        return { status: 'error', error: 'Failed to create plan' };
      }
      
      // Display the plan
      outputRenderer.addSystemMessage(`📋 Task Plan (${currentPlan.length} steps):`);
      
      currentPlan.forEach((step, index) => {
        outputRenderer.addSystemMessage(`${index + 1}. ${step.type.toUpperCase()}: ${step.description}`);
      });
      
      // Assess task difficulty
      const difficulty = taskPlanner.assessTaskDifficulty(taskDescription, currentPlan);
      outputRenderer.addSystemMessage(`⏱️ Estimated time: ~${difficulty.estimatedMinutes} minute${difficulty.estimatedMinutes > 1 ? 's' : ''} (${difficulty.difficulty} complexity)`);
      
      // Execute each step in the plan
      for (currentStepIndex = 0; currentStepIndex < currentPlan.length; currentStepIndex++) {
        const step = currentPlan[currentStepIndex];
        
        // Display current step
        outputRenderer.addSystemMessage(`\n▶️ Step ${currentStepIndex + 1}/${currentPlan.length}: ${step.description}`);
        
        // Execute the step
        const stepResult = await executeStep(step);
        
        // Handle step failure
        if (stepResult.status === 'error') {
          outputRenderer.addErrorMessage(`Failed to complete step ${currentStepIndex + 1}: ${stepResult.error}`);
          
          // Ask the LLM for recovery suggestions
          const recoveryResult = await suggestRecovery(step, stepResult.error);
          outputRenderer.addSystemMessage(`💡 Recovery suggestion: ${recoveryResult.suggestion}`);
          
          // We may want to continue with next steps even if one fails
          if (recoveryResult.fatal) {
            outputRenderer.addErrorMessage('Cannot continue with the task due to critical error.');
            return { status: 'error', error: stepResult.error, completedSteps: currentStepIndex };
          }
        } else if (stepResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Completed step ${currentStepIndex + 1}`);
        }
        
        // Force render after each step
        screen.render();
      }
      
      // Task completed successfully
      outputRenderer.addSystemMessage(`\n✅ Task completed successfully!`);
      
      return { 
        status: 'success',
        steps: currentPlan.length,
        completedSteps: currentStepIndex
      };
    } catch (error) {
      logger.error(`Failed to execute task: ${taskDescription}`, { error });
      outputRenderer.addErrorMessage(`Error executing task: ${error.message}`);
      
      return { 
        status: 'error', 
        error: error.message,
        completedSteps: currentStepIndex
      };
    } finally {
      // Reset the task tracking
      currentTask = null;
      currentPlan = [];
      currentStepIndex = 0;
    }
  }
  
  /**
   * Execute a single step from the plan
   * 
   * @param {Object} step Step to execute
   * @returns {Promise<Object>} Step result
   */
  async function executeStep(step) {
    try {
      logger.debug(`Executing step: ${step.type}`, { step });
      
      switch (step.type) {
        case ActionType.READ_FILE:
          return await executeReadFile(step);
        
        case ActionType.SEARCH_FILES:
          return await executeSearchFiles(step);
        
        case ActionType.UPDATE_FILE:
          return await executeUpdateFile(step);
        
        case ActionType.CREATE_FILE:
          return await executeCreateFile(step);
        
        case ActionType.DELETE_FILE:
          return await executeDeleteFile(step);
        
        case ActionType.GENERATE_CODE:
          return await executeGenerateCode(step);
        
        case ActionType.ANALYZE:
          return await executeAnalyze(step);
        
        case ActionType.ANSWER_QUESTION:
          return await executeAnswerQuestion(step);
        
        default:
          return {
            type: step.type,
            status: 'error',
            error: `Unknown step type: ${step.type}`
          };
      }
    } catch (error) {
      logger.error(`Failed to execute step: ${step.type}`, { error });
      return {
        type: step.type,
        status: 'error',
        error: error.message
      };
    }
  }
  
  /**
   * Execute read file action
   * 
   * @param {Object} step Read file step
   * @returns {Promise<Object>} Action result
   */
  async function executeReadFile(step) {
    // Check if we have specific files to read
    if (!step.files || step.files.length === 0) {
      outputRenderer.addErrorMessage('No files specified for reading');
      return { status: 'error', error: 'No files specified' };
    }
    
    const results = [];
    
    // Read each file
    for (const filePath of step.files) {
      // Display action in UI
      outputRenderer.addSystemMessage(`📄 Reading file: ${filePath}`);
      
      // Execute the action
      const result = await fileOperations.readFile(filePath);
      
      // Display result in UI
      if (result.status === 'success') {
        outputRenderer.addCodeBlock(result.rawContent, getFileLanguage(filePath));
        results.push(result);
      } else {
        outputRenderer.addErrorMessage(`Failed to read file: ${result.error}`);
        return { status: 'error', error: result.error };
      }
    }
    
    return { status: 'success', results };
  }
  
  /**
   * Execute search files action
   * 
   * @param {Object} step Search files step
   * @returns {Promise<Object>} Action result
   */
  async function executeSearchFiles(step) {
    let pattern = '*';
    let baseDir = '';
    
    // Extract pattern from description or use files
    if (step.files && step.files.length > 0) {
      pattern = step.files[0];
    } else {
      // Try to extract pattern from description
      const patternMatch = /pattern[:\s]+([^\s]+)/i.exec(step.description);
      if (patternMatch) {
        pattern = patternMatch[1];
      }
    }
    
    // Display action in UI
    outputRenderer.addSystemMessage(`🔍 Searching for files: ${pattern}${baseDir ? ` in ${baseDir}` : ''}`);
    
    // Execute the action
    const result = await fileOperations.searchFiles(pattern, baseDir);
    
    // Display result in UI
    if (result.status === 'success') {
      if (result.matches.length > 0) {
        outputRenderer.addSystemMessage(`Found ${result.matches.length} matching files:`);
        outputRenderer.addCodeBlock(result.matches.join('\n'), 'bash');
      } else {
        outputRenderer.addSystemMessage('No matching files found.');
      }
    } else {
      outputRenderer.addErrorMessage(`Failed to search files: ${result.error}`);
      return { status: 'error', error: result.error };
    }
    
    return { status: 'success', result };
  }
  
  /**
   * Execute update file action
   * 
   * @param {Object} step Update file step
   * @returns {Promise<Object>} Action result
   */
  async function executeUpdateFile(step) {
    // For update operations, we'll need to query the LLM
    // to generate the actual changes based on step description
    if (!step.files || step.files.length === 0) {
      outputRenderer.addErrorMessage('No files specified for update');
      return { status: 'error', error: 'No files specified' };
    }
    
    // Go through each file to update
    for (const filePath of step.files) {
      // First, read the file
      outputRenderer.addSystemMessage(`📄 Reading file for update: ${filePath}`);
      const readResult = await fileOperations.readFile(filePath);
      
      if (readResult.status !== 'success') {
        outputRenderer.addErrorMessage(`Failed to read file: ${readResult.error}`);
        return { status: 'error', error: readResult.error };
      }
      
      // Display current content
      outputRenderer.addCodeBlock(readResult.rawContent, getFileLanguage(filePath));
      
      // Generate changes using the LLM
      outputRenderer.addSystemMessage(`🧠 Generating updates for: ${filePath}`);
      
      const updatePrompt = `You are helping update code in the file: ${filePath}
Current content:
\`\`\`
${readResult.rawContent}
\`\`\`

Update needed: ${step.description}

Provide the complete updated file content, with appropriate changes to satisfy the requirement.
Only return the updated code without explanations or markdown.
`;
      
      const generatedUpdate = await llm.generateResponse(updatePrompt);
      
      // Extract the code (remove any markdown or explanation the LLM might add)
      const newContent = extractCodeContent(generatedUpdate.text);
      
      // Display action in UI
      outputRenderer.addSystemMessage(`✏️ Update file: ${filePath}`);
      
      // Execute the action
      const updateResult = await fileOperations.updateFile(filePath, newContent);
      
      // Handle errors
      if (updateResult.status === 'error') {
        outputRenderer.addErrorMessage(`Failed to update file: ${updateResult.error}`);
        return { status: 'error', error: updateResult.error };
      }
      
      // Show a message about the changes
      outputRenderer.addSystemMessage(`Changes: ${updateResult.additions} additions, ${updateResult.removals} removals`);
      
      // Auto-confirm if enabled
      if (autoConfirm) {
        const confirmResult = await fileOperations.confirmUpdate(updateResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Updated ${filePath} with ${updateResult.additions} additions and ${updateResult.removals} removals`);
        } else {
          outputRenderer.addErrorMessage(`Failed to update file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      }
      
      // Request confirmation from user
      const confirmation = await confirmOperation(screen, updateResult);
      
      // Handle confirmation result
      if (confirmation.choice === 'yes' || confirmation.choice === 'yes-to-all') {
        // Set auto-confirm if yes-to-all
        if (confirmation.choice === 'yes-to-all') {
          autoConfirm = true;
        }
        
        // Confirm the update
        const confirmResult = await fileOperations.confirmUpdate(updateResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Updated ${filePath} with ${updateResult.additions} additions and ${updateResult.removals} removals`);
        } else {
          outputRenderer.addErrorMessage(`Failed to update file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      } else if (confirmation.choice === 'custom') {
        // Apply custom content
        const customResult = await fileOperations.updateFile(filePath, confirmation.customContent);
        const confirmResult = await fileOperations.confirmUpdate(customResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Updated ${filePath} with custom changes`);
        } else {
          outputRenderer.addErrorMessage(`Failed to update file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      } else {
        // Cancelled or rejected
        outputRenderer.addSystemMessage(`❌ Update cancelled for ${filePath}`);
        return { status: 'cancelled', filePath };
      }
    }
    
    return { status: 'error', error: 'No files were processed' };
  }
  
  /**
   * Execute create file action
   * 
   * @param {Object} step Create file step
   * @returns {Promise<Object>} Action result
   */
  async function executeCreateFile(step) {
    // For create operations, we'll need to query the LLM
    // to generate the content based on step description
    if (!step.files || step.files.length === 0) {
      outputRenderer.addErrorMessage('No files specified for creation');
      return { status: 'error', error: 'No files specified' };
    }
    
    // Go through each file to create
    for (const filePath of step.files) {
      // Generate content using the LLM
      outputRenderer.addSystemMessage(`🧠 Generating content for new file: ${filePath}`);
      
      const createPrompt = `You are helping create a new file: ${filePath}

File purpose: ${step.description}

Please generate appropriate content for this file. Consider the file extension and purpose.
Only return the code without explanations or markdown.
`;
      
      const generatedContent = await llm.generateResponse(createPrompt);
      
      // Extract the code (remove any markdown or explanation the LLM might add)
      const content = extractCodeContent(generatedContent.text);
      
      // Display action in UI
      outputRenderer.addSystemMessage(`🔨 Creating file: ${filePath}`);
      outputRenderer.addCodeBlock(content, getFileLanguage(filePath));
      
      // Execute the action
      const createResult = await fileOperations.createFile(filePath, content);
      
      // Handle errors
      if (createResult.status === 'error') {
        outputRenderer.addErrorMessage(`Failed to create file: ${createResult.error}`);
        return { status: 'error', error: createResult.error };
      }
      
      // Auto-confirm if enabled
      if (autoConfirm) {
        const confirmResult = await fileOperations.confirmCreate(createResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Created ${filePath}`);
        } else {
          outputRenderer.addErrorMessage(`Failed to create file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      }
      
      // Request confirmation from user
      const confirmation = await confirmOperation(screen, createResult);
      
      // Handle confirmation result
      if (confirmation.choice === 'yes' || confirmation.choice === 'yes-to-all') {
        // Set auto-confirm if yes-to-all
        if (confirmation.choice === 'yes-to-all') {
          autoConfirm = true;
        }
        
        // Confirm the creation
        const confirmResult = await fileOperations.confirmCreate(createResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Created ${filePath}`);
        } else {
          outputRenderer.addErrorMessage(`Failed to create file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      } else if (confirmation.choice === 'custom') {
        // Apply custom content
        const customResult = await fileOperations.createFile(filePath, confirmation.customContent);
        const confirmResult = await fileOperations.confirmCreate(customResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Created ${filePath} with custom content`);
        } else {
          outputRenderer.addErrorMessage(`Failed to create file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      } else {
        // Cancelled or rejected
        outputRenderer.addSystemMessage(`❌ Creation cancelled for ${filePath}`);
        return { status: 'cancelled', filePath };
      }
    }
    
    return { status: 'error', error: 'No files were processed' };
  }
  
  /**
   * Execute delete file action
   * 
   * @param {Object} step Delete file step
   * @returns {Promise<Object>} Action result
   */
  async function executeDeleteFile(step) {
    if (!step.files || step.files.length === 0) {
      outputRenderer.addErrorMessage('No files specified for deletion');
      return { status: 'error', error: 'No files specified' };
    }
    
    // Go through each file to delete
    for (const filePath of step.files) {
      // Display action in UI
      outputRenderer.addSystemMessage(`🗑️ Delete file: ${filePath}`);
      
      // Execute the action
      const deleteResult = await fileOperations.deleteFile(filePath);
      
      // Handle errors
      if (deleteResult.status === 'error') {
        outputRenderer.addErrorMessage(`Failed to delete file: ${deleteResult.error}`);
        return { status: 'error', error: deleteResult.error };
      }
      
      // Request confirmation from user (never auto-confirm deletion)
      const confirmation = await confirmOperation(screen, deleteResult);
      
      // Handle confirmation result
      if (confirmation.choice === 'yes') {
        // Confirm the deletion
        const confirmResult = await fileOperations.confirmDelete(deleteResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Deleted ${filePath}`);
        } else {
          outputRenderer.addErrorMessage(`Failed to delete file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      } else {
        // Cancelled or rejected
        outputRenderer.addSystemMessage(`❌ Deletion cancelled for ${filePath}`);
        return { status: 'cancelled', filePath };
      }
    }
    
    return { status: 'error', error: 'No files were processed' };
  }
  
  /**
   * Execute generate code action
   * 
   * @param {Object} step Generate code step
   * @returns {Promise<Object>} Action result
   */
  async function executeGenerateCode(step) {
    try {
      // Display action in UI
      outputRenderer.addSystemMessage(`💻 Generating code: ${step.description}`);
      
      // Generate code using the LLM
      const prompt = `You are FrankCode, an AI programming assistant.
Generate high-quality code based on the following requirements:

${step.description}

${step.files && step.files.length > 0 ? `The code should be for file: ${step.files[0]}` : ''}

${step.reasoning ? `Additional context: ${step.reasoning}` : ''}

Provide only the code without any explanations or markdown formatting.`;
      
      const response = await llm.generateResponse(prompt);
      
      // Display the generated code
      const code = extractCodeContent(response.text);
      outputRenderer.addCodeBlock(code, step.files && step.files.length > 0 ? getFileLanguage(step.files[0]) : 'javascript');
      
      // If we have a file specified, offer to save it
      if (step.files && step.files.length > 0) {
        outputRenderer.addSystemMessage(`Would you like to save this code to ${step.files[0]}?`);
        outputRenderer.addSystemMessage(`Type /yes to save, or /no to just keep the code in the conversation.`);
        
        // The actual saving will be handled by the input handler
        return { 
          status: 'success', 
          code,
          pendingAction: {
            type: 'save_code',
            filePath: step.files[0],
            content: code
          }
        };
      }
      
      return { status: 'success', code };
    } catch (error) {
      logger.error(`Failed to generate code: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Failed to generate code: ${error.message}`);
      return { status: 'error', error: error.message };
    }
  }
  
  /**
   * Execute analyze action
   * 
   * @param {Object} step Analyze step
   * @returns {Promise<Object>} Action result
   */
  async function executeAnalyze(step) {
    try {
      // Display action in UI
      outputRenderer.addSystemMessage(`🔍 Analyzing: ${step.description}`);
      
      // Build context based on the files
      let fileContents = '';
      if (step.files && step.files.length > 0) {
        for (const filePath of step.files) {
          const readResult = await fileOperations.readFile(filePath);
          if (readResult.status === 'success') {
            fileContents += `File: ${filePath}\n\n${readResult.rawContent}\n\n`;
          }
        }
      }
      
      // Generate analysis using the LLM
      const prompt = `You are FrankCode, an AI programming assistant.
Analyze the following code and provide insights based on this request:

${step.description}

${fileContents ? `Code to analyze:\n${fileContents}` : ''}

${step.reasoning ? `Analysis focus: ${step.reasoning}` : ''}

Provide a detailed analysis addressing the request.`;
      
      const response = await llm.generateResponse(prompt, {
        temperature: 0.3 // Lower temperature for more analytical responses
      });
      
      // Display the analysis
      outputRenderer.addAssistantMessage(response.text);
      
      return { status: 'success', analysis: response.text };
    } catch (error) {
      logger.error(`Failed to analyze: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Failed to analyze: ${error.message}`);
      return { status: 'error', error: error.message };
    }
  }
  
  /**
   * Execute answer question action
   * 
   * @param {Object} step Answer question step
   * @returns {Promise<Object>} Action result
   */
  async function executeAnswerQuestion(step) {
    try {
      // Display action in UI
      outputRenderer.addSystemMessage(`❓ Answering: ${step.description}`);
      
      // Build context based on the files
      let fileContents = '';
      if (step.files && step.files.length > 0) {
        for (const filePath of step.files) {
          const readResult = await fileOperations.readFile(filePath);
          if (readResult.status === 'success') {
            fileContents += `File: ${filePath}\n\n${readResult.rawContent}\n\n`;
          }
        }
      }
      
      // Generate answer using the LLM
      const prompt = `You are FrankCode, an AI programming assistant.
Answer the following question:

${step.description}

${fileContents ? `Related code for context:\n${fileContents}` : ''}

${step.reasoning ? `Additional context: ${step.reasoning}` : ''}

Provide a clear and comprehensive answer.`;
      
      const response = await llm.generateResponse(prompt);
      
      // Display the answer
      outputRenderer.addAssistantMessage(response.text);
      
      return { status: 'success', answer: response.text };
    } catch (error) {
      logger.error(`Failed to answer question: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Failed to answer question: ${error.message}`);
      return { status: 'error', error: error.message };
    }
  }
  
  /**
   * Suggest recovery options for a failed step
   * 
   * @param {Object} step Failed step
   * @param {string} error Error message
   * @returns {Promise<Object>} Recovery suggestions
   */
  async function suggestRecovery(step, error) {
    try {
      const prompt = `You are FrankCode, an AI programming assistant.
A step in my task execution has failed. Please suggest how to recover:

Failed step: ${step.type} - ${step.description}
${step.files && step.files.length > 0 ? `Files involved: ${step.files.join(', ')}` : ''}
Error message: ${error}

Provide:
1. A brief explanation of what might have gone wrong
2. A specific suggestion on how to recover or work around this issue
3. Indicate if this is a fatal error that should halt the entire task (true/false)`;
      
      const response = await llm.generateResponse(prompt, {
        temperature: 0.3 // Lower temperature for more precise recovery suggestions
      });
      
      // Try to extract whether it's fatal
      const fatalMatch = /fatal:?\s*(true|false)/i.exec(response.text);
      const isFatal = fatalMatch ? fatalMatch[1].toLowerCase() === 'true' : false;
      
      return {
        suggestion: response.text,
        fatal: isFatal
      };
    } catch (error) {
      logger.error(`Failed to suggest recovery: ${error.message}`, { error });
      return {
        suggestion: "Failed to generate recovery suggestions. You may need to modify your task or approach.",
        fatal: true
      };
    }
  }
  
  /**
   * Extract code content from LLM response, removing markdown and explanations
   * 
   * @param {string} text LLM response text
   * @returns {string} Extracted code content
   */
  function extractCodeContent(text) {
    // Check for code block
    const codeBlockMatch = /```(?:\w*\n)?([\s\S]*?)```/g.exec(text);
    if (codeBlockMatch) {
      return codeBlockMatch[1].trim();
    }
    
    // No code block, strip any explanations (look for patterns like explanations followed by actual code)
    const lines = text.split('\n');
    
    // Look for patterns indicating explanations
    const codeStartIndex = lines.findIndex((line, index) => {
      // Common patterns that indicate the start of code after explanations
      const codeStartPatterns = [
        /^(class|function|import|const|let|var|#|\/\/|\*|\/\*|<!)/,
        /^[a-zA-Z0-9_]+\s*\(\s*\)\s*{/,
        /^export /,
        /^public /,
        /^private /,
        /^module\./
      ];
      
      return index > 0 && codeStartPatterns.some(pattern => pattern.test(line));
    });
    
    if (codeStartIndex > 0) {
      return lines.slice(codeStartIndex).join('\n').trim();
    }
    
    // If no obvious code start found, return the original text
    return text.trim();
  }
  
  /**
   * Get the language for syntax highlighting based on file extension
   * 
   * @param {string} filePath File path
   * @returns {string} Language for syntax highlighting
   */
  function getFileLanguage(filePath) {
    const extension = filePath.split('.').pop().toLowerCase();
    
    const languageMap = {
      'js': 'javascript',
      'jsx': 'javascript',
      'ts': 'typescript',
      'tsx': 'typescript',
      'py': 'python',
      'rb': 'ruby',
      'go': 'go',
      'java': 'java',
      'c': 'c',
      'cpp': 'cpp',
      'h': 'c',
      'hpp': 'cpp',
      'cs': 'csharp',
      'php': 'php',
      'html': 'html',
      'css': 'css',
      'scss': 'scss',
      'sass': 'scss',
      'json': 'json',
      'md': 'markdown',
      'xml': 'xml',
      'sh': 'bash',
      'bash': 'bash',
      'yml': 'yaml',
      'yaml': 'yaml',
      'sql': 'sql',
      'rs': 'rust',
      'swift': 'swift'
    };
    
    return languageMap[extension] || 'text';
  }
  
  /**
   * Execute a direct agent action
   * 
   * @param {Object} action Action to execute
   * @returns {Promise<Object>} Action result
   */
  async function executeAction(action) {
    try {
      // Create a step from the action
      const step = {
        type: action.type,
        description: action.description || '',
        files: action.files || [],
        reasoning: action.reasoning || ''
      };
      
      // Execute the step
      return await executeStep(step);
    } catch (error) {
      logger.error(`Failed to execute action: ${action.type}`, { error });
      return {
        type: action.type,
        status: 'error',
        error: error.message
      };
    }
  }
  
  // Return the agent runner interface
  return {
    executeTask,
    executeAction,
    ActionType
  };
}

module.exports = {
  createAgentRunner
};