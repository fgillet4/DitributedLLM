/**
 * AutoGen Integration for FrankCode
 * 
 * This module provides integration with AutoGen concepts to enhance agent capabilities.
 */

const { logger } = require('../utils/logger');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);
const path = require('path');
const fs = require('fs').promises;

/**
 * Create an AutoGen-inspired agent enhancer
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.agent Agent instance
 * @param {Object} options.outputRenderer Output renderer
 * @param {Object} options.apiClient API client
 * @returns {Object} The agent enhancer interface
 */
function createAgentEnhancer({ agent, outputRenderer, apiClient }) {
  /**
   * Add planning capabilities to break tasks into steps
   * 
   * @param {string} task The task to plan
   * @returns {Promise<Array<Object>>} The planned steps
   */
  async function planTask(task) {
    try {
      outputRenderer.addSystemMessage(`🧠 Planning task: ${task}`);
      
      const prompt = `You are FrankCode, an AI coding assistant that helps with complex programming tasks.
I need you to break down the following task into a clear step-by-step plan:

TASK: ${task}

Your task is to break this down into smaller steps. For each step:
1. Specify the type of action required (read_file, search_files, update_file, create_file, generate_code, analyze)
2. Provide a clear description of what needs to be done
3. List any specific files that need to be examined or modified
4. Explain the reasoning for this step

FORMAT YOUR RESPONSE AS:
STEP 1:
Type: [action_type]
Description: [clear description]
Files: [file paths, if relevant]
Reasoning: [brief explanation]

STEP 2:
...

Ensure your plan is comprehensive yet concise. Identify 3-7 key steps to make the task manageable.`;
      
      const response = await apiClient.generateResponse(prompt, {
        temperature: 0.3,
        maxTokens: 1024
      });
      
      const steps = parseSteps(response.text);
      
      // Display the plan
      outputRenderer.addSystemMessage(`📋 Task Plan (${steps.length} steps):`);
      
      steps.forEach((step, index) => {
        outputRenderer.addSystemMessage(`${index + 1}. ${step.type.toUpperCase()}: ${step.description}`);
      });
      
      return steps;
    } catch (error) {
      logger.error(`Failed to plan task: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Error planning task: ${error.message}`);
      return [];
    }
  }
  
  /**
   * Parse the steps from the LLM response
   */
  function parseSteps(response) {
    const steps = [];
    const stepRegex = /STEP\s+(\d+):\s*\n([\s\S]*?)(?=STEP\s+\d+:|$)/g;
    
    let match;
    while ((match = stepRegex.exec(response)) !== null) {
      const stepContent = match[2].trim();
      
      // Parse step details
      const typeMatch = /Type:\s*(\w+)/.exec(stepContent);
      const descMatch = /Description:\s*(.+?)(?=\n\w+:|$)/s.exec(stepContent);
      const filesMatch = /Files:\s*(.+?)(?=\n\w+:|$)/s.exec(stepContent);
      const reasoningMatch = /Reasoning:\s*(.+?)(?=\n\w+:|$)/s.exec(stepContent);
      
      if (typeMatch && descMatch) {
        steps.push({
          type: typeMatch[1].toLowerCase(),
          description: descMatch[1].trim(),
          files: filesMatch ? 
            filesMatch[1].trim().split(/,\s*/).filter(f => f !== 'N/A' && f !== '') : 
            [],
          reasoning: reasoningMatch ? reasoningMatch[1].trim() : ''
        });
      }
    }
    
    // If no steps were parsed but we have content, create a fallback step
    if (steps.length === 0 && response.trim().length > 0) {
      steps.push({
        type: 'analyze',
        description: 'Analyze the task and provide guidance',
        files: [],
        reasoning: 'No structured plan could be created, falling back to general analysis'
      });
    }
    
    return steps;
  }
  
  /**
   * Execute a task with planning and step-by-step execution
   * 
   * @param {string} task The task to execute
   * @returns {Promise<Object>} The execution result
   */
  async function executeTask(task) {
    try {
      // Plan the task
      const steps = await planTask(task);
      
      if (steps.length === 0) {
        outputRenderer.addErrorMessage('Failed to create a plan for this task.');
        return { status: 'error', error: 'Failed to create plan' };
      }
      
      // Execute each step
      for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        
        // Display current step
        outputRenderer.addSystemMessage(`\n▶️ Step ${i + 1}/${steps.length}: ${step.description}`);
        
        // Generate a prompt for this step
        const prompt = `You are FrankCode, an AI coding assistant.
I need your help with this step in a larger task:

TASK: ${task}
STEP: ${step.description}
${step.files && step.files.length > 0 ? `FILES: ${step.files.join(', ')}` : ''}
${step.reasoning ? `CONTEXT: ${step.reasoning}` : ''}

Please provide a detailed and helpful response to complete this step.`;
        
        // Execute the step using the agent
        const response = await agent.processMessage(prompt);
        
        // Display assistant response (this should happen automatically in processMessage)
        // Check for file modifications if any
        if (response && response.fileModifications && response.fileModifications.length > 0) {
          // Handle file modifications - these will be handled separately
        }
        
        // Give user a chance to review/cancel
        if (i < steps.length - 1) {
          outputRenderer.addSystemMessage('\nPress Enter to continue to the next step, or type "cancel" to stop.');
        }
      }
      
      // Task completed successfully
      outputRenderer.addSystemMessage(`\n✅ Task completed successfully!`);
      
      return { 
        status: 'success',
        steps: steps.length
      };
    } catch (error) {
      logger.error(`Failed to execute task: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Error executing task: ${error.message}`);
      
      return { 
        status: 'error', 
        error: error.message
      };
    }
  }
  
  /**
   * Create a file in the specified location
   * 
   * @param {string} filePath File path
   * @param {string} content File content
   * @returns {Promise<boolean>} Success indicator
   */
  async function createFile(filePath, content) {
    try {
      // Get absolute path
      const fullPath = path.isAbsolute(filePath) ? filePath : path.join(process.cwd(), filePath);
      
      // Ensure directory exists
      const dirPath = path.dirname(fullPath);
      await fs.mkdir(dirPath, { recursive: true }).catch(() => {});
      
      // Write file
      await fs.writeFile(fullPath, content, 'utf8');
      
      outputRenderer.addSystemMessage(`✅ Created file: ${filePath}`);
      return true;
    } catch (error) {
      logger.error(`Failed to create file: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Error creating file: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Execute code in a safe environment
   * 
   * @param {string} code The code to execute
   * @param {string} language The language (python, javascript, etc.)
   * @returns {Promise<Object>} The execution result
   */
  async function executeCode(code, language = 'python') {
    try {
      // Create a temporary file
      const extension = language === 'python' ? '.py' : 
                       language === 'javascript' ? '.js' : 
                       language === 'bash' ? '.sh' : '.txt';
      
      const tempFile = `temp_${Date.now()}${extension}`;
      await fs.writeFile(tempFile, code, 'utf8');
      
      // Execute the code
      let command;
      switch(language) {
        case 'python':
          command = `python ${tempFile}`;
          break;
        case 'javascript':
          command = `node ${tempFile}`;
          break;
        case 'bash':
          command = `bash ${tempFile}`;
          break;
        default:
          throw new Error(`Unsupported language: ${language}`);
      }
      
      outputRenderer.addSystemMessage(`⚙️ Executing ${language} code...`);
      
      const { stdout, stderr } = await execPromise(command);
      
      // Clean up
      await fs.unlink(tempFile).catch(() => {});
      
      // Display result
      if (stdout) {
        outputRenderer.addSystemMessage('🔹 Output:');
        outputRenderer.addCodeBlock(stdout, 'bash');
      }
      
      if (stderr) {
        outputRenderer.addErrorMessage('Error output:');
        outputRenderer.addCodeBlock(stderr, 'bash');
      }
      
      return { 
        success: !stderr, 
        stdout, 
        stderr 
      };
    } catch (error) {
      logger.error(`Failed to execute code: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Error executing code: ${error.message}`);
      
      // Try to clean up
      try {
        await fs.unlink(tempFile).catch(() => {});
      } catch {}
      
      return { 
        success: false, 
        error: error.message 
      };
    }
  }
  
  // Return the agent enhancer interface
  return {
    planTask,
    executeTask,
    createFile,
    executeCode
  };
}

module.exports = {
  createAgentEnhancer
};