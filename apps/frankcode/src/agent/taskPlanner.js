/**
 * Task Planning Module for FrankCode
 * 
 * This module is responsible for breaking down complex tasks into
 * manageable steps and creating an execution plan.
 */

const { logger } = require('../utils/logger');
const { countTokens } = require('./tokenizer');

/**
 * Create a task planner
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.llm LLM client for generating plans
 * @param {Object} options.agent Agent instance
 * @returns {Object} The task planner interface
 */
function createTaskPlanner({ llm, agent }) {
  /**
   * Break down a complex task into steps
   * 
   * @param {string} task The high-level task description
   * @param {Object} context Current project context
   * @returns {Promise<Array<Object>>} Planned steps
   */
  async function planTask(task, context) {
    try {
      logger.info(`Planning task: ${task}`);
      
      // Generate the planning prompt
      const prompt = generatePlanningPrompt(task, context);
      
      // Use the LLM to generate a plan
      const response = await llm.generateResponse(prompt, {
        temperature: 0.3, // Lower temperature for more consistent planning
        maxTokens: 1024   // Allow for detailed plans
      });
      
      // Parse the plan from the response
      const steps = parsePlanSteps(response.text);
      
      logger.info(`Task plan created with ${steps.length} steps`);
      return steps;
    } catch (error) {
      logger.error(`Failed to plan task: ${error.message}`, { error });
      return [{ 
        type: 'error', 
        description: `Failed to create plan: ${error.message}`,
        action: 'notify'
      }];
    }
  }
  
  /**
   * Generate a prompt for task planning
   * 
   * @param {string} task The task description
   * @param {Object} context Current project context
   * @returns {string} The planning prompt
   */
  function generatePlanningPrompt(task, context) {
    // Include only the most relevant files to keep context manageable
    const relevantFiles = getRelevantFilesForTask(task, context);
    
    return `You are FrankCode, an AI coding assistant that helps with complex programming tasks.
I need you to break down the following task into a clear step-by-step plan:

TASK: ${task}

Current project context:
${relevantFiles.map(file => `- ${file.filePath}: ${file.summary}`).join('\n')}

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
  }
  
  /**
   * Parse the steps from the LLM response
   * 
   * @param {string} response The LLM response text
   * @returns {Array<Object>} Parsed steps
   */
  function parsePlanSteps(response) {
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
   * Get relevant files for a task based on content and keywords
   * 
   * @param {string} task The task description
   * @param {Object} context Current project context
   * @returns {Array<Object>} Relevant files
   */
  function getRelevantFilesForTask(task, context) {
    // Simple relevance scoring based on keywords in the task
    const taskWords = task.toLowerCase().split(/\W+/).filter(w => w.length > 3);
    const relevanceScores = new Map();
    
    // Score each file based on matching keywords
    for (const file of context.fileContexts) {
      let score = 0;
      
      // File name relevance (weighted higher)
      const fileName = file.filePath.toLowerCase();
      taskWords.forEach(word => {
        if (fileName.includes(word)) {
          score += 5;
        }
      });
      
      // Content relevance
      const content = file.content.toLowerCase();
      taskWords.forEach(word => {
        const regex = new RegExp(word, 'g');
        const matches = content.match(regex);
        if (matches) {
          score += matches.length;
        }
      });
      
      relevanceScores.set(file, score);
    }
    
    // Sort files by relevance score and take top 5
    return Array.from(context.fileContexts)
      .sort((a, b) => relevanceScores.get(b) - relevanceScores.get(a))
      .slice(0, 5);
  }
  
  /**
   * Estimate the difficulty and time required for a task
   * 
   * @param {string} task The task description
   * @param {Array<Object>} steps The planned steps
   * @returns {Object} Difficulty assessment
   */
  function assessTaskDifficulty(task, steps) {
    // Count total files to be modified
    const uniqueFiles = new Set();
    steps.forEach(step => {
      step.files.forEach(file => uniqueFiles.add(file));
    });
    
    // Assess based on number of steps and files
    const filesToModify = uniqueFiles.size;
    const stepCount = steps.length;
    
    // Count complexity factors in the task description
    const complexityFactors = [
      /refactor/i, 
      /optimize/i, 
      /architecture/i, 
      /redesign/i, 
      /performance/i, 
      /security/i
    ];
    
    const complexityScore = complexityFactors.reduce((score, regex) => 
      regex.test(task) ? score + 1 : score, 0);
    
    let difficulty;
    if (stepCount <= 2 && filesToModify <= 1 && complexityScore === 0) {
      difficulty = 'easy';
    } else if (stepCount <= 5 && filesToModify <= 3 && complexityScore <= 1) {
      difficulty = 'medium';
    } else {
      difficulty = 'complex';
    }
    
    // Estimate time (very rough estimate)
    let estimatedMinutes;
    switch (difficulty) {
      case 'easy':
        estimatedMinutes = 1;
        break;
      case 'medium':
        estimatedMinutes = 3;
        break;
      case 'complex':
        estimatedMinutes = 5 + (stepCount - 5) * 2;
        break;
    }
    
    return {
      difficulty,
      estimatedMinutes,
      stepCount,
      filesToModify
    };
  }
  
  // Return the task planner interface
  return {
    planTask,
    assessTaskDifficulty
  };
}

module.exports = {
  createTaskPlanner
};