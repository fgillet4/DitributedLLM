/**
 * Input Handler for TUI
 * 
 * Manages user input and command processing
 */

const { logger } = require('../utils');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);
const path = require('path');

/**
 * Create an input handler
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.widget The input widget
 * @param {Object} options.outputRenderer Output renderer
 * @param {Object} options.agent Agent instance
 * @param {Object} options.fileTree File tree component
 * @param {Object} options.screen Blessed screen
 * @returns {Object} The input handler interface
 */
function createInputHandler({ widget, outputRenderer, agent, fileTree, screen }) {
  // Command history
  const history = [];
  let historyIndex = -1;
  
  // File modification tracking
  const pendingModifications = [];
  let currentModificationIndex = -1;
  
  // Initialize
  function init() {
    // Handle input submission
    widget.key('enter', async () => {
      const input = widget.getValue().trim();
      
      if (input === '') {
        return;
      }
      
      // Add to history
      history.push(input);
      historyIndex = history.length;
      
      // Clear input
      widget.setValue('');
      widget.screen.render();
      
      // Process the input
      await processInput(input);
    });
    
    // Handle history navigation
    widget.key('up', () => {
      if (historyIndex > 0) {
        historyIndex--;
        widget.setValue(history[historyIndex]);
        widget.screen.render();
      }
    });
    
    widget.key('down', () => {
      if (historyIndex < history.length - 1) {
        historyIndex++;
        widget.setValue(history[historyIndex]);
      } else {
        historyIndex = history.length;
        widget.setValue('');
      }
      widget.screen.render();
    });
    
    // Handle tab completion
    widget.key('tab', () => {
      // Implement tab completion later
    });
  }
  
  /**
   * Process user input
   * 
   * @param {string} input User input
   */
  async function processInput(input) {
    try {
      // Check for command prefixes
      if (input.startsWith('/')) {
        await processCommand(input.slice(1));
        return;
      }
      
      // Display user input
      outputRenderer.addUserMessage(input);
      
      // Send to agent for processing
      const response = await agent.processMessage(input);
      
      // Display assistant response
      outputRenderer.addAssistantMessage(response.text);
      
      // Check for file modifications
      if (response.fileModifications && response.fileModifications.length > 0) {
        await handleFileModifications(response.fileModifications);
      }
    } catch (error) {
      logger.error('Failed to process input', { error });
      outputRenderer.addErrorMessage(`Error processing input: ${error.message}`);
    }
  }
  
  /**
   * Process a command
   * 
   * @param {string} command Command string
   */
  async function processCommand(command) {
    try {
      // Split command and arguments
      const parts = command.split(/\s+/);
      const cmd = parts[0].toLowerCase();
      const args = parts.slice(1);
      
      // Process different commands
      switch (cmd) {
        case 'help':
          showHelp();
          break;
        
        case 'clear':
          outputRenderer.clear();
          break;
        
        case 'exit':
        case 'quit':
          process.exit(0);
          break;
        
        case 'refresh':
          await fileTree.refresh();
          outputRenderer.addSystemMessage('File tree refreshed');
          break;
        
        case 'exec':
        case 'shell':
          await executeShellCommand(args.join(' '));
          break;
        
        case 'load':
          await loadFile(args[0]);
          break;
        
        case 'save':
          await saveConversation(args[0]);
          break;
        
        case 'reset':
          agent.reset();
          outputRenderer.addSystemMessage('Agent context and conversation reset');
          break;
        
        case 'yes':
        case 'y':
          await approveModification(true);
          break;
        
        case 'no':
        case 'n':
          await approveModification(false);
          break;
        
        case 'yesall':
        case 'ya':
          await approveAllModifications();
          break;
        
        case 'noall':
        case 'na':
          await rejectAllModifications();
          break;
        
        default:
          outputRenderer.addSystemMessage(`Unknown command: ${cmd}. Type /help for available commands.`);
      }
    } catch (error) {
      logger.error('Failed to process command', { error });
      outputRenderer.addErrorMessage(`Error processing command: ${error.message}`);
    }
  }
  
  /**
   * Show help text
   */
  function showHelp() {
    const helpText = `
Available commands:
  /help             - Show this help text
  /clear            - Clear the conversation
  /exit, /quit      - Exit the application
  /refresh          - Refresh the file tree
  /exec <command>   - Execute a shell command
  /shell <command>  - Same as /exec
  /load <file>      - Load a file into context
  /save <file>      - Save the conversation to a file
  /reset            - Reset agent context and conversation
  /yes, /y          - Approve the current file modification
  /no, /n           - Reject the current file modification
  /yesall, /ya      - Approve all pending file modifications
  /noall, /na       - Reject all pending file modifications
    `;
    
    outputRenderer.addSystemMessage(helpText);
  }
  
  /**
   * Handle file modifications
   * 
   * @param {Array<Object>} modifications File modifications
   */
  async function handleFileModifications(modifications) {
    if (modifications.length === 0) {
      return;
    }
    
    // Store modifications
    pendingModifications.length = 0;
    pendingModifications.push(...modifications);
    currentModificationIndex = 0;
    
    // Display the first modification
    displayCurrentModification();
  }
  
  /**
   * Display the current modification
   */
  function displayCurrentModification() {
    if (pendingModifications.length === 0 || currentModificationIndex >= pendingModifications.length) {
      return;
    }
    
    const mod = pendingModifications[currentModificationIndex];
    
    // Show the file modification
    outputRenderer.addSystemMessage(`Proposed file modification (${currentModificationIndex + 1}/${pendingModifications.length}):`);
    outputRenderer.addCodeBlock(`File: ${mod.filePath}\n\n${mod.content}`, 'javascript');
    outputRenderer.addSystemMessage('Type /yes (or /y) to approve, /no (or /n) to reject, /yesall (or /ya) to approve all, /noall (or /na) to reject all');
  }
  
  /**
   * Approve or reject a modification
   * 
   * @param {boolean} approve Whether to approve the modification
   */
  async function approveModification(approve) {
    if (pendingModifications.length === 0 || currentModificationIndex >= pendingModifications.length) {
      outputRenderer.addSystemMessage('No pending file modifications.');
      return;
    }
    
    const mod = pendingModifications[currentModificationIndex];
    
    if (approve) {
      try {
        // Apply the modification
        const fullPath = path.resolve(path.join(screen.cwd, mod.filePath));
        await agent.modifyFile(fullPath, mod.content);
        
        outputRenderer.addSystemMessage(`✅ Applied changes to ${mod.filePath}`);
      } catch (error) {
        logger.error(`Failed to apply modification to ${mod.filePath}`, { error });
        outputRenderer.addErrorMessage(`Error applying modification: ${error.message}`);
      }
    } else {
      outputRenderer.addSystemMessage(`❌ Rejected changes to ${mod.filePath}`);
    }
    
    // Move to next modification
    currentModificationIndex++;
    
    // Display next modification if any
    if (currentModificationIndex < pendingModifications.length) {
      displayCurrentModification();
    } else {
      outputRenderer.addSystemMessage('All modifications processed.');
    }
  }
  
  /**
   * Approve all pending modifications
   */
  async function approveAllModifications() {
    if (pendingModifications.length === 0) {
      outputRenderer.addSystemMessage('No pending file modifications.');
      return;
    }
    
    const total = pendingModifications.length;
    let applied = 0;
    let failed = 0;
    
    for (const mod of pendingModifications) {
      try {
        // Apply the modification
        const fullPath = path.resolve(path.join(screen.cwd, mod.filePath));
        await agent.modifyFile(fullPath, mod.content);
        applied++;
      } catch (error) {
        logger.error(`Failed to apply modification to ${mod.filePath}`, { error });
        failed++;
      }
    }
    
    outputRenderer.addSystemMessage(`Applied ${applied}/${total} modifications (${failed} failed).`);
    
    // Clear pending modifications
    pendingModifications.length = 0;
    currentModificationIndex = 0;
  }
  
  /**
   * Reject all pending modifications
   */
  function rejectAllModifications() {
    if (pendingModifications.length === 0) {
      outputRenderer.addSystemMessage('No pending file modifications.');
      return;
    }
    
    const total = pendingModifications.length;
    outputRenderer.addSystemMessage(`Rejected all ${total} pending modifications.`);
    
    // Clear pending modifications
    pendingModifications.length = 0;
    currentModificationIndex = 0;
  }
  
  /**
   * Execute a shell command
   * 
   * @param {string} command Command to execute
   */
  async function executeShellCommand(command) {
    if (!command) {
      outputRenderer.addErrorMessage('No command specified');
      return;
    }
    
    try {
      outputRenderer.addSystemMessage(`Executing: ${command}`);
      
      const { stdout, stderr } = await execPromise(command, {
        cwd: screen.cwd,
        shell: true
      });
      
      if (stdout) {
        outputRenderer.addCodeBlock(stdout, 'shell');
      }
      
      if (stderr) {
        outputRenderer.addErrorMessage(stderr);
      }
    } catch (error) {
      outputRenderer.addErrorMessage(`Command failed: ${error.message}`);
      if (error.stdout) {
        outputRenderer.addCodeBlock(error.stdout, 'shell');
      }
      if (error.stderr) {
        outputRenderer.addErrorMessage(error.stderr);
      }
    }
  }
  
  /**
   * Load a file into context
   * 
   * @param {string} filePath Path to the file
   */
  async function loadFile(filePath) {
    if (!filePath) {
      outputRenderer.addErrorMessage('No file specified');
      return;
    }
    
    try {
      // Resolve path
      const fullPath = path.resolve(path.join(screen.cwd, filePath));
      
      // Load into agent context
      await agent.loadFileContext(fullPath);
      
      outputRenderer.addSystemMessage(`Loaded file into context: ${filePath}`);
    } catch (error) {
      logger.error(`Failed to load file: ${filePath}`, { error });
      outputRenderer.addErrorMessage(`Error loading file: ${error.message}`);
    }
  }
  
  /**
   * Save conversation to a file
   * 
   * @param {string} filePath Path to the file
   */
  async function saveConversation(filePath) {
    if (!filePath) {
      outputRenderer.addErrorMessage('No file specified');
      return;
    }
    
    try {
      // Resolve path
      const fullPath = path.resolve(path.join(screen.cwd, filePath));
      
      // Get conversation history
      const history = agent.getConversationHistory();
      
      // Format conversation
      const formatted = history.map(msg => `${msg.role.toUpperCase()}: ${msg.content}`).join('\n\n');
      
      // Write to file
      const fs = require('fs').promises;
      await fs.writeFile(fullPath, formatted, 'utf8');
      
      outputRenderer.addSystemMessage(`Conversation saved to: ${filePath}`);
    } catch (error) {
      logger.error(`Failed to save conversation: ${filePath}`, { error });
      outputRenderer.addErrorMessage(`Error saving conversation: ${error.message}`);
    }
  }
  
  // Initialize immediately
  init();
  
  // Return the input handler interface
  return {
    processInput,
    processCommand,
    handleFileModifications
  };
}