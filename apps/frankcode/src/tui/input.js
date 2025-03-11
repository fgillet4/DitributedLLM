/**
 * Input Handler for TUI
 * 
 * Manages user input and command processing
 */

const { logger } = require('../utils/logger');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);
const path = require('path');
const fs = require('fs').promises;
const fileOperations = require('../utils/fileOperations');

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
  
  // Agent command processor - will be set after initialization
  let agentCommandProcessor = null;
  let apiClient = null;
  if (screen && screen.apiClient) {
    apiClient = screen.apiClient;
  }
  // Initialize
  function init() {
    // Add event listener for keypress to ensure UI updates
    widget.on('keypress', function() {
      // Force a screen render after each key press
      screen.render();
    });

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
      widget.screen.render(); // Ensure UI updates
    });
  }
  /**
   * Clear the conversation history
   */
  async function clearConversation() {
    try {
      // Reset the agent's conversation history
      agent.reset();
      
      // Clear the output display
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
   * Generate a summary of the conversation history
   */
  async function generateSummary() {
    try {
      // Get conversation history
      const history = agent.getConversationHistory();
      
      if (!history || history.length === 0) {
        outputRenderer.addSystemMessage('No conversation history to summarize.');
        return;
      }
      
      // Format the conversation
      const formattedHistory = history.map(msg => {
        // Clean up the content (remove thinking tags)
        const cleanContent = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
        return `${msg.role.toUpperCase()}: ${cleanContent.substring(0, 300)}${cleanContent.length > 300 ? '...' : ''}`;
      }).join('\n\n');
      
      // Create prompt for the LLM
      const prompt = `Create a concise summary of this conversation between a user and FrankCode (an AI coding assistant). Format it with clear sections for:

  1. Main topics and tasks discussed
  2. Problems solved or features implemented
  3. Files modified or discussed
  4. Next steps based on the conversation
  5. The most important achievements

  Use bold headers (**Header:**) and bullet points. Here's the conversation:

  ${formattedHistory}`;
      
      // Generate summary using the LLM
      outputRenderer.addSystemMessage('Generating conversation summary...');
      
      const response = await apiClient.generateResponse(prompt, {
        temperature: 0.3,
        maxTokens: 1024
      });
      
      return response.text;
    } catch (error) {
      logger.error('Failed to generate summary', { error });
      outputRenderer.addErrorMessage(`Error generating summary: ${error.message}`);
      return 'Failed to generate summary.';
    }
  }

  /**
   * Compact the conversation with a summary
   */
  async function compactConversation() {
    try {
      // Generate summary
      const summary = await generateSummary();
      
      if (!summary) {
        return;
      }
      
      // Reset the agent state
      agent.reset();
      
      // Add summary as context
      agent.addSystemContext(summary);
      
      // Clear the output display
      outputRenderer.clear();
      
      // Display the summary
      outputRenderer.addSystemMessage('***Session Summary***');
      
      // Format and display the summary
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
   * Export the conversation to a file
   */
  async function exportConversation(filePath) {
    try {
      // Create default filename if none provided
      if (!filePath) {
        filePath = `conversation_${new Date().toISOString().replace(/[:.]/g, '-')}.md`;
      }
      
      // Get absolute path
      const fullPath = path.isAbsolute(filePath) ? filePath : path.join(screen.cwd, filePath);
      
      // Get conversation history
      const history = agent.getConversationHistory();
      
      if (!history || history.length === 0) {
        outputRenderer.addSystemMessage('No conversation history to export.');
        return;
      }
      
      // Generate summary
      const summary = await generateSummary();
      
      // Format the conversation (remove thinking tags)
      const formattedHistory = history.map(msg => {
        const cleanContent = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
        return `${msg.role.toUpperCase()}:\n${cleanContent}\n\n---\n\n`;
      }).join('');
      
      
       // Try to generate a summary, but handle errors gracefully
       let summaryText = "Failed to generate summary.";    try {
      if (apiClient && history.length > 1) {
        const formattedHistoryForSummary = history.map(msg => {
          const cleanContent = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
          return `${msg.role.toUpperCase()}: ${cleanContent.substring(0, 300)}${cleanContent.length > 300 ? '...' : ''}`;
        }).join('\n\n');
        
        const summaryPrompt = `Create a concise summary of this conversation between a user and FrankCode (an AI coding assistant). Format it with clear sections for:

1. Main topics and tasks discussed
2. Problems solved or features implemented
3. Files modified or discussed
4. Next steps based on the conversation
5. The most important achievements

Use bold headers (**Header:**) and bullet points. Here's the conversation:

${formattedHistoryForSummary}`;
        
        outputRenderer.addSystemMessage('Generating conversation summary...');
        
        const response = await apiClient.generateResponse(summaryPrompt, {
          temperature: 0.3,
          maxTokens: 1024
        });
        
        if (response && response.text) {
          summary = response.text;
        }
      }
    } catch (error) {
      logger.error('Failed to generate summary', { error });
      outputRenderer.addErrorMessage(`Error generating summary: ${error.message}`);
      // Continue with the export even if summary generation fails
    }
    
    // Combine summary and history
    const content = `# Conversation Summary\n\n${summary}\n\n# Full Conversation\n\n${formattedHistory}`;

      
      // Ensure directory exists
      const dirPath = path.dirname(fullPath);
    try {
      // Use fs directly without promises (for compatibility)
      const fs = require('fs');
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
      }
      
      // Write to file using synchronous method (more reliable)
      fs.writeFileSync(fullPath, content, 'utf8');
      
      outputRenderer.addSystemMessage(`Conversation exported to ${fullPath}`);
      logger.info(`Conversation exported to ${fullPath}`);
      return true;
    } catch (dirError) {
      logger.error(`Failed to create directory: ${dirError.message}`);
      
      // Try writing to the current directory as fallback
      try {
        const fallbackPath = path.join('.', path.basename(filePath));
        fs.writeFileSync(fallbackPath, content, 'utf8');
        outputRenderer.addSystemMessage(`Conversation exported to ${fallbackPath} (fallback location)`);
        logger.info(`Conversation exported to ${fallbackPath} (fallback)`);
        return true;
      } catch (fallbackError) {
        throw new Error(`Failed to write to fallback location: ${fallbackError.message}`);
      }
    }
  } catch (error) {
    logger.error('Failed to export conversation', { error });
    outputRenderer.addErrorMessage(`Error exporting conversation: ${error.message}`);
    return false;
  }
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
      
      // Process file modifications directly if the input matches our format
      const fileBlockRegex = /\`\`\`file:(.*?)\n([\s\S]*?)\`\`\`/g;
      let match;
      const fileModifications = [];
  
      while ((match = fileBlockRegex.exec(input)) !== null) {
        const filePath = match[1].trim();
        const content = match[2];
        
        fileModifications.push({
          filePath,
          content
        });
      }
  
      if (fileModifications.length > 0) {
        await handleFileModifications(fileModifications);
        return;
      }
  
      // Check if this might be an agent task
      if (agentCommandProcessor && agentCommandProcessor.isPotentialAgentTask(input)) {
        const wasHandled = await agentCommandProcessor.processCommand(input);
        if (wasHandled) {
          return;
        }
      }
      
      try {
        // Send to agent for processing
        const response = await agent.processMessage(input);
        
        // Display assistant response
        if (response && response.text) {
          outputRenderer.addAssistantMessage(response.text);
          
          // Check for file modifications
          if (response.fileModifications && response.fileModifications.length > 0) {
            await handleFileModifications(response.fileModifications);
          }
          
          // Check if this looks like a potential agent task that wasn't recognized
          const suggestion = agentCommandProcessor ? agentCommandProcessor.suggestAgentCapability(input) : null;
          if (suggestion) {
            outputRenderer.addSystemMessage(`💡 ${suggestion}`);
            
            // Show examples occasionally (20% chance)
            if (Math.random() < 0.2) {
              const examples = agentCommandProcessor.getExampleCommands();
              const randomExample = examples[Math.floor(Math.random() * examples.length)];
              outputRenderer.addSystemMessage(`Example: "${randomExample}"`);
            }
          }
        } else {
          outputRenderer.addErrorMessage('No response received from the agent.');
        }
      } catch (processingError) {
        // Handle specific agent processing errors in a cleaner way
        logger.error('Agent processing error', { processingError });
        
        let errorMessage = processingError.message || 'Unknown error';
        
        // Check for common connection errors
        if (errorMessage.includes('ECONNREFUSED')) {
          outputRenderer.addErrorMessage(`Connection to Ollama failed. Try running with --offline flag or make sure Ollama is running with 'ollama serve'`);
        } else {
          outputRenderer.addErrorMessage(`Error: ${errorMessage}`);
        }
        
        // Force render to recover the UI
        widget.screen.render();
      }
    } catch (error) {
      logger.error('Failed to process input', { error });
      outputRenderer.addErrorMessage(`Error processing input: ${error.message}`);
      
      // Force render to recover the UI
      widget.screen.render();
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
          await clearConversation();
          break;
        
        case 'compact':
          await compactConversation();
          break;
        
        case 'export':
          await exportConversation(args[0]);
          break;

        case 'exit':
        case 'quit':
          process.exit(0);
          break;
        
        case 'refresh':
          await fileTree.refresh();
          outputRenderer.addSystemMessage('File tree refreshed');
          break;
        
        case 'bash':
        case 'exec':
        case 'shell':
          await executeShellCommand(args.join(' '));
          break;
          
        case 'ls':
          await executeShellCommand('ls -la');
          break;
          
        case 'pwd':
          await executeShellCommand('pwd');
          break;
          
        case 'models':
          await listAndSelectModels();
          break;
          
        case 'selectmodel':
          if (!args[0]) {
            outputRenderer.addErrorMessage('Please specify a model name or number');
            return;
          }
          await selectModel(args[0]);
          break;

        case 'create':
          if (args.length < 1) {
            outputRenderer.addErrorMessage('Usage: /create <filename> [content]');
            return;
          }
          const filename = args[0];
          const content = args.slice(1).join(' ') || '';
          await createFile(filename, content);
          break;

        case 'edit':
          if (args.length < 1) {
            outputRenderer.addErrorMessage('Usage: /edit <filename> [content]');
            return;
          }
          const editFilename = args[0];
          const editContent = args.slice(1).join(' ');
          await editFile(editFilename, editContent);
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
          
        case 'offline':
          setOfflineMode(true);
          break;
          
        case 'online':
          setOfflineMode(false);
          break;
          case 'plan':

        case 'execute':
          if (agentEnhancer) {
            if (!args[0]) {
              outputRenderer.addErrorMessage('Please specify a task to execute');
              return;
            }
            const task = args.join(' ');
            await agentEnhancer.executeTask(task);
          } else {
            outputRenderer.addErrorMessage('Task execution not available');
          }
          break;

        case 'code':
          if (agentEnhancer) {
            if (args.length < 2) {
              outputRenderer.addErrorMessage('Usage: /code language filePath');
              return;
            }
            const [language, filePath] = args;
            const content = await agent.generateCode(language);
            if (content) {
              await agentEnhancer.createFile(filePath, content);
            }
          } else {
            outputRenderer.addErrorMessage('Code generation not available');
          }
          break;

        case 'run':
          if (agentEnhancer) {
            if (!args[0]) {
              outputRenderer.addErrorMessage('Please specify a code file to run');
              return;
            }
            const filePath = args[0];
            const language = path.extname(filePath).substring(1); // Get extension without dot
            const content = await readFile(filePath);
            if (content) {
              await agentEnhancer.executeCode(content, language);
            }
          } else {
            outputRenderer.addErrorMessage('Code execution not available');
          }
          break;

          case 'agent':
            if (args.length > 0) {
              const task = args.join(' ');
              if (agentCommandProcessor) {
                await agentCommandProcessor.executeTask(task);
              } else {
                outputRenderer.addErrorMessage('Agent command processor not initialized');
              }
            } else {
              outputRenderer.addSystemMessage('Please provide a task for the agent');
            }
            break;
        
        default:
          outputRenderer.addSystemMessage(`Unknown command: ${cmd}. Type /help for available commands.`);
      }
      
      // Force a render after processing command
      widget.screen.render();
    } catch (error) {
      logger.error('Failed to process command', { error });
      outputRenderer.addErrorMessage(`Error processing command: ${error.message}`);
      widget.screen.render();
    }
  }
  
  /**
   * Select a model to use
   * 
   * @param {string} modelIdentifier Model name or number
   */
  async function selectModel(modelIdentifier) {
    try {
      // Get the endpoint (default to localhost if not set)
      const endpoint = widget.screen.ollamaEndpoint || 'http://localhost:11434';
      
      const response = await fetch(`${endpoint}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      const models = data.models || [];
  
      
      if (models.length === 0) {
        outputRenderer.addSystemMessage('No models found');
        return;
      }
      
      // Find the model by name or index
      let selectedModel;
      
      if (!isNaN(parseInt(modelIdentifier))) {
        // Select by index
        const index = parseInt(modelIdentifier) - 1;
        if (index >= 0 && index < models.length) {
          selectedModel = models[index];
        }
      } else {
        // Select by name
        selectedModel = models.find(m => m.name === modelIdentifier);
      }
      
      if (!selectedModel) {
        outputRenderer.addErrorMessage(`Model "${modelIdentifier}" not found`);
        return;
      }
      
      // Update the apiClient
      const model = selectedModel.name;
      
      // Try to connect with new model
      outputRenderer.addSystemMessage(`Switching to model: ${model}`);
      
      try {
        // Create new client with selected model
        const url = new URL(endpoint);
    
        const newClient = createClient({
          host: url.hostname,
          port: parseInt(url.port),
          model,
          temperature: 0.7,
          api: 'ollama'
        });
        
        // Replace the old client in the agent
        agent.setApiClient(newClient);
        
        outputRenderer.addSystemMessage(`Successfully switched to model: ${model}`);
        
        // Update status bar without using screen reference
        if (widget && widget.screen && widget.screen.statusBar) {
          widget.screen.statusBar.update(`Model: ${model}`);
        }
      } catch (error) {
        outputRenderer.addErrorMessage(`Failed to switch model: ${error.message}`);
      }
    } catch (error) {
      outputRenderer.addErrorMessage(`Failed to fetch models: ${error.message}`);
    }
  }
  
  /**
   * List and select models from Ollama
   */
  async function listAndSelectModels() {
    try {
      outputRenderer.addSystemMessage('Fetching available models from Ollama...');
      
      // Try different possible Ollama endpoints
      const endpoints = [
        'http://localhost:11434/api/tags',
        'http://127.0.0.1:11434/api/tags'
      ];
      
      // If screen.config exists, also try that host
      if (widget.screen && widget.screen.config && widget.screen.config.llm) {
        const configHost = widget.screen.config.llm.coordinatorHost;
        const configPort = widget.screen.config.llm.coordinatorPort;
        endpoints.push(`http://${configHost}:${configPort}/api/tags`);
      }
      
      let response = null;
      let endpoint = null;
      
      // Try each endpoint until one works
      for (const ep of endpoints) {
        try {
          const resp = await fetch(ep);
          if (resp.ok) {
            response = resp;
            endpoint = ep;
            break;
          }
        } catch (err) {
          // Continue to next endpoint
        }
      }
      
      if (!response) {
        throw new Error('Could not connect to any Ollama endpoint');
      }
      
      const data = await response.json();
      const models = data.models || [];
      
      if (models.length === 0) {
        outputRenderer.addSystemMessage('No models found. Make sure Ollama is running with `ollama serve`');
        return;
      }
      
      // Display available models
      outputRenderer.addSystemMessage(`Found ${models.length} models:`);
      models.forEach((model, index) => {
        const size = model.size ? `(${Math.round(model.size / (1024 * 1024))} MB)` : '(unknown size)';
        outputRenderer.addSystemMessage(`${index + 1}. ${model.name} ${size}`);
      });
      
      // Prompt to select a model
      outputRenderer.addSystemMessage('Type /selectmodel <number> or /selectmodel <name> to select a model');
      
      // Store the endpoint for future use
      widget.screen.ollamaEndpoint = endpoint.replace('/api/tags', '');
    } catch (error) {
      outputRenderer.addErrorMessage(`Failed to fetch models: ${error.message}`);
      outputRenderer.addSystemMessage('Make sure Ollama is running with `ollama serve`. Try running: /offline to continue without LLM.');
    }
  }
  
  /**
   * Show help text
   */
  function showHelp() {
    const helpText = `
  Available commands:
    /help             - Show this help text
    /clear            - Clear the conversation history
    /compact          - Compact conversation history into a summary
    /export [file]    - Export conversation with summary to a file 
    /refresh          - Refresh the file tree
    /plan task        - Break down a task into smaller steps
    /execute task     - Execute a task with step-by-step guidance
    /code lang file   - Generate code in specified language and save to file
    /run file         - Run a code file and show output
    /exec <command>   - Execute a shell command
    /shell <command>  - Same as /exec
    /ls               - List files (shortcut for /exec ls)
    /pwd              - Show current directory (shortcut for /exec pwd)
    /models           - List available Ollama models
    /selectmodel <n>  - Select a model by number or name
    /offline          - Switch to offline mode (no LLM)
    /online           - Switch to online mode (try connecting to LLM)
    /load <file>      - Load a file into context
    /save <file>      - Save the conversation to a file
    /reset            - Reset agent context and conversation
    /yes, /y          - Approve the current file modification
    /no, /n           - Reject the current file modification
    /yesall, /ya      - Approve all pending file modifications
    /noall, /na       - Reject all pending file modifications
    /create <file>    - Create a new file
    /edit <file>      - Edit an existing file

  File Operations:
    You can create or modify files by using the syntax:
    \`\`\`file:path/to/file.js
    // File content here
    \`\`\`

  Keyboard shortcuts:
    Ctrl+C            - Exit application
    Ctrl+R            - Refresh file tree
    Ctrl+L            - Clear conversation
    Ctrl+S            - Save conversation
    Ctrl+F            - Focus conversation panel (scroll mode)
    Tab               - Cycle focus between panels
    PageUp/PageDown   - Scroll conversation when focused
    Up/Down arrows    - Scroll conversation when focused
      `;
      
    outputRenderer.addSystemMessage(helpText);
  }

  /**
   * Set offline mode
   * 
   * @param {boolean} offline Whether to enable offline mode
   */
  async function setOfflineMode(offline) {
    try {
      const { createClient } = require('../api');
      
      if (offline) {
        // Create a dummy client for offline mode
        const offlineClient = {
          generateResponse: async (prompt) => ({ 
            text: "Running in offline mode. LLM services are not available.\n\nYour prompt was:\n" + prompt,
            tokens: 0 
          }),
          streamResponse: async (prompt, onToken, onComplete) => {
            onToken("Running in offline mode. LLM services are not available.");
            onComplete({ tokens: 0 });
          },
          getConnectionStatus: () => false,
          getModelInfo: async () => ({ name: 'offline', parameters: {} })
        };
        
        // Set in the agent
        agent.setApiClient(offlineClient);
        outputRenderer.addSystemMessage("Switched to offline mode. LLM services will not be used.");
      } else {
        // Try to create an online client
        try {
          const host = widget.screen.config?.llm?.coordinatorHost || 'localhost';
          const port = widget.screen.config?.llm?.coordinatorPort || 11434;
          const model = widget.screen.config?.llm?.model || 'llama2';
          
          const onlineClient = createClient({
            host,
            port,
            model,
            temperature: 0.7,
            api: 'ollama'
          });
          
          // Set in the agent
          agent.setApiClient(onlineClient);
          outputRenderer.addSystemMessage("Attempting to switch to online mode.");
        } catch (error) {
          outputRenderer.addErrorMessage(`Failed to go online: ${error.message}`);
        }
      }
    } catch (error) {
      outputRenderer.addErrorMessage(`Failed to change mode: ${error.message}`);
    }
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
        // Apply the modification using fileOperations
        await fileOperations.writeFile(mod.filePath, mod.content);
        
        outputRenderer.addSystemMessage(`✅ Applied changes to ${mod.filePath}`);
        
        // Refresh file tree
        await fileTree.refresh();
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
        // Apply the modification using fileOperations
        await fileOperations.writeFile(mod.filePath, mod.content);
        applied++;
      } catch (error) {
        logger.error(`Failed to apply modification to ${mod.filePath}`, { error });
        failed++;
      }
    }
    
    outputRenderer.addSystemMessage(`Applied ${applied}/${total} modifications (${failed} failed).`);
    
    // Refresh file tree if any changes were applied
    if (applied > 0) {
      await fileTree.refresh();
    }
    
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
 * Create a new file
 * 
 * @param {string} filePath File path
 * @param {string} content File content
 */
  async function createFile(filePath, content) {
    if (!filePath) {
      outputRenderer.addErrorMessage('No file specified');
      return;
    }
    
    try {
      // Use fileOperations service instead of direct fs calls
      await fileOperations.createFile(filePath, content);
      
      outputRenderer.addSystemMessage(`Created file: ${filePath}`);
      
      // Refresh file tree
      await fileTree.refresh();
    } catch (error) {
      logger.error(`Failed to create file: ${filePath}`, { error });
      outputRenderer.addErrorMessage(`Error creating file: ${error.message}`);
    }
  }

  /**
   * Edit an existing file
   * 
   * @param {string} filePath File path
   * @param {string} content New content
   */
  async function editFile(filePath, content) {
    if (!filePath) {
      outputRenderer.addErrorMessage('No file specified');
      return;
    }
    
    try {
      // Check if content is empty (just displaying the file)
      if (!content || content.trim() === '') {
        // Read the file using fileOperations
        const fileContent = await fileOperations.readFile(filePath);
        outputRenderer.addSystemMessage(`Content of ${filePath}:`);
        outputRenderer.addCodeBlock(fileContent);
        return;
      }
      
      // Update file using fileOperations
      await fileOperations.writeFile(filePath, content);
      
      outputRenderer.addSystemMessage(`Updated file: ${filePath}`);
    } catch (error) {
      logger.error(`Failed to edit file: ${filePath}`, { error });
      outputRenderer.addErrorMessage(`Error editing file: ${error.message}`);
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
      // Check if file exists using fileOperations
      const exists = await fileOperations.fileExists(filePath);
      
      if (!exists) {
        outputRenderer.addErrorMessage(`File not found: ${filePath}`);
        return;
      }
      
      // Resolve to full path for the agent
      const fullPath = fileOperations.resolvePath(filePath);
      
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
      // Get conversation history
      const history = agent.getConversationHistory();
      
      // Format conversation
      const formatted = history.map(msg => {
        return `${msg.role.toUpperCase()}:\n${msg.content}\n\n`;
      }).join('---\n\n');
      
      // Write to file using fileOperations
      await fileOperations.writeFile(filePath, formatted);
      
      outputRenderer.addSystemMessage(`Conversation saved to: ${filePath}`);
    } catch (error) {
      logger.error(`Failed to save conversation: ${filePath}`, { error });
      outputRenderer.addErrorMessage(`Error saving conversation: ${error.message}`);
    }
  }
  
  init();

  // Create conversation manager
  let conversationManager = null;
  try {
    conversationManager = createConversationManager({
      agent,
      outputRenderer,
      llm: apiClient
    });
    logger.debug('Conversation manager initialized');
  } catch (error) {
    logger.error('Failed to initialize conversation manager', { error });
  }
  let agentEnhancer = null;
  try {
    const { createAgentEnhancer } = require('../agent/agentEnhancer');
    agentEnhancer = createAgentEnhancer({
      agent,
      outputRenderer,
      apiClient
    });
    logger.debug('Agent enhancer initialized');
  } catch (error) {
    logger.error('Failed to initialize agent enhancer', { error });
  }
  

  // Return the input handler interface
  return {
    processInput,
    processCommand,
    handleFileModifications,
    agentCommandProcessor, // Allow setting from outside
  };
}

// Export the function
module.exports = {
  createInputHandler
};