/**
 * Main Terminal UI Application
 * 
 * Creates and manages the terminal interface using blessed/blessed-contrib
 */

const blessed = require('blessed');
const contrib = require('blessed-contrib');
const { applyTheme } = require('./themes');
const { createInputHandler } = require('./input');
const { createOutputRenderer } = require('./output');
const { createStatusBar } = require('./statusBar');
const { createFileTree } = require('./fileTree');
const { logger } = require('../utils');
const { createAgentCommandProcessor } = require('../agent/agentUtils');

/**
 * Create the TUI application
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.agent The agent instance
 * @param {Object} options.apiClient The API client
 * @param {Object} options.tokenMonitor Token monitoring utility
 * @param {Object} options.config UI configuration
 * @param {string} options.projectRoot Project root directory
 * @returns {Object} The TUI application object
 */
function createApp({ agent, apiClient, tokenMonitor, config, projectRoot }) {
  // Create screen object
  const screen = blessed.screen({
    smartCSR: true,
    title: 'FrankCode',
    fullUnicode: true,
    dockBorders: true,
    autoPadding: true,
    sendFocus: false,
    mouseEnabled: true,
    useMouse: true
  });

  // Store config and project root in screen for access by components
  screen.config = config;
  screen.cwd = projectRoot;
  
  // Apply theme
  applyTheme(screen, config.theme);
  
  // Create layout grid
  const grid = new contrib.grid({ rows: 12, cols: 12, screen });
  
  // Create file tree panel (left side, 2/12 of width)
  const fileTreePanel = grid.set(0, 0, 10, 2, contrib.tree, {
    label: 'Project Files',
    style: {
      selected: {
        fg: 'white',
        bg: 'blue'
      }
    },
    template: {
      lines: true
    },
    tags: true
  });
  
  // Create conversation panel (right side, 10/12 of width)
  const conversationPanel = grid.set(0, 2, 8, 10, blessed.log, {
    label: 'Conversation',
    scrollable: true,
    alwaysScroll: true,
    scrollbar: {
      ch: ' ',
      style: {
        bg: 'blue'
      },
      track: {
        bg: 'black'
      }
    },
    // Add these settings for smoother scrolling
    mouse: true,
    scrollable: true,
    alwaysScroll: true,
    scrollbar: true,
    keys: true,
    vi: false,  // Turn off vi mode to enable normal selection
    inputOnFocus: false,
    // Increase scroll amount for smoother experience
    scrollAmount: 3,
    // Lower scroll time for smoother animation
    scrollSpeed: 10,
    // Allow mousewheel scrolling
    wheelBehavior: 'scroll',
    // Improved border style
    border: {
      type: 'line',
      fg: 'blue'
    },
    // Better padding for content
    padding: {
      left: 1,
      right: 1
    }
  });
  
  // Add additional key bindings for scrolling
  conversationPanel.key(['pageup'], () => {
    conversationPanel.setScroll(conversationPanel.getScroll() - conversationPanel.height + 2);
    screen.render();
  });
  
  conversationPanel.key(['pagedown'], () => {
    conversationPanel.setScroll(conversationPanel.getScroll() + conversationPanel.height - 2);
    screen.render();
  });
  
  // Allow using up/down arrow keys to scroll when focus is on conversation panel
  conversationPanel.key(['up'], () => {
    conversationPanel.scroll(-3); // Scroll up 3 lines
    screen.render();
  });
  
  conversationPanel.key(['down'], () => {
    conversationPanel.scroll(3); // Scroll down 3 lines
    screen.render();
  });
  
  // Add a key binding to toggle focus to the conversation panel for scrolling
  screen.key(['C-f'], () => {
    conversationPanel.focus();
    statusBarController.update('Scroll mode active. Press Tab to return to input.');
    screen.render();
  });
  
  // Update the tab key handler to cycle through all elements
  screen.key(['tab'], () => {
    if (screen.focused === inputBox) {
      fileTreePanel.focus();
    } else if (screen.focused === fileTreePanel) {
      conversationPanel.focus();
    } else {
      inputBox.focus();
    }
    screen.render();
  });
  
  // Add a mouse handler for better wheel behavior
  conversationPanel.on('wheeldown', () => {
    conversationPanel.scroll(3); // Scroll down 3 lines
    screen.render();
  });
  
  conversationPanel.on('wheelup', () => {
    conversationPanel.scroll(-3); // Scroll up 3 lines
    screen.render();
  });
  
  const inputBox = grid.set(8, 2, 2, 10, blessed.textarea, {
    label: 'Command',
    inputOnFocus: true,
    padding: {
      top: 1,
      left: 2
    },
    style: {
      fg: 'white',
      bg: 'black',
      focus: {
        fg: 'white',
        bg: 'black',
        border: {
          fg: 'blue'
        }
      }
    },
    border: {
      type: 'line'
    }
  });
  
  // Create status bar (bottom of screen, full width)
  const statusBar = grid.set(10, 0, 2, 12, blessed.box, {
    tags: true,
    content: ' {bold}Status:{/bold} Ready',
    style: {
      fg: 'white',
      bg: 'blue'
    },
    padding: {
      left: 1,
      right: 1
    }
  });
  
  // Initialize components
  const fileTree = createFileTree({
    widget: fileTreePanel,
    projectRoot,
    agent
  });
  
  const outputRenderer = createOutputRenderer({
    widget: conversationPanel,
    tokenMonitor
  });
  
  const statusBarController = createStatusBar({
    widget: statusBar,
    apiClient,
    tokenMonitor
  });
  
  const inputHandler = createInputHandler({
    widget: inputBox,
    outputRenderer,
    agent,
    fileTree,
    screen
  });

  try {
    const agentCommandProcessor = createAgentCommandProcessor({
      agent,
      llm: apiClient,
      screen,
      outputRenderer
    });
  // Add to input handler
  inputHandler.agentCommandProcessor = agentCommandProcessor;
    
  // Add help for agent commands
  if (agentCommandProcessor) {
    screen.key(['F1'], () => {
      const exampleCommands = agentCommandProcessor.getExampleCommands();
      
      outputRenderer.addSystemMessage('\n📚 Agent Command Examples:');
      exampleCommands.forEach(example => {
        outputRenderer.addSystemMessage(`• ${example}`);
      });
      
      outputRenderer.addSystemMessage('\nTry using these command patterns or press F1 again for more examples.');
      screen.render();
    });
  }
} catch (error) {
  logger.error('Failed to initialize agent command processor:', error);
  // Continue without agent capabilities
}
  const renderInterval = setInterval(() => {
    screen.render();
  }, 100);
  
  // Set up key bindings
  screen.key(['C-c'], () => {
    return process.exit(0);
  });
  
  screen.key(['tab'], () => {
    if (screen.focused === inputBox) {
      fileTreePanel.focus();
    } else {
      inputBox.focus();
    }
  });
  
  screen.key(['C-r'], () => {
    fileTree.refresh();
    statusBarController.update('Refreshing project files...');
  });
  
  screen.key(['C-l'], () => {
    conversationPanel.setContent('');
    screen.render();
    statusBarController.update('Conversation cleared');
  });
  
  screen.key(['C-s'], () => {
    // Save conversation implementation
    statusBarController.update('Conversation saved');
  });
  
  // Focus input by default
  inputBox.focus();
  
  // Initialize file tree
  fileTree.init();
  
  

// Return the application object
// Return the application object
return {
  screen,
  statusBar: statusBarController,
  start: () => {
    // Initial rendering
    screen.render();
    
    // Welcome message
    outputRenderer.addSystemMessage('Welcome to FrankCode! Type your question or command below.');
    statusBarController.update('Ready');
    
    // Other initialization...
  },
  destroy: () => {
    // Cleanup code...
    screen.destroy();
  }
};
}

module.exports = {
createApp
};