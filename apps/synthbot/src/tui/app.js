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
  // Create a screen object
  const screen = blessed.screen({
    smartCSR: true,
    title: 'SynthBot',
    fullUnicode: true,
    dockBorders: true,
    autoPadding: true
  });
  
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
    mouse: true,
    tags: true
  });
  
  // Create input box (bottom right, span 10/12 of width)
  const inputBox = grid.set(8, 2, 2, 10, blessed.textarea, {
    label: 'Command',
    inputOnFocus: true,
    padding: {
      top: 1,
      left: 2
    },
    style: {
      focus: {
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
  return {
    screen,
    start: () => {
      // Initial rendering
      screen.render();
      
      // Welcome message
      outputRenderer.addSystemMessage('Welcome to SynthBot! Type your question or command below.');
      statusBarController.update('Ready');
    },
    destroy: () => {
      screen.destroy();
    }
  };
}

module.exports = {
  createApp
};