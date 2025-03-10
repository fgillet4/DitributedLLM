/**
 * Confirmation UI Component for FrankCode Agent
 * 
 * This module provides UI components for confirming file operations
 * with visual diffs and interactive prompts.
 */

const blessed = require('blessed');
const chalk = require('chalk');
const { logger } = require('../utils/logger');

/**
 * Create a confirmation dialog for file operations
 * 
 * @param {Object} screen The blessed screen object
 * @param {Object} operation The operation result (from fileOperations.js)
 * @returns {Promise<string>} User's choice ('yes', 'no', 'yes-to-all', 'custom')
 */
function createConfirmationDialog(screen, operation) {
  return new Promise((resolve) => {
    // Create a box for the confirmation dialog
    const box = blessed.box({
      top: 'center',
      left: 'center',
      width: '80%',
      height: '80%',
      border: {
        type: 'line',
        fg: 'blue'
      },
      tags: true,
      keys: true,
      vi: true,
      mouse: true,
      scrollable: true,
      alwaysScroll: true,
      scrollbar: {
        ch: ' ',
        bg: 'blue'
      }
    });
    
    // Create title based on operation type
    let title;
    switch (operation.operation) {
      case 'update':
        title = `{bold}Confirm Update: ${operation.filePath}{/bold}`;
        break;
      case 'create':
        title = `{bold}Confirm Creation: ${operation.filePath}{/bold}`;
        break;
      case 'delete':
        title = `{bold}Confirm Deletion: ${operation.filePath}{/bold}`;
        break;
      default:
        title = `{bold}Confirm Operation: ${operation.operation}{/bold}`;
    }
    
    // Add title
    box.setLine(0, title);
    box.setLine(1, '');
    
    // Add operation details
    let lineIndex = 2;
    
    if (operation.operation === 'update') {
      box.setLine(lineIndex++, `{yellow-fg}Changes: ${operation.additions} additions, ${operation.removals} removals{/yellow-fg}`);
      box.setLine(lineIndex++, '');
      
      // Add diff display
      if (operation.diffDisplay) {
        // Convert chalk colors to blessed tags for the terminal UI
        const diffDisplay = operation.diffDisplay
          .replace(/\u001b\[32m/g, '{green-fg}') // green
          .replace(/\u001b\[31m/g, '{red-fg}')   // red
          .replace(/\u001b\[90m/g, '{grey-fg}')  // grey
          .replace(/\u001b\[39m/g, '{/}');       // reset
          
        diffDisplay.split('\n').forEach((line) => {
          box.setLine(lineIndex++, line);
        });
      } else {
        // Simple diff
        const oldLines = operation.originalContent.split('\n');
        const newLines = operation.newContent.split('\n');
        
        for (let i = 0; i < Math.max(oldLines.length, newLines.length); i++) {
          const oldLine = oldLines[i] !== undefined ? oldLines[i] : '';
          const newLine = newLines[i] !== undefined ? newLines[i] : '';
          
          if (oldLine !== newLine) {
            if (oldLine) {
              box.setLine(lineIndex++, `{red-fg}- ${i + 1} ${oldLine}{/red-fg}`);
            }
            if (newLine) {
              box.setLine(lineIndex++, `{green-fg}+ ${i + 1} ${newLine}{/green-fg}`);
            }
          } else {
            box.setLine(lineIndex++, `  ${i + 1} ${oldLine}`);
          }
        }
      }
    } else if (operation.operation === 'create') {
      box.setLine(lineIndex++, `{green-fg}New file: ${operation.filePath}{/green-fg}`);
      box.setLine(lineIndex++, '');
      
      // Show content
      const contentLines = operation.content.split('\n');
      contentLines.forEach((line, index) => {
        box.setLine(lineIndex++, `{green-fg}+ ${index + 1} ${line}{/green-fg}`);
      });
    } else if (operation.operation === 'delete') {
      box.setLine(lineIndex++, `{red-fg}Delete file: ${operation.filePath}{/red-fg}`);
      box.setLine(lineIndex++, '');
      
      // Show content being deleted
      const contentLines = operation.content.split('\n');
      contentLines.forEach((line, index) => {
        box.setLine(lineIndex++, `{red-fg}- ${index + 1} ${line}{/red-fg}`);
      });
    }
    
    // Add empty line
    box.setLine(lineIndex++, '');
    
    // Add buttons
    const buttonsLine = lineIndex++;
    box.setLine(buttonsLine, '{center}{bold}[Y]{/bold}es | {bold}[N]{/bold}o | {bold}[A]{/bold}ll | {bold}[C]{/bold}ustom | {bold}[Esc]{/bold} Cancel{/center}');
    
    // Add the box to the screen
    screen.append(box);
    
    // Focus the box
    box.focus();
    
    // Handle key events
    box.key(['y', 'Y'], () => {
      cleanup();
      resolve('yes');
    });
    
    box.key(['n', 'N'], () => {
      cleanup();
      resolve('no');
    });
    
    box.key(['a', 'A'], () => {
      cleanup();
      resolve('yes-to-all');
    });
    
    box.key(['c', 'C'], () => {
      cleanup();
      resolve('custom');
    });
    
    box.key(['escape'], () => {
      cleanup();
      resolve('cancel');
    });
    
    // Render the screen
    screen.render();
    
    // Function to clean up the dialog
    function cleanup() {
      screen.remove(box);
      screen.render();
    }
  });
}

/**
 * Create a custom editor for modifying file content
 * 
 * @param {Object} screen The blessed screen object
 * @param {Object} operation The operation result (from fileOperations.js)
 * @returns {Promise<string>} Modified content or null if cancelled
 */
function createCustomEditor(screen, operation) {
  return new Promise((resolve) => {
    // Create a box for the editor
    const box = blessed.textarea({
      top: 'center',
      left: 'center',
      width: '90%',
      height: '90%',
      border: {
        type: 'line',
        fg: 'blue'
      },
      inputOnFocus: true,
      keys: true,
      vi: true,
      mouse: true,
      scrollable: true,
      alwaysScroll: true,
      scrollbar: {
        ch: ' ',
        bg: 'blue'
      }
    });
    
    // Set initial content
    let initialContent;
    if (operation.operation === 'update') {
      initialContent = operation.newContent;
    } else if (operation.operation === 'create') {
      initialContent = operation.content;
    } else {
      initialContent = '';
    }
    
    box.setValue(initialContent);
    
    // Create help text
    const helpText = blessed.box({
      bottom: 0,
      left: 'center',
      width: '100%',
      height: 1,
      content: '{bold}Ctrl-S{/bold} Save | {bold}Esc{/bold} Cancel',
      tags: true
    });
    
    // Create title
    const title = blessed.box({
      top: 0,
      left: 'center',
      width: '100%',
      height: 1,
      content: `{bold}Editing: ${operation.filePath}{/bold}`,
      tags: true
    });
    
    // Add the components to the screen
    screen.append(box);
    screen.append(helpText);
    screen.append(title);
    
    // Focus the box
    box.focus();
    
    // Handle key events
    box.key(['C-s'], () => {
      const content = box.getValue();
      cleanup();
      resolve(content);
    });
    
    box.key(['escape'], () => {
      cleanup();
      resolve(null);
    });
    
    // Render the screen
    screen.render();
    
    // Function to clean up the editor
    function cleanup() {
      screen.remove(box);
      screen.remove(helpText);
      screen.remove(title);
      screen.render();
    }
  });
}

/**
 * Show a confirmation dialog for a file operation
 * 
 * @param {Object} screen The blessed screen object
 * @param {Object} operation The operation result
 * @returns {Promise<Object>} Confirmation result with user's choice
 */
async function confirmOperation(screen, operation) {
  try {
    // Log the confirmation request
    logger.debug(`Requesting confirmation for operation: ${operation.operation}`, { filePath: operation.filePath });
    
    // Check if operation is valid
    if (operation.status === 'error') {
      return {
        choice: 'error',
        operation,
        error: operation.error
      };
    }
    
    // Show confirmation dialog
    const choice = await createConfirmationDialog(screen, operation);
    
    // Handle custom editing
    let customContent = null;
    if (choice === 'custom') {
      customContent = await createCustomEditor(screen, operation);
      
      // If cancelled, return cancel
      if (customContent === null) {
        return {
          choice: 'cancel',
          operation
        };
      }
      
      // Update the operation with custom content
      if (operation.operation === 'update' || operation.operation === 'create') {
        operation.newContent = customContent;
        operation.content = customContent;
      }
    }
    
    // Return the result
    return {
      choice,
      operation,
      customContent
    };
  } catch (error) {
    logger.error(`Failed to confirm operation: ${operation.operation}`, { error });
    return {
      choice: 'error',
      operation,
      error: error.message
    };
  }
}

module.exports = {
  confirmOperation,
  createConfirmationDialog,
  createCustomEditor
};