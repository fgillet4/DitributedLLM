/**
 * File Tree Component
 * 
 * Displays and manages the project file tree in the TUI
 */

const path = require('path');
const fs = require('fs').promises;
const { logger } = require('../utils');
const { listDirectoryFiles, getFileDetails } = require('../agent/fileManager');

/**
 * Create a file tree component
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.widget The blessed widget
 * @param {string} options.projectRoot Project root directory
 * @param {Object} options.agent Agent instance
 * @returns {Object} The file tree interface
 */
function createFileTree({ widget, projectRoot, agent }) {
  // Tree data structure
  let tree = {};
  
  // Currently selected file
  let selectedFilePath = null;
  
  /**
   * Initialize the file tree
   */
  async function init() {
    try {
      // Load tree data
      await loadTree();
      
      // Set up event handlers
      widget.on('select', node => {
        // Handle node selection
        if (node.isFile) {
          selectedFilePath = node.filePath;
          onFileSelected(node.filePath);
        }
      });
      
      // Set up key bindings for the tree
      widget.key(['enter'], () => {
        const node = widget.selectedNode;
        
        if (node && node.isFile) {
          onFileSelected(node.filePath);
        } else if (node) {
          node.expanded = !node.expanded;
          widget.setData(tree);
          widget.screen.render();
        }
      });
      
      widget.key(['r'], () => {
        refresh();
      });
    } catch (error) {
      logger.error('Failed to initialize file tree', { error });
    }
  }
  
  /**
   * Load the tree data
   */
  async function loadTree() {
    try {
      // Build the tree structure
      tree = await buildDirectoryTree(projectRoot);
      
      // Set the tree data
      widget.setData(tree);
      
      // Render the screen
      widget.screen.render();
    } catch (error) {
      logger.error('Failed to load file tree', { error });
    }
  }
  
  /**
   * Build a directory tree structure
   * 
   * @param {string} dirPath Directory path
   * @param {string} relativePath Relative path from project root
   * @returns {Object} Tree structure
   */
  async function buildDirectoryTree(dirPath, relativePath = '') {
    try {
      // Get directory contents
      const entries = await fs.readdir(dirPath, { withFileTypes: true });
      
      // Root node
      const rootName = relativePath === '' ? path.basename(dirPath) : path.basename(relativePath);
      const root = {
        name: rootName,
        extended: true,
        children: {}
      };
      
      // Separate directories and files
      const dirs = entries.filter(entry => entry.isDirectory() && !entry.name.startsWith('.'));
      const files = entries.filter(entry => entry.isFile());
      
      // Add directories first
      for (const dir of dirs) {
        const dirRelativePath = path.join(relativePath, dir.name);
        const dirFullPath = path.join(dirPath, dir.name);
        
        // Skip node_modules, .git, etc.
        if (['node_modules', '.git', 'dist', 'build'].includes(dir.name)) {
          continue;
        }
        
        // Recursively build children
        const childTree = await buildDirectoryTree(dirFullPath, dirRelativePath);
        
        // Add to root if it has children
        if (Object.keys(childTree.children).length > 0) {
          root.children[dir.name] = childTree;
        }
      }
      
      // Add files
      for (const file of files) {
        const fileRelativePath = path.join(relativePath, file.name);
        const fileFullPath = path.join(dirPath, file.name);
        
        // Create file node
        root.children[file.name] = {
          name: file.name,
          filePath: fileRelativePath,
          isFile: true
        };
      }
      
      return root;
    } catch (error) {
      logger.error(`Failed to build directory tree for ${dirPath}`, { error });
      return { name: path.basename(dirPath), extended: true, children: {} };
    }
  }
  
  /**
   * Refresh the file tree
   */
  async function refresh() {
    try {
      // Clear tree
      tree = {};
      
      // Reload tree data
      await loadTree();
      
      logger.info('File tree refreshed');
    } catch (error) {
      logger.error('Failed to refresh file tree', { error });
    }
  }
  
  /**
   * Handle file selection
   * 
   * @param {string} filePath Path to the file
   */
  async function onFileSelected(filePath) {
    try {
      // Log selection
      logger.debug(`File selected: ${filePath}`);
      
      // Get absolute path
      const fullPath = path.join(projectRoot, filePath);
      
      // Attempt to load file context
      await agent.loadFileContext(fullPath);
      
      // Notify user
      widget.screen.emit('file-selected', {
        path: filePath,
        fullPath
      });
    } catch (error) {
      logger.error(`Failed to process file selection: ${filePath}`, { error });
    }
  }
  
  /**
   * Get the currently selected file path
   * 
   * @returns {string} Selected file path
   */
  function getSelectedFile() {
    return selectedFilePath;
  }
  
  // Return the file tree interface
  return {
    init,
    refresh,
    getSelectedFile
  };
}

module.exports = {
  createFileTree
};