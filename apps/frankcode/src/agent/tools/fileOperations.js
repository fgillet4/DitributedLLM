/**
 * File Operation Tools for FrankCode Agent
 * 
 * This module provides tools for the agent to perform file operations
 * such as reading, writing, searching, and updating files.
 */

const fs = require('fs').promises;
const path = require('path');
const chalk = require('chalk');
const { promisify } = require('util');
const glob = promisify(require('glob'));
const { logger } = require('../../utils/logger');
const diff = require('diff');

/**
 * Read a file and display the content with line numbers
 * 
 * @param {string} filePath Path to the file
 * @param {number} offset Line number to start reading from (0-indexed)
 * @param {number} limit Maximum number of lines to read
 * @returns {Promise<Object>} File content with metadata
 */
async function readFile(filePath, offset = 0, limit = -1) {
  try {
    // Log the operation
    logger.info(`Reading file: ${filePath}`, { offset, limit });
    
    // Make sure the file exists
    try {
      await fs.access(filePath);
    } catch (error) {
      throw new Error(`File not found: ${filePath}`);
    }
    
    // Read the file content
    const content = await fs.readFile(filePath, 'utf8');
    const lines = content.split('\n');
    
    // Apply offset and limit
    const startLine = Math.max(0, offset);
    const endLine = limit > 0 ? Math.min(lines.length, startLine + limit) : lines.length;
    const selectedLines = lines.slice(startLine, endLine);
    
    // Format the display content with line numbers
    const displayContent = selectedLines.map((line, index) => {
      const lineNumber = startLine + index + 1;
      return `${lineNumber.toString().padStart(6)} ${line}`;
    }).join('\n');
    
    // Return the result
    return {
      operation: 'read',
      filePath,
      content: displayContent,
      rawContent: selectedLines.join('\n'),
      startLine,
      endLine: endLine - 1,
      totalLines: lines.length,
      status: 'success'
    };
  } catch (error) {
    logger.error(`Failed to read file: ${filePath}`, { error });
    return {
      operation: 'read',
      filePath,
      error: error.message,
      status: 'error'
    };
  }
}

/**
 * Search for files matching a pattern
 * 
 * @param {string} pattern Glob pattern to match files
 * @param {string} baseDir Base directory for the search
 * @returns {Promise<Object>} Search results
 */
async function searchFiles(pattern, baseDir = process.cwd()) {
  try {
    // Log the operation
    logger.info(`Searching for files: ${pattern} in ${baseDir}`);
    
    // Normalize the pattern
    const normalizedPattern = path.isAbsolute(pattern) 
      ? pattern 
      : path.join(baseDir, pattern);
    
    // Find matching files
    const matches = await glob(normalizedPattern, { nodir: true });
    
    // Return the result
    return {
      operation: 'search',
      pattern,
      baseDir,
      matches,
      matchCount: matches.length,
      status: 'success'
    };
  } catch (error) {
    logger.error(`Failed to search for files: ${pattern}`, { error });
    return {
      operation: 'search',
      pattern,
      baseDir,
      error: error.message,
      status: 'error'
    };
  }
}

/**
 * Update a file with new content
 * 
 * @param {string} filePath Path to the file
 * @param {string} newContent New file content
 * @param {boolean} showDiff Whether to show the diff
 * @returns {Promise<Object>} Update result with diff
 */
async function updateFile(filePath, newContent, showDiff = true) {
  try {
    // Log the operation
    logger.info(`Updating file: ${filePath}`);
    
    // Make sure the file exists
    let originalContent = '';
    try {
      originalContent = await fs.readFile(filePath, 'utf8');
    } catch (error) {
      throw new Error(`File not found: ${filePath}`);
    }
    
    // Generate diff
    const changes = diff.diffLines(originalContent, newContent);
    
    // Count additions and removals
    let additions = 0;
    let removals = 0;
    
    changes.forEach(change => {
      if (change.added) {
        additions += change.count;
      } else if (change.removed) {
        removals += change.count;
      }
    });
    
    // Generate diff display
    let diffDisplay = '';
    if (showDiff) {
      const originalLines = originalContent.split('\n');
      let lineNumber = 1;
      
      changes.forEach(change => {
        if (change.added) {
          // Green for additions
          change.value.split('\n').forEach(line => {
            if (line === '') return;
            diffDisplay += chalk.green(`+ ${lineNumber} ${line}\n`);
          });
        } else if (change.removed) {
          // Red for removals
          change.value.split('\n').forEach(line => {
            if (line === '') return;
            diffDisplay += chalk.red(`- ${lineNumber++} ${line}\n`);
          });
        } else {
          // Grey for context
          change.value.split('\n').forEach(line => {
            if (line === '') return;
            diffDisplay += chalk.grey(`  ${lineNumber++} ${line}\n`);
          });
        }
      });
    }
    
    // Return the result
    return {
      operation: 'update',
      filePath,
      diffDisplay,
      changes,
      additions,
      removals,
      originalContent,
      newContent,
      status: 'pending' // Requires confirmation
    };
  } catch (error) {
    logger.error(`Failed to update file: ${filePath}`, { error });
    return {
      operation: 'update',
      filePath,
      error: error.message,
      status: 'error'
    };
  }
}

/**
 * Confirm and apply file update
 * 
 * @param {Object} updateResult Result from updateFile
 * @returns {Promise<Object>} Final update result
 */
async function confirmUpdate(updateResult) {
  try {
    // Check if the update is valid
    if (updateResult.status === 'error') {
      throw new Error(`Cannot confirm an update with errors: ${updateResult.error}`);
    }
    
    // Apply the update
    await fs.writeFile(updateResult.filePath, updateResult.newContent);
    
    // Return the result
    return {
      operation: 'update',
      filePath: updateResult.filePath,
      additions: updateResult.additions,
      removals: updateResult.removals,
      status: 'success'
    };
  } catch (error) {
    logger.error(`Failed to confirm update: ${updateResult.filePath}`, { error });
    return {
      operation: 'update',
      filePath: updateResult.filePath,
      error: error.message,
      status: 'error'
    };
  }
}

/**
 * Create a new file
 * 
 * @param {string} filePath Path to the new file
 * @param {string} content File content
 * @returns {Promise<Object>} Creation result
 */
async function createFile(filePath, content) {
  try {
    // Log the operation
    logger.info(`Creating file: ${filePath}`);
    
    // Check if the file already exists
    try {
      await fs.access(filePath);
      throw new Error(`File already exists: ${filePath}`);
    } catch (error) {
      // File doesn't exist, which is what we want
      if (error.code !== 'ENOENT') {
        throw error;
      }
    }
    
    // Ensure the directory exists
    const dirPath = path.dirname(filePath);
    await fs.mkdir(dirPath, { recursive: true });
    
    // Return the result without actually creating the file (requires confirmation)
    return {
      operation: 'create',
      filePath,
      content,
      status: 'pending' // Requires confirmation
    };
  } catch (error) {
    logger.error(`Failed to create file: ${filePath}`, { error });
    return {
      operation: 'create',
      filePath,
      error: error.message,
      status: 'error'
    };
  }
}

/**
 * Confirm and apply file creation
 * 
 * @param {Object} createResult Result from createFile
 * @returns {Promise<Object>} Final creation result
 */
async function confirmCreate(createResult) {
  try {
    // Check if the creation is valid
    if (createResult.status === 'error') {
      throw new Error(`Cannot confirm a creation with errors: ${createResult.error}`);
    }
    
    // Apply the creation
    await fs.writeFile(createResult.filePath, createResult.content);
    
    // Return the result
    return {
      operation: 'create',
      filePath: createResult.filePath,
      status: 'success'
    };
  } catch (error) {
    logger.error(`Failed to confirm creation: ${createResult.filePath}`, { error });
    return {
      operation: 'create',
      filePath: createResult.filePath,
      error: error.message,
      status: 'error'
    };
  }
}

/**
 * Delete a file
 * 
 * @param {string} filePath Path to the file
 * @returns {Promise<Object>} Deletion result
 */
async function deleteFile(filePath) {
  try {
    // Log the operation
    logger.info(`Deleting file: ${filePath}`);
    
    // Make sure the file exists
    try {
      await fs.access(filePath);
    } catch (error) {
      throw new Error(`File not found: ${filePath}`);
    }
    
    // Read the file content before deleting (for confirmation)
    const content = await fs.readFile(filePath, 'utf8');
    
    // Return the result without actually deleting the file (requires confirmation)
    return {
      operation: 'delete',
      filePath,
      content,
      status: 'pending' // Requires confirmation
    };
  } catch (error) {
    logger.error(`Failed to delete file: ${filePath}`, { error });
    return {
      operation: 'delete',
      filePath,
      error: error.message,
      status: 'error'
    };
  }
}

/**
 * Confirm and apply file deletion
 * 
 * @param {Object} deleteResult Result from deleteFile
 * @returns {Promise<Object>} Final deletion result
 */
async function confirmDelete(deleteResult) {
  try {
    // Check if the deletion is valid
    if (deleteResult.status === 'error') {
      throw new Error(`Cannot confirm a deletion with errors: ${deleteResult.error}`);
    }
    
    // Apply the deletion
    await fs.unlink(deleteResult.filePath);
    
    // Return the result
    return {
      operation: 'delete',
      filePath: deleteResult.filePath,
      status: 'success'
    };
  } catch (error) {
    logger.error(`Failed to confirm deletion: ${deleteResult.filePath}`, { error });
    return {
      operation: 'delete',
      filePath: deleteResult.filePath,
      error: error.message,
      status: 'error'
    };
  }
}

module.exports = {
  readFile,
  searchFiles,
  updateFile,
  confirmUpdate,
  createFile,
  confirmCreate,
  deleteFile,
  confirmDelete
};