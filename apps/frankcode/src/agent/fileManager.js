/**
 * File Manager for the Agent
 * 
 * Handles file operations including reading, writing, and listing files.
 */

const fs = require('fs').promises;
const path = require('path');
const fastGlob = require('fast-glob');
const ignore = require('ignore');
const { logger } = require('../utils');

/**
 * Read a file from the filesystem
 * 
 * @param {string} filePath Path to the file
 * @returns {Promise<string>} File content
 */
async function readFile(filePath) {
  try {
    const content = await fs.readFile(filePath, 'utf8');
    return content;
  } catch (error) {
    logger.error(`Error reading file: ${filePath}`, { error });
    throw error;
  }
}

/**
 * Write content to a file
 * 
 * @param {string} filePath Path to the file
 * @param {string} content Content to write
 * @returns {Promise<void>}
 */
async function writeFile(filePath, content) {
  try {
    // Ensure the directory exists
    const dir = path.dirname(filePath);
    await fs.mkdir(dir, { recursive: true });
    
    // Write the file
    await fs.writeFile(filePath, content);
  } catch (error) {
    logger.error(`Error writing file: ${filePath}`, { error });
    throw error;
  }
}

/**
 * Check if a file exists
 * 
 * @param {string} filePath Path to the file
 * @returns {Promise<boolean>} Whether the file exists
 */
async function checkFileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

/**
 * List all files in a directory, applying filters
 * 
 * @param {string} dirPath Path to the directory
 * @param {Object} options Options object
 * @param {Array<string>} options.exclude Patterns to exclude
 * @param {Array<string>} options.include Patterns to include
 * @returns {Promise<Array<string>>} List of file paths
 */
async function listDirectoryFiles(dirPath, options = {}) {
  try {
    const { exclude = [], include = ['**'] } = options;
    
    // Check for .gitignore and add to excludes
    let ignoreRules = ignore().add(exclude);
    
    try {
      const gitignorePath = path.join(dirPath, '.gitignore');
      const gitignoreExists = await checkFileExists(gitignorePath);
      
      if (gitignoreExists) {
        const gitignoreContent = await readFile(gitignorePath);
        ignoreRules = ignoreRules.add(gitignoreContent);
      }
    } catch (error) {
      logger.warn('Failed to process .gitignore', { error });
      // Continue without gitignore
    }
    
    // Use fast-glob to get all files
    const files = await fastGlob(include, {
      cwd: dirPath,
      onlyFiles: true,
      absolute: false,
      dot: true
    });
    
    // Apply ignore rules
    const filteredPaths = ignoreRules.filter(files);
    
    // Convert to absolute paths
    return filteredPaths.map(f => path.resolve(dirPath, f));
  } catch (error) {
    logger.error(`Error listing files in directory: ${dirPath}`, { error });
    throw error;
  }
}

/**
 * Get details about a file
 * 
 * @param {string} filePath Path to the file
 * @returns {Promise<Object>} File details
 */
async function getFileDetails(filePath) {
  try {
    const stats = await fs.stat(filePath);
    
    return {
      path: filePath,
      name: path.basename(filePath),
      extension: path.extname(filePath),
      size: stats.size,
      modified: stats.mtime,
      created: stats.birthtime,
      isDirectory: stats.isDirectory()
    };
  } catch (error) {
    logger.error(`Error getting file details: ${filePath}`, { error });
    throw error;
  }
}

/**
 * Create a directory
 * 
 * @param {string} dirPath Path to the directory
 * @returns {Promise<void>}
 */
async function createDirectory(dirPath) {
  try {
    await fs.mkdir(dirPath, { recursive: true });
  } catch (error) {
    logger.error(`Error creating directory: ${dirPath}`, { error });
    throw error;
  }
}

/**
 * Delete a file
 * 
 * @param {string} filePath Path to the file
 * @returns {Promise<void>}
 */
async function deleteFile(filePath) {
  try {
    await fs.unlink(filePath);
  } catch (error) {
    logger.error(`Error deleting file: ${filePath}`, { error });
    throw error;
  }
}

module.exports = {
  readFile,
  writeFile,
  checkFileExists,
  listDirectoryFiles,
  getFileDetails,
  createDirectory,
  deleteFile
};