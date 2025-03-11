/**
 * File Operations Service
 * 
 * Provides a unified interface for all file operations in the application
 */

const fs = require('fs').promises;
const path = require('path');
const { logger } = require('../utils/logger');

class FileOperationsService {
  constructor(projectRoot) {
    this.projectRoot = projectRoot || process.cwd();
  }

  /**
   * Resolve a relative path to an absolute path
   * 
   * @param {string} relativePath Relative path
   * @returns {string} Absolute path
   */
  resolvePath(relativePath) {
    if (path.isAbsolute(relativePath)) {
      return relativePath;
    }
    return path.resolve(this.projectRoot, relativePath);
  }

  /**
   * Read a file
   * 
   * @param {string} filePath Path to the file
   * @returns {Promise<string>} File content
   */
  async readFile(filePath) {
    try {
      const fullPath = this.resolvePath(filePath);
      return await fs.readFile(fullPath, 'utf8');
    } catch (error) {
      logger.error(`Error reading file: ${filePath}`, { error });
      throw new Error(`Failed to read file ${filePath}: ${error.message}`);
    }
  }

  /**
   * Write content to a file
   * 
   * @param {string} filePath Path to the file
   * @param {string} content File content
   * @returns {Promise<void>}
   */
  async writeFile(filePath, content) {
    try {
      const fullPath = this.resolvePath(filePath);
      
      // Ensure the directory exists
      const dirPath = path.dirname(fullPath);
      await fs.mkdir(dirPath, { recursive: true });
      
      // Write the file
      await fs.writeFile(fullPath, content, 'utf8');
      
      logger.info(`File written: ${filePath}`);
    } catch (error) {
      logger.error(`Error writing file: ${filePath}`, { error });
      throw new Error(`Failed to write file ${filePath}: ${error.message}`);
    }
  }

  /**
   * Create a new file
   * 
   * @param {string} filePath Path to the file
   * @param {string} content Initial content
   * @returns {Promise<void>}
   */
  async createFile(filePath, content = '') {
    try {
      const fullPath = this.resolvePath(filePath);
      
      // Check if file already exists
      try {
        await fs.access(fullPath);
        throw new Error(`File already exists: ${filePath}`);
      } catch (error) {
        // File doesn't exist, which is what we want
        if (error.code !== 'ENOENT') {
          throw error;
        }
      }
      
      // Ensure the directory exists
      const dirPath = path.dirname(fullPath);
      await fs.mkdir(dirPath, { recursive: true });
      
      // Create the file
      await fs.writeFile(fullPath, content, 'utf8');
      
      logger.info(`File created: ${filePath}`);
    } catch (error) {
      logger.error(`Error creating file: ${filePath}`, { error });
      throw new Error(`Failed to create file ${filePath}: ${error.message}`);
    }
  }

  /**
   * Delete a file
   * 
   * @param {string} filePath Path to the file
   * @returns {Promise<void>}
   */
  async deleteFile(filePath) {
    try {
      const fullPath = this.resolvePath(filePath);
      await fs.unlink(fullPath);
      logger.info(`File deleted: ${filePath}`);
    } catch (error) {
      logger.error(`Error deleting file: ${filePath}`, { error });
      throw new Error(`Failed to delete file ${filePath}: ${error.message}`);
    }
  }

  /**
   * Check if a file exists
   * 
   * @param {string} filePath Path to the file
   * @returns {Promise<boolean>} Whether the file exists
   */
  async fileExists(filePath) {
    try {
      const fullPath = this.resolvePath(filePath);
      await fs.access(fullPath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * List files in a directory
   * 
   * @param {string} dirPath Directory path
   * @param {boolean} recursive Whether to search recursively
   * @returns {Promise<Array<string>>} List of file paths
   */
  async listFiles(dirPath, recursive = false) {
    try {
      const fullPath = this.resolvePath(dirPath);
      
      const entries = await fs.readdir(fullPath, { withFileTypes: true });
      
      let files = [];
      
      for (const entry of entries) {
        const entryPath = path.join(dirPath, entry.name);
        
        if (entry.isDirectory()) {
          if (recursive) {
            const subFiles = await this.listFiles(entryPath, true);
            files = files.concat(subFiles);
          }
        } else {
          files.push(entryPath);
        }
      }
      
      return files;
    } catch (error) {
      logger.error(`Error listing files in directory: ${dirPath}`, { error });
      throw new Error(`Failed to list files in ${dirPath}: ${error.message}`);
    }
  }

  /**
   * Create a directory
   * 
   * @param {string} dirPath Directory path
   * @returns {Promise<void>}
   */
  async createDirectory(dirPath) {
    try {
      const fullPath = this.resolvePath(dirPath);
      await fs.mkdir(fullPath, { recursive: true });
      logger.info(`Directory created: ${dirPath}`);
    } catch (error) {
      logger.error(`Error creating directory: ${dirPath}`, { error });
      throw new Error(`Failed to create directory ${dirPath}: ${error.message}`);
    }
  }

  /**
   * Get file information
   * 
   * @param {string} filePath Path to the file
   * @returns {Promise<Object>} File information
   */
  async getFileInfo(filePath) {
    try {
      const fullPath = this.resolvePath(filePath);
      const stats = await fs.stat(fullPath);
      
      return {
        path: filePath,
        size: stats.size,
        created: stats.birthtime,
        modified: stats.mtime,
        isDirectory: stats.isDirectory(),
        extension: path.extname(filePath),
        basename: path.basename(filePath)
      };
    } catch (error) {
      logger.error(`Error getting file info: ${filePath}`, { error });
      throw new Error(`Failed to get file info for ${filePath}: ${error.message}`);
    }
  }
}

// Create and export a singleton instance
const fileOperations = new FileOperationsService(process.cwd());

module.exports = fileOperations;