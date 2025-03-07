/**
 * File System utilities
 * 
 * Provides file system operations for the application
 */

const fs = require('fs').promises;
const path = require('path');
const fastGlob = require('fast-glob');
const ignore = require('ignore');
const { logger } = require('./logger');

/**
 * Scan a project directory for files
 * 
 * @param {string} projectRoot Project root directory
 * @param {Object} options Scan options
 * @param {Array<string>} options.exclude Patterns to exclude
 * @param {Array<string>} options.prioritize Patterns to prioritize
 * @returns {Promise<Array<string>>} List of file paths
 */
async function scanProjectFiles(projectRoot, options = {}) {
  try {
    const { exclude = [], prioritize = [] } = options;
    
    // Create ignore instance
    const ig = ignore().add(exclude);
    
    // Check for .gitignore
    try {
      const gitignorePath = path.join(projectRoot, '.gitignore');
      const gitignoreContent = await fs.readFile(gitignorePath, 'utf8');
      ig.add(gitignoreContent);
    } catch (error) {
      // .gitignore not found, continue
    }
    
    // Find all files
    const allFiles = await fastGlob('**/*', {
      cwd: projectRoot,
      onlyFiles: true,
      dot: true
    });
    
    // Filter files using ignore rules
    const filteredFiles = ig.filter(allFiles);
    
    // Sort prioritized files first
    const priorityPatterns = prioritize.map(pattern => new RegExp(pattern.replace('*', '.*')));
    
    filteredFiles.sort((a, b) => {
      const aIsPriority = priorityPatterns.some(pattern => pattern.test(a));
      const bIsPriority = priorityPatterns.some(pattern => pattern.test(b));
      
      if (aIsPriority && !bIsPriority) return -1;
      if (!aIsPriority && bIsPriority) return 1;
      return a.localeCompare(b);
    });
    
    // Convert to absolute paths
    return filteredFiles.map(file => path.join(projectRoot, file));
  } catch (error) {
    logger.error('Failed to scan project files', { error });
    return [];
  }
}

/**
 * Ensure a directory exists
 * 
 * @param {string} dirPath Directory path
 * @returns {Promise<void>}
 */
async function ensureDir(dirPath) {
  try {
    await fs.mkdir(dirPath, { recursive: true });
  } catch (error) {
    logger.error(`Failed to create directory: ${dirPath}`, { error });
    throw error;
  }
}

/**
 * Check if a path exists
 * 
 * @param {string} filePath File path
 * @returns {Promise<boolean>} Whether the path exists
 */
async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

/**
 * Get file stats
 * 
 * @param {string} filePath File path
 * @returns {Promise<Object>} File stats
 */
async function getStats(filePath) {
  try {
    const stats = await fs.stat(filePath);
    return {
      path: filePath,
      size: stats.size,
      isDirectory: stats.isDirectory(),
      isFile: stats.isFile(),
      created: stats.birthtime,
      modified: stats.mtime,
      accessed: stats.atime
    };
  } catch (error) {
    logger.error(`Failed to get stats for: ${filePath}`, { error });
    throw error;
  }
}

/**
 * Determine the file type based on extension
 * 
 * @param {string} filePath File path
 * @returns {string} File type
 */
function getFileType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  
  // Define file type mappings
  const typeMap = {
    // Code files
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.py': 'python',
    '.rb': 'ruby',
    '.php': 'php',
    '.java': 'java',
    '.go': 'go',
    '.cs': 'csharp',
    '.c': 'c',
    '.cpp': 'cpp',
    '.rs': 'rust',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
    
    // Web files
    '.html': 'html',
    '.htm': 'html',
    '.css': 'css',
    '.scss': 'sass',
    '.sass': 'sass',
    '.less': 'less',
    '.json': 'json',
    '.xml': 'xml',
    '.svg': 'svg',
    
    // Config files
    '.yml': 'yaml',
    '.yaml': 'yaml',
    '.toml': 'toml',
    '.ini': 'ini',
    '.env': 'env',
    
    // Documentation
    '.md': 'markdown',
    '.markdown': 'markdown',
    '.txt': 'text',
    '.rtf': 'rtf',
    '.pdf': 'pdf',
    '.doc': 'word',
    '.docx': 'word',
    '.xls': 'excel',
    '.xlsx': 'excel',
    '.ppt': 'powerpoint',
    '.pptx': 'powerpoint',
    
    // Others
    '.csv': 'csv',
    '.tsv': 'tsv',
    '.sql': 'sql',
    '.sh': 'shell',
    '.bash': 'shell',
    '.zsh': 'shell',
    '.fish': 'shell',
    '.bat': 'batch',
    '.ps1': 'powershell'
  };
  
  return typeMap[ext] || 'unknown';
}

/**
 * Create a temporary file
 * 
 * @param {string} content File content
 * @param {string} extension File extension
 * @returns {Promise<string>} Path to the temporary file
 */
async function createTempFile(content, extension = '.txt') {
  try {
    // Create temp directory if it doesn't exist
    const tempDir = path.join(process.cwd(), 'temp');
    await ensureDir(tempDir);
    
    // Generate random filename
    const randomName = Math.random().toString(36).substring(2, 15);
    const tempFilePath = path.join(tempDir, `${randomName}${extension}`);
    
    // Write content to file
    await fs.writeFile(tempFilePath, content, 'utf8');
    
    return tempFilePath;
  } catch (error) {
    logger.error('Failed to create temporary file', { error });
    throw error;
  }
}

module.exports = {
  scanProjectFiles,
  ensureDir,
  pathExists,
  getStats,
  getFileType,
  createTempFile
};