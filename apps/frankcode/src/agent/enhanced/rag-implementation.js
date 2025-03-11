/**
 * RAG (Retrieval-Augmented Generation) Module for FrankCode
 * 
 * This module adds code-aware retrieval capabilities to FrankCode,
 * allowing it to gain better context from the codebase for more
 * accurate and context-aware code generation and editing.
 */

const fs = require('fs').promises;
const path = require('path');
const { createEmbedding } = require('./embeddings');
const { logger } = require('../utils/logger');

/**
 * Create a Code-aware RAG system
 * 
 * @param {Object} options Configuration options
 * @param {string} options.projectRoot Project root directory
 * @param {Object} options.vectorStore Vector store for embeddings
 * @returns {Object} The RAG interface
 */
function createCodeRAG({ projectRoot, vectorStore }) {
  // Cache for file contents and embeddings
  const fileCache = new Map();
  
  /**
   * Index a file for retrieval
   * 
   * @param {string} filePath Path to the file
   * @returns {Promise<boolean>} Success indicator
   */
  async function indexFile(filePath) {
    try {
      // Get absolute path
      const fullPath = path.resolve(projectRoot, filePath);
      
      // Read file content
      const content = await fs.readFile(fullPath, 'utf8');
      
      // Split file into chunks (by function/class/meaningful blocks)
      const chunks = splitCodeIntoChunks(content, path.extname(filePath));
      
      // Generate embeddings and store them
      for (const chunk of chunks) {
        const embedding = await createEmbedding(chunk.content);
        await vectorStore.addItem({
          id: `${filePath}:${chunk.start}-${chunk.end}`,
          content: chunk.content, 
          embedding,
          metadata: {
            filePath,
            lineStart: chunk.start,
            lineEnd: chunk.end,
            type: chunk.type // function, class, etc.
          }
        });
      }
      
      // Update cache
      fileCache.set(filePath, {
        content,
        lastIndexed: Date.now()
      });
      
      logger.info(`Indexed file: ${filePath}`);
      return true;
    } catch (error) {
      logger.error(`Failed to index file: ${filePath}`, { error });
      return false;
    }
  }

  /**
   * Split code into meaningful chunks for better retrieval
   * 
   * @param {string} content File content
   * @param {string} fileExtension File extension
   * @returns {Array<Object>} Array of code chunks
   */
  function splitCodeIntoChunks(content, fileExtension) {
    const lines = content.split('\n');
    const chunks = [];
    let currentChunk = { content: '', start: 0, end: 0, type: 'unknown' };
    
    // Detect file type and use appropriate chunking strategy
    if (['.js', '.ts', '.jsx', '.tsx'].includes(fileExtension)) {
      // JavaScript/TypeScript chunking logic
      let inFunction = false;
      let inClass = false;
      let braceCount = 0;
      
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        
        // Detect function or class declaration
        if (!inFunction && !inClass) {
          if (line.match(/function\s+\w+\s*\(/) || line.match(/const\s+\w+\s*=\s*\(.*\)\s*=>/)) {
            inFunction = true;
            currentChunk = { content: line, start: i, end: i, type: 'function' };
            braceCount = countBraces(line);
          } else if (line.match(/class\s+\w+/)) {
            inClass = true;
            currentChunk = { content: line, start: i, end: i, type: 'class' };
            braceCount = countBraces(line);
          } else {
            // Add to a generic chunk if not in a specific construct
            if (chunks.length === 0 || chunks[chunks.length - 1].type !== 'generic') {
              currentChunk = { content: line, start: i, end: i, type: 'generic' };
              chunks.push(currentChunk);
            } else {
              chunks[chunks.length - 1].content += '\n' + line;
              chunks[chunks.length - 1].end = i;
            }
          }
        } else {
          // Continue building current function or class
          currentChunk.content += '\n' + line;
          currentChunk.end = i;
          
          braceCount += countBraces(line);
          
          // Check if we've completed a block
          if (braceCount === 0) {
            chunks.push(currentChunk);
            inFunction = false;
            inClass = false;
          }
        }
      }
    } else {
      // Default chunking strategy for other file types
      let currentBlock = [];
      let blockStart = 0;
      
      for (let i = 0; i < lines.length; i++) {
        currentBlock.push(lines[i]);
        
        // Split by reasonable chunk size or empty lines as a heuristic
        if (currentBlock.length > 50 || (lines[i].trim() === '' && currentBlock.length > 10)) {
          chunks.push({
            content: currentBlock.join('\n'),
            start: blockStart,
            end: i,
            type: 'block'
          });
          
          currentBlock = [];
          blockStart = i + 1;
        }
      }
      
      // Add the last block if not empty
      if (currentBlock.length > 0) {
        chunks.push({
          content: currentBlock.join('\n'),
          start: blockStart,
          end: lines.length - 1,
          type: 'block'
        });
      }
    }
    
    return chunks;
  }
  
  /**
   * Count opening and closing braces in a line
   * 
   * @param {string} line Line of code
   * @returns {number} Net brace count
   */
  function countBraces(line) {
    let count = 0;
    for (const char of line) {
      if (char === '{') count++;
      else if (char === '}') count--;
    }
    return count;
  }
  
  /**
   * Index multiple files
   * 
   * @param {Array<string>} filePaths Array of file paths
   * @returns {Promise<Array<string>>} Successfully indexed files
   */
  async function indexFiles(filePaths) {
    const results = [];
    
    for (const filePath of filePaths) {
      const success = await indexFile(filePath);
      if (success) {
        results.push(filePath);
      }
    }
    
    return results;
  }
  
  /**
   * Retrieve relevant code context based on a query
   * 
   * @param {string} query Query string
   * @param {Object} options Query options
   * @param {number} options.maxResults Maximum number of results
   * @param {Array<string>} options.fileTypes Filter by file types
   * @param {boolean} options.includeDocs Include documentation blocks
   * @returns {Promise<Array<Object>>} Relevant code contexts
   */
  async function retrieveContext(query, options = {}) {
    try {
      const { 
        maxResults = 5, 
        fileTypes = null,
        includeDocs = true
      } = options;
      
      // Generate embedding for the query
      const queryEmbedding = await createEmbedding(query);
      
      // Search for similar code chunks
      let results = await vectorStore.search(queryEmbedding, { limit: maxResults * 2 });
      
      // Apply filters
      if (fileTypes) {
        results = results.filter(item => {
          const ext = path.extname(item.metadata.filePath);
          return fileTypes.includes(ext);
        });
      }
      
      // Filter out docs if not requested
      if (!includeDocs) {
        results = results.filter(item => 
          !item.metadata.type.includes('comment') && 
          !item.metadata.type.includes('documentation')
        );
      }
      
      // Limit results
      results = results.slice(0, maxResults);
      
      // Format results for use
      return results.map(item => ({
        content: item.content,
        filePath: item.metadata.filePath,
        lineStart: item.metadata.lineStart,
        lineEnd: item.metadata.lineEnd,
        type: item.metadata.type,
        score: item.score
      }));
    } catch (error) {
      logger.error(`Failed to retrieve context: ${error.message}`, { error });
      return [];
    }
  }
  
  /**
   * Scan the project directory and index all relevant files
   * 
   * @param {Object} options Scanning options
   * @param {Array<string>} options.include Glob patterns to include
   * @param {Array<string>} options.exclude Glob patterns to exclude
   * @returns {Promise<number>} Number of indexed files
   */
  async function indexProject(options = {}) {
    try {
      const { 
        include = ['**/*.js', '**/*.ts', '**/*.jsx', '**/*.tsx', '**/*.py', '**/*.java', '**/*.c', '**/*.cpp'],
        exclude = ['**/node_modules/**', '**/dist/**', '**/build/**', '**/.git/**']
      } = options;
      
      // Scan the project for files
      const { listDirectoryFiles } = require('./fileManager');
      const files = await listDirectoryFiles(projectRoot, { include, exclude });
      
      // Index the files
      const indexedFiles = await indexFiles(files);
      
      logger.info(`Indexed ${indexedFiles.length} files in project`);
      return indexedFiles.length;
    } catch (error) {
      logger.error(`Failed to index project: ${error.message}`, { error });
      return 0;
    }
  }
  
  /**
   * Format retrieved code context for use in prompts
   * 
   * @param {Array<Object>} contexts Retrieved contexts
   * @returns {string} Formatted context for prompts
   */
  function formatContextForPrompt(contexts) {
    let formattedContext = '--- RELEVANT CODE CONTEXT ---\n\n';
    
    for (const ctx of contexts) {
      formattedContext += `File: ${ctx.filePath} (lines ${ctx.lineStart}-${ctx.lineEnd})\n`;
      formattedContext += `Type: ${ctx.type}\n`;
      formattedContext += '```\n';
      formattedContext += ctx.content;
      formattedContext += '\n```\n\n';
    }
    
    return formattedContext;
  }
  
  // Return the RAG interface
  return {
    indexFile,
    indexFiles,
    indexProject,
    retrieveContext,
    formatContextForPrompt
  };
}

module.exports = {
  createCodeRAG
};