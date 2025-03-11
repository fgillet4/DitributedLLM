/**
 * Embeddings Module for RAG
 * 
 * This module handles the creation and management of embeddings
 * for code chunks, using the Jina Embeddings v3 model from Hugging Face.
 */

const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');
const { logger } = require('../../utils/logger');

// Configuration for embeddings
const EMBEDDING_DIMENSION = 768; // Jina Embeddings v3 dimensions

/**
 * Create an embedding for a text string using the Jina Embeddings model
 * 
 * @param {string} text Text to create embedding for
 * @returns {Promise<Array<number>>} Embedding vector
 */
async function createEmbedding(text) {
  try {
    // Call the Python script that uses Jina Embeddings
    const embedding = await callPythonEmbedding(text);
    return embedding;
  } catch (error) {
    logger.error(`Failed to create embedding: ${error.message}`, { error });
    
    // Return a zero vector as fallback
    return new Array(EMBEDDING_DIMENSION).fill(0);
  }
}

/**
 * Call Python script to generate embeddings using Jina Embeddings
 * 
 * @param {string} text Text to embed
 * @returns {Promise<Array<number>>} Embedding vector
 */
function callPythonEmbedding(text) {
  return new Promise((resolve, reject) => {
    // Create a temporary file with the text to embed
    const tempFilePath = path.join(__dirname, '../../../temp', `embed_${Date.now()}.txt`);
    const outputPath = path.join(__dirname, '../../../temp', `embed_result_${Date.now()}.json`);
    
    // Ensure temp directory exists
    fs.mkdir(path.join(__dirname, '../../../temp'), { recursive: true })
      .then(() => fs.writeFile(tempFilePath, text, 'utf8'))
      .then(() => {
        // Call the Python script
        const python = spawn('python', [
          path.join(__dirname, '../../utils/embeddings.py'),
          '--input', tempFilePath,
          '--output', outputPath,
          '--model', 'jinaai/jina-embeddings-v3'
        ]);
        
        let errorOutput = '';
        
        // Handle Python script output
        python.stderr.on('data', (data) => {
          errorOutput += data.toString();
        });
        
        python.on('close', async (code) => {
          // Clean up temp file
          fs.unlink(tempFilePath).catch(() => {});
          
          if (code !== 0) {
            // If Python script failed, return error
            reject(new Error(`Python embedding script failed with code ${code}: ${errorOutput}`));
            return;
          }
          
          try {
            // Read the output file
            const data = await fs.readFile(outputPath, 'utf8');
            const result = JSON.parse(data);
            
            // Clean up output file
            fs.unlink(outputPath).catch(() => {});
            
            // Return the embedding
            resolve(result.embedding);
          } catch (err) {
            reject(err);
          }
        });
      })
      .catch(reject);
  });
}

/**
 * Calculate the cosine similarity between two embedding vectors
 * 
 * @param {Array<number>} vecA First embedding vector
 * @param {Array<number>} vecB Second embedding vector
 * @returns {number} Cosine similarity (-1 to 1)
 */
function cosineSimilarity(vecA, vecB) {
  // Calculate dot product
  let dotProduct = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
  }
  
  // Calculate magnitudes
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < vecA.length; i++) {
    magA += vecA[i] * vecA[i];
    magB += vecB[i] * vecB[i];
  }
  magA = Math.sqrt(magA);
  magB = Math.sqrt(magB);
  
  // Calculate cosine similarity
  if (magA === 0 || magB === 0) {
    return 0;
  }
  return dotProduct / (magA * magB);
}

/**
 * Create a simple vector store for embeddings
 * 
 * @param {Object} options Store options
 * @param {string} options.storePath Path to store the vector database
 * @returns {Object} Vector store object
 */
function createVectorStore({ storePath }) {
  // In-memory store
  let items = [];
  
  // Store interface
  const store = {
    /**
     * Add an item to the vector store
     * 
     * @param {Object} item Item to add
     * @param {string} item.id Unique identifier
     * @param {string} item.content Content text
     * @param {Array<number>} item.embedding Embedding vector
     * @param {Object} item.metadata Additional metadata
     * @returns {Promise<void>}
     */
    async addItem(item) {
      // Remove existing item with the same ID if any
      items = items.filter(existing => existing.id !== item.id);
      
      // Add the new item
      items.push(item);
      
      // Save to disk (async)
      this.saveToFile().catch(error => {
        logger.error(`Failed to save vector store: ${error.message}`, { error });
      });
    },
    
    /**
     * Search for similar items
     * 
     * @param {Array<number>} embedding Query embedding
     * @param {Object} options Search options
     * @param {number} options.limit Maximum number of results
     * @param {number} options.minScore Minimum similarity score
     * @returns {Promise<Array<Object>>} Matching items
     */
    async search(embedding, options = {}) {
      const { limit = 5, minScore = 0.7 } = options;
      
      // Calculate similarity scores
      const results = items.map(item => ({
        ...item,
        score: cosineSimilarity(embedding, item.embedding)
      }));
      
      // Filter and sort by score
      return results
        .filter(item => item.score >= minScore)
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
    },
    
    /**
     * Save the vector store to disk
     * 
     * @returns {Promise<void>}
     */
    async saveToFile() {
      try {
        // Ensure directory exists
        const dir = path.dirname(storePath);
        await fs.mkdir(dir, { recursive: true });
        
        // Save to file
        await fs.writeFile(
          storePath,
          JSON.stringify(items, null, 2),
          'utf8'
        );
      } catch (error) {
        logger.error(`Failed to save vector store: ${error.message}`, { error });
        throw error;
      }
    },
    
    /**
     * Load the vector store from disk
     * 
     * @returns {Promise<void>}
     */
    async loadFromFile() {
      try {
        // Check if file exists
        const exists = await fs.access(storePath)
          .then(() => true)
          .catch(() => false);
        
        if (exists) {
          // Load from file
          const data = await fs.readFile(storePath, 'utf8');
          items = JSON.parse(data);
          logger.info(`Loaded ${items.length} items from vector store`);
        } else {
          logger.info('Vector store file not found, starting with empty store');
          items = [];
        }
      } catch (error) {
        logger.error(`Failed to load vector store: ${error.message}`, { error });
        items = [];
      }
    },
    
    /**
     * Get the number of items in the store
     * 
     * @returns {number} Item count
     */
    getCount() {
      return items.length;
    },
    
    /**
     * Clear the vector store
     * 
     * @returns {Promise<void>}
     */
    async clear() {
      items = [];
      await this.saveToFile();
    }
  };
  
  // Load existing data
  store.loadFromFile();
  
  return store;
}

module.exports = {
  createEmbedding,
  cosineSimilarity,
  createVectorStore
};