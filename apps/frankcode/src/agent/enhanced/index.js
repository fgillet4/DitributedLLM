/**
 * Enhanced Agent Module Index
 *
 * This file exports all enhanced agent modules for use in other parts of the application.
 */

const { createEmbedding, createVectorStore } = require('./embeddings');
const { createRagImplementation } = require('./rag-implementation');
const { createGraphRagModule } = require('./graphrag-module');
const { createMemoryController } = require('./memory-controller');
const { createEnhancedAgent } = require('./enhanced-agent');
const { createEnhancedAgentController } = require('../enhanced-agent-controller');

module.exports = {
  createEmbedding,
  createVectorStore,
  createRagImplementation,
  createGraphRagModule,
  createMemoryController,
  createEnhancedAgent,
  createEnhancedAgentController
};