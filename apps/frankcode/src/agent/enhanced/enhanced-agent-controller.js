/**
 * Enhanced Agent Controller
 *
 * This module provides a controller for the enhanced agent capabilities
 * including RAG, GraphRAG and memory functionality.
 */

const path = require('path');
const fs = require('fs').promises;
const { logger } = require('../../utils/logger');
const { createEmbedding, createVectorStore } = require('./embeddings');
const { createRagImplementation } = require('./rag-implementation');
const { createGraphRagModule } = require('./graphrag-module');
const { createMemoryController } = require('./memory-controller');
const { createEnhancedAgent } = require('./enhanced-agent');

/**
 * Create an enhanced agent controller
 * 
 * @param {Object} options Controller options
 * @param {Object} options.agent The main agent
 * @param {Object} options.apiClient The API client
 * @param {Object} options.tokenMonitor The token monitor
 * @param {Object} options.screen The screen object
 * @param {Object} options.outputRenderer The output renderer
 * @param {string} options.projectRoot The project root path
 * @returns {Object} The enhanced agent controller
 */
function createEnhancedAgentController({ agent, apiClient, tokenMonitor, screen, outputRenderer, projectRoot }) {
  // Initialize components
  const storePath = path.join(projectRoot, '.frankcode', 'vector-store.json');
  const graphPath = path.join(projectRoot, '.frankcode', 'code-graph.json');
  const memoryPath = path.join(projectRoot, '.frankcode', 'memory-store.json');
  
  // Ensure .frankcode directory exists
  fs.mkdir(path.join(projectRoot, '.frankcode'), { recursive: true })
    .catch(err => logger.error(`Failed to create .frankcode directory: ${err.message}`));
  
  // Create vector store
  const vectorStore = createVectorStore({ storePath });
  
  // Create RAG implementation
  const rag = createRagImplementation({ 
    vectorStore, 
    createEmbedding,
    projectRoot
  });
  
  // Create GraphRAG module
  const graphRag = createGraphRagModule({
    storePath: graphPath,
    projectRoot
  });
  
  // Create memory controller
  const memory = createMemoryController({
    storePath: memoryPath
  });
  
  // Create enhanced agent
  const enhancedAgent = createEnhancedAgent({
    agent,
    rag,
    graphRag,
    memory,
    apiClient
  });

  // Command handlers
  /**
   * Process a command from the user
   * 
   * @param {string} message The user message
   * @returns {Promise<Object>} The response
   */
  async function processMessage(message) {
    // Check if it's a command
    if (message.startsWith('/')) {
      const parts = message.slice(1).split(' ');
      const command = parts[0];
      const args = parts.slice(1);
      
      switch (command) {
        case 'rag':
          return await handleRagCommand(args);
        case 'graph':
          return await handleGraphCommand(args);
        case 'memory':
          return await handleMemoryCommand(args);
        case 'enhance':
          return await handleEnhanceCommand(args);
        default:
          return { handled: false };
      }
    }
    
    // If not a command, enhance the request and let normal flow continue
    try {
      await enhancedAgent.enhanceRequestWithContext(message);
      return { handled: false };
    } catch (error) {
      logger.error(`Enhanced agent error: ${error.message}`, { error });
      return { handled: false };
    }
  }
  
  /**
   * Handle RAG commands
   * 
   * @param {Array<string>} args Command arguments
   * @returns {Promise<Object>} Response
   */
  async function handleRagCommand(args) {
    const subCommand = args[0];
    
    try {
      switch (subCommand) {
        case 'index':
          outputRenderer.addSystemMessage('Indexing project files for RAG...');
          const indexStats = await rag.indexProject();
          outputRenderer.addSystemMessage(`✅ RAG indexing complete! Indexed ${indexStats.fileCount} files and ${indexStats.chunkCount} code chunks.`);
          return { handled: true };
          
        case 'search':
          const query = args.slice(1).join(' ');
          if (!query) {
            outputRenderer.addSystemMessage('❌ Please provide a search query.');
            return { handled: true };
          }
          
          outputRenderer.addSystemMessage(`Searching for code related to "${query}"...`);
          const results = await rag.search(query, { limit: 5 });
          
          if (results.length === 0) {
            outputRenderer.addSystemMessage('No matching code found in the project.');
          } else {
            outputRenderer.addSystemMessage(`Found ${results.length} relevant code snippets:`);
            
            for (const result of results) {
              outputRenderer.addSystemMessage(`\n--- ${result.metadata.filePath} (Score: ${result.score.toFixed(2)}) ---\n${result.content}\n`);
            }
          }
          return { handled: true };
          
        case 'status':
          const status = await rag.getStatus();
          outputRenderer.addSystemMessage(`RAG Status:\nIndexed files: ${status.fileCount}\nCode chunks: ${status.chunkCount}\nLast updated: ${status.lastUpdated || 'Never'}`);
          return { handled: true };
          
        case 'clear':
          await rag.clearIndex();
          outputRenderer.addSystemMessage('✅ RAG index cleared.');
          return { handled: true };
          
        default:
          outputRenderer.addSystemMessage('❌ Unknown RAG command. Available commands: index, search, status, clear');
          return { handled: true };
      }
    } catch (error) {
      logger.error(`RAG command error: ${error.message}`, { error });
      outputRenderer.addSystemMessage(`❌ Error executing RAG command: ${error.message}`);
      return { handled: true };
    }
  }
  
  /**
   * Handle GraphRAG commands
   * 
   * @param {Array<string>} args Command arguments
   * @returns {Promise<Object>} Response
   */
  async function handleGraphCommand(args) {
    const subCommand = args[0];
    
    try {
      switch (subCommand) {
        case 'index':
          outputRenderer.addSystemMessage('Building code relationship graph...');
          const indexStats = await graphRag.buildGraph();
          outputRenderer.addSystemMessage(`✅ Graph building complete! ${indexStats.nodeCount} nodes and ${indexStats.edgeCount} relationships indexed.`);
          return { handled: true };
          
        case 'search':
          const query = args.slice(1).join(' ');
          if (!query) {
            outputRenderer.addSystemMessage('❌ Please provide a search query.');
            return { handled: true };
          }
          
          outputRenderer.addSystemMessage(`Searching for code entities related to "${query}"...`);
          const results = await graphRag.search(query, { limit: 5 });
          
          if (results.length === 0) {
            outputRenderer.addSystemMessage('No matching code entities found in the project.');
          } else {
            outputRenderer.addSystemMessage(`Found ${results.length} relevant code entities:`);
            
            for (const result of results) {
              const relatedEntities = result.related.map(r => r.name).join(', ');
              outputRenderer.addSystemMessage(`\n--- ${result.name} (${result.type}) in ${result.filePath} ---\nRelated to: ${relatedEntities || 'None'}\n`);
            }
          }
          return { handled: true };
          
        case 'status':
          const status = await graphRag.getStatus();
          outputRenderer.addSystemMessage(`Graph Status:\nNodes: ${status.nodeCount}\nRelationships: ${status.edgeCount}\nLast updated: ${status.lastUpdated || 'Never'}`);
          return { handled: true };
          
        case 'clear':
          await graphRag.clearGraph();
          outputRenderer.addSystemMessage('✅ Code graph cleared.');
          return { handled: true };
          
        default:
          outputRenderer.addSystemMessage('❌ Unknown graph command. Available commands: index, search, status, clear');
          return { handled: true };
      }
    } catch (error) {
      logger.error(`Graph command error: ${error.message}`, { error });
      outputRenderer.addSystemMessage(`❌ Error executing graph command: ${error.message}`);
      return { handled: true };
    }
  }
  
  /**
   * Handle memory commands
   * 
   * @param {Array<string>} args Command arguments
   * @returns {Promise<Object>} Response
   */
  async function handleMemoryCommand(args) {
    const subCommand = args[0];
    
    try {
      switch (subCommand) {
        case 'learn':
          const context = args.slice(1).join(' ');
          if (!context) {
            outputRenderer.addSystemMessage('❌ Please provide content to learn.');
            return { handled: true };
          }
          
          await memory.storeMemory(context);
          outputRenderer.addSystemMessage('✅ I\'ve learned this information and will remember it for future interactions.');
          return { handled: true };
          
        case 'recall':
          const query = args.slice(1).join(' ');
          if (!query) {
            outputRenderer.addSystemMessage('❌ Please provide a query to recall memories.');
            return { handled: true };
          }
          
          const memories = await memory.recallMemories(query, { limit: 5 });
          
          if (memories.length === 0) {
            outputRenderer.addSystemMessage('I don\'t have any memories related to that query.');
          } else {
            outputRenderer.addSystemMessage(`Here's what I remember about "${query}":`);
            
            for (const mem of memories) {
              outputRenderer.addSystemMessage(`\n- ${mem.content} (Relevance: ${mem.score.toFixed(2)})`);
            }
          }
          return { handled: true };
          
        case 'list':
          const allMemories = await memory.getAllMemories();
          
          if (allMemories.length === 0) {
            outputRenderer.addSystemMessage('I don\'t have any stored memories yet.');
          } else {
            outputRenderer.addSystemMessage(`I have ${allMemories.length} memories stored:`);
            
            for (const mem of allMemories) {
              outputRenderer.addSystemMessage(`\n- ${mem.content}`);
            }
          }
          return { handled: true };
          
        case 'clear':
          await memory.clearMemories();
          outputRenderer.addSystemMessage('✅ All memories have been cleared.');
          return { handled: true };
          
        default:
          outputRenderer.addSystemMessage('❌ Unknown memory command. Available commands: learn, recall, list, clear');
          return { handled: true };
      }
    } catch (error) {
      logger.error(`Memory command error: ${error.message}`, { error });
      outputRenderer.addSystemMessage(`❌ Error executing memory command: ${error.message}`);
      return { handled: true };
    }
  }
  
  /**
   * Handle enhance commands
   * 
   * @param {Array<string>} args Command arguments
   * @returns {Promise<Object>} Response
   */
  async function handleEnhanceCommand(args) {
    const subCommand = args[0] || 'status';
    
    try {
      switch (subCommand) {
        case 'status':
          const ragStatus = await rag.getStatus();
          const graphStatus = await graphRag.getStatus();
          const memoryStatus = await memory.getStatus();
          
          outputRenderer.addSystemMessage(`Enhanced Agent Status:

RAG System:
- Indexed files: ${ragStatus.fileCount}
- Code chunks: ${ragStatus.chunkCount}
- Last updated: ${ragStatus.lastUpdated || 'Never'}

GraphRAG System:
- Nodes: ${graphStatus.nodeCount}
- Relationships: ${graphStatus.edgeCount}
- Last updated: ${graphStatus.lastUpdated || 'Never'}

Memory System:
- Stored memories: ${memoryStatus.memoryCount}
- Last updated: ${memoryStatus.lastUpdated || 'Never'}

Use the following commands to work with enhanced features:
- /rag index - Index your codebase for RAG
- /rag search <query> - Search for relevant code
- /graph index - Build a code relationship graph
- /graph search <query> - Find related code entities
- /memory learn <context> - Teach me something to remember
- /memory recall <query> - Recall related memories`);
          return { handled: true };
          
        case 'prompt':
          const task = args.slice(1).join(' ');
          if (!task) {
            outputRenderer.addSystemMessage('❌ Please provide a task description.');
            return { handled: true };
          }
          
          const enhancedPrompt = await enhancedAgent.generateEnhancedPrompt(task);
          outputRenderer.addSystemMessage(`Enhanced prompt for "${task}":\n\n${enhancedPrompt}`);
          return { handled: true };
          
        default:
          outputRenderer.addSystemMessage('❌ Unknown enhance command. Available commands: status, prompt');
          return { handled: true };
      }
    } catch (error) {
      logger.error(`Enhance command error: ${error.message}`, { error });
      outputRenderer.addSystemMessage(`❌ Error executing enhance command: ${error.message}`);
      return { handled: true };
    }
  }
  
  /**
   * Learn from a file modification
   * 
   * @param {Object} modification Modification details
   * @param {string} modification.filePath File path
   * @param {string} modification.content New content
   * @param {string} modification.originalContent Original content
   * @param {boolean} approved Whether the modification was approved
   * @returns {Promise<void>}
   */
  async function learnFromModification(modification, approved) {
    try {
      if (approved) {
        // Store information about the modification
        const { filePath, content, originalContent } = modification;
        
        // Create a memory of the modification
        const memoryContent = `File ${filePath} was ${originalContent ? 'modified' : 'created'}. The approved content was stored.`;
        await memory.storeMemory(memoryContent, { 
          type: 'file_modification',
          filePath,
          timestamp: new Date().toISOString()
        });
        
        // Re-index the file in RAG if RAG is enabled and has been indexed before
        const ragStatus = await rag.getStatus();
        if (ragStatus.fileCount > 0) {
          await rag.indexFile(filePath, content);
        }
        
        // Update graph if graph has been built before
        const graphStatus = await graphRag.getStatus();
        if (graphStatus.nodeCount > 0) {
          await graphRag.updateFileInGraph(filePath, content);
        }
        
        logger.info(`Enhanced agent learned from modification to ${filePath}`);
      }
    } catch (error) {
      logger.error(`Failed to learn from modification: ${error.message}`, { error });
    }
  }
  
  // Return the controller interface
  return {
    processMessage,
    handleRagCommand,
    handleGraphCommand,
    handleMemoryCommand,
    handleEnhanceCommand,
    learnFromModification
  };
}

module.exports = {
  createEnhancedAgentController
};