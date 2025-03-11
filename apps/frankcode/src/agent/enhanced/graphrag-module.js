/**
 * GraphRAG Module
 *
 * This module implements a graph-based code understanding system
 * to track relationships between code entities.
 */

const path = require('path');
const fs = require('fs').promises;
const glob = require('fast-glob');
const { logger } = require('../../utils/logger');
const { createEmbedding } = require('./embeddings');

// Configuration
const EXCLUDE_PATTERNS = [
  'node_modules/**',
  '.git/**',
  'dist/**',
  'build/**',
  '**/*.min.js',
  '**/*.bundle.js',
  '**/*.log',
  '**/package-lock.json'
];

/**
 * Create a GraphRAG module
 * 
 * @param {Object} options GraphRAG options
 * @param {string} options.storePath Path to store the graph
 * @param {string} options.projectRoot The project root path
 * @returns {Object} The GraphRAG module
 */
function createGraphRagModule({ storePath, projectRoot }) {
  // Graph state
  let graph = {
    nodes: [],
    edges: []
  };
  
  // Metadata
  let lastUpdated = null;
  
  /**
   * Build a graph of code relationships
   * 
   * @param {Object} options Build options
   * @param {Array<string>} options.include Glob patterns to include
   * @param {Array<string>} options.exclude Glob patterns to exclude
   * @returns {Promise<Object>} Build statistics
   */
  async function buildGraph(options = {}) {
    try {
      const { include = ['**/*.js', '**/*.jsx', '**/*.ts', '**/*.tsx'], 
              exclude = EXCLUDE_PATTERNS } = options;
      
      // Reset graph
      graph = {
        nodes: [],
        edges: []
      };
      
      // Find all code files
      const files = await glob(include, {
        cwd: projectRoot,
        ignore: exclude,
        absolute: true
      });
      
      // Process each file
      for (const filePath of files) {
        try {
          const relativePath = path.relative(projectRoot, filePath);
          const content = await fs.readFile(filePath, 'utf8');
          
          // Skip empty files
          if (!content.trim()) {
            continue;
          }
          
          // Parse and add to graph
          await parseFileAndAddToGraph(relativePath, content);
        } catch (error) {
          logger.error(`Failed to process file ${filePath} for graph: ${error.message}`, { error });
        }
      }
      
      // Find relationships between nodes
      findRelationshipsBetweenNodes();
      
      // Update metadata
      lastUpdated = new Date().toISOString();
      
      // Save graph
      await saveGraph();
      
      return {
        nodeCount: graph.nodes.length,
        edgeCount: graph.edges.length
      };
    } catch (error) {
      logger.error(`Failed to build graph: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * Parse a file and add entities to the graph
   * 
   * @param {string} filePath Relative file path
   * @param {string} content File content
   * @returns {Promise<void>}
   */
  async function parseFileAndAddToGraph(filePath, content) {
    // Extract entities (simple regex-based approach)
    const functions = extractFunctions(content);
    const classes = extractClasses(content);
    const exports = extractExports(content, filePath);
    const imports = extractImports(content, filePath);
    
    // Add file node
    const fileNode = {
      id: `file:${filePath}`,
      name: filePath,
      type: 'file',
      filePath,
      embedding: await createEmbedding(`File: ${filePath}`)
    };
    
    addNode(fileNode);
    
    // Add function nodes and connect to file
    for (const func of functions) {
      const funcNode = {
        id: `function:${filePath}:${func.name}`,
        name: func.name,
        type: 'function',
        filePath,
        embedding: await createEmbedding(`Function: ${func.name} in ${filePath}`)
      };
      
      addNode(funcNode);
      addEdge(fileNode.id, funcNode.id, 'contains');
    }
    
    // Add class nodes and connect to file
    for (const cls of classes) {
      const classNode = {
        id: `class:${filePath}:${cls.name}`,
        name: cls.name,
        type: 'class',
        filePath,
        embedding: await createEmbedding(`Class: ${cls.name} in ${filePath}`)
      };
      
      addNode(classNode);
      addEdge(fileNode.id, classNode.id, 'contains');
      
      // Add methods as function nodes and connect to class
      for (const method of cls.methods) {
        const methodNode = {
          id: `method:${filePath}:${cls.name}.${method}`,
          name: `${cls.name}.${method}`,
          type: 'method',
          filePath,
          embedding: await createEmbedding(`Method: ${method} in class ${cls.name} in ${filePath}`)
        };
        
        addNode(methodNode);
        addEdge(classNode.id, methodNode.id, 'contains');
      }
    }
    
    // Add export edges
    for (const exp of exports) {
      // Find the node being exported
      const sourceNodeId = findNodeId(exp.source, filePath);
      if (sourceNodeId) {
        // Add export node
        const exportNode = {
          id: `export:${filePath}:${exp.name}`,
          name: exp.name,
          type: 'export',
          filePath,
          embedding: await createEmbedding(`Export: ${exp.name} from ${filePath}`)
        };
        
        addNode(exportNode);
        addEdge(sourceNodeId, exportNode.id, 'exports');
        addEdge(fileNode.id, exportNode.id, 'contains');
      }
    }
    
    // Add import edges
    for (const imp of imports) {
      // Add import node
      const importNode = {
        id: `import:${filePath}:${imp.name}`,
        name: imp.name,
        type: 'import',
        filePath,
        targetPath: imp.source,
        embedding: await createEmbedding(`Import: ${imp.name} from ${imp.source} in ${filePath}`)
      };
      
      addNode(importNode);
      addEdge(fileNode.id, importNode.id, 'contains');
    }
  }
  
  /**
   * Find relationships between nodes
   */
  function findRelationshipsBetweenNodes() {
    // Connect import nodes to their target exports
    const importNodes = graph.nodes.filter(node => node.type === 'import');
    const exportNodes = graph.nodes.filter(node => node.type === 'export');
    
    for (const importNode of importNodes) {
      // Get target path
      const targetPath = importNode.targetPath;
      if (!targetPath) continue;
      
      // Find matching export nodes
      const matchingExports = exportNodes.filter(node => {
        // Handle relative imports
        if (targetPath.startsWith('./') || targetPath.startsWith('../')) {
          const importFilePath = importNode.filePath;
          const importDir = path.dirname(importFilePath);
          const resolvedPath = path.normalize(path.join(importDir, targetPath));
          
          // Match with or without extension
          return node.filePath === resolvedPath || 
                 node.filePath === `${resolvedPath}.js` || 
                 node.filePath === `${resolvedPath}.jsx` || 
                 node.filePath === `${resolvedPath}.ts` || 
                 node.filePath === `${resolvedPath}.tsx`;
        }
        
        // Handle package imports
        return false; // For now, we don't handle package imports
      });
      
      // Connect import to exports
      for (const exportNode of matchingExports) {
        addEdge(importNode.id, exportNode.id, 'imports');
      }
    }
  }
  
  /**
   * Update a file in the graph
   * 
   * @param {string} filePath Relative file path
   * @param {string} content New file content
   * @returns {Promise<void>}
   */
  async function updateFileInGraph(filePath, content) {
    try {
      // Remove existing file and related nodes
      removeFileFromGraph(filePath);
      
      // Parse and add to graph
      await parseFileAndAddToGraph(filePath, content);
      
      // Find relationships
      findRelationshipsBetweenNodes();
      
      // Update metadata
      lastUpdated = new Date().toISOString();
      
      // Save graph
      await saveGraph();
    } catch (error) {
      logger.error(`Failed to update file in graph: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * Remove a file and related nodes from the graph
   * 
   * @param {string} filePath Relative file path
   */
  function removeFileFromGraph(filePath) {
    // Find all nodes for this file
    const fileNodes = graph.nodes.filter(node => node.filePath === filePath);
    const nodeIds = fileNodes.map(node => node.id);
    
    // Remove nodes
    graph.nodes = graph.nodes.filter(node => node.filePath !== filePath);
    
    // Remove edges
    graph.edges = graph.edges.filter(edge => 
      !nodeIds.includes(edge.source) && !nodeIds.includes(edge.target)
    );
  }
  
  /**
   * Add a node to the graph
   * 
   * @param {Object} node The node to add
   */
  function addNode(node) {
    // Check if node already exists
    const index = graph.nodes.findIndex(n => n.id === node.id);
    
    if (index >= 0) {
      // Update existing node
      graph.nodes[index] = { ...graph.nodes[index], ...node };
    } else {
      // Add new node
      graph.nodes.push(node);
    }
  }
  
  /**
   * Add an edge to the graph
   * 
   * @param {string} sourceId Source node ID
   * @param {string} targetId Target node ID
   * @param {string} type Edge type
   */
  function addEdge(sourceId, targetId, type) {
    const edgeId = `${sourceId}-${type}->${targetId}`;
    
    // Check if edge already exists
    const exists = graph.edges.some(e => e.id === edgeId);
    
    if (!exists) {
      // Add new edge
      graph.edges.push({
        id: edgeId,
        source: sourceId,
        target: targetId,
        type
      });
    }
  }
  
  /**
   * Find a node ID by name and file path
   * 
   * @param {string} name Entity name
   * @param {string} filePath File path
   * @returns {string|null} Node ID
   */
  function findNodeId(name, filePath) {
    // Try to find function, class, or method node
    const node = graph.nodes.find(n => 
      n.filePath === filePath && 
      (n.name === name || n.name.endsWith(`.${name}`))
    );
    
    return node ? node.id : null;
  }
  
  /**
   * Extract functions from code content
   * 
   * @param {string} content Code content
   * @returns {Array<Object>} Extracted functions
   */
  function extractFunctions(content) {
    const functions = [];
    
    // Regular function declarations
    const funcRegex = /function\s+(\w+)\s*\(/g;
    let match;
    while ((match = funcRegex.exec(content)) !== null) {
      functions.push({ name: match[1] });
    }
    
    // Arrow functions and function expressions
    const arrowFuncRegex = /const\s+(\w+)\s*=\s*(\([^)]*\)|[^=]*)\s*=>/g;
    while ((match = arrowFuncRegex.exec(content)) !== null) {
      functions.push({ name: match[1] });
    }
    
    return functions;
  }
  
  /**
   * Extract classes from code content
   * 
   * @param {string} content Code content
   * @returns {Array<Object>} Extracted classes
   */
  function extractClasses(content) {
    const classes = [];
    
    // Class declarations
    const classRegex = /class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{/g;
    let match;
    while ((match = classRegex.exec(content)) !== null) {
      // Find methods in class
      const classStart = match.index;
      let braceCount = 0;
      let classEnd = classStart;
      
      // Find the class end by counting braces
      for (let i = match.index; i < content.length; i++) {
        if (content[i] === '{') braceCount++;
        if (content[i] === '}') {
          braceCount--;
          if (braceCount === 0) {
            classEnd = i;
            break;
          }
        }
      }
      
      // Extract the class body
      const classBody = content.substring(classStart, classEnd + 1);
      
      // Find methods
      const methodRegex = /(?:async\s+)?(\w+)\s*\([^)]*\)\s*{/g;
      const methods = [];
      let methodMatch;
      while ((methodMatch = methodRegex.exec(classBody)) !== null) {
        // Skip constructor
        if (methodMatch[1] !== 'constructor') {
          methods.push(methodMatch[1]);
        }
      }
      
      classes.push({
        name: match[1],
        extends: match[2] || null,
        methods
      });
    }
    
    return classes;
  }
  
  /**
   * Extract exports from code content
   * 
   * @param {string} content Code content
   * @param {string} filePath File path
   * @returns {Array<Object>} Extracted exports
   */
  function extractExports(content, filePath) {
    const exports = [];
    
    // module.exports = ...
    const moduleExportsRegex = /module\.exports\s*=\s*(\w+)/g;
    let match;
    while ((match = moduleExportsRegex.exec(content)) !== null) {
      exports.push({ name: match[1], source: match[1] });
    }
    
    // module.exports = { ... }
    const moduleExportsObjRegex = /module\.exports\s*=\s*{([^}]*)}/g;
    while ((match = moduleExportsObjRegex.exec(content)) !== null) {
      const objContent = match[1];
      const propRegex = /\b(\w+)(?:\s*:\s*(\w+))?\b/g;
      let propMatch;
      
      while ((propMatch = propRegex.exec(objContent)) !== null) {
        const name = propMatch[1];
        const source = propMatch[2] || propMatch[1];
        exports.push({ name, source });
      }
    }
    
    // export function ...
    const exportFuncRegex = /export\s+function\s+(\w+)/g;
    while ((match = exportFuncRegex.exec(content)) !== null) {
      exports.push({ name: match[1], source: match[1] });
    }
    
    // export class ...
    const exportClassRegex = /export\s+class\s+(\w+)/g;
    while ((match = exportClassRegex.exec(content)) !== null) {
      exports.push({ name: match[1], source: match[1] });
    }
    
    // export const ... = ...
    const exportConstRegex = /export\s+const\s+(\w+)\s*=/g;
    while ((match = exportConstRegex.exec(content)) !== null) {
      exports.push({ name: match[1], source: match[1] });
    }
    
    // export default ...
    const exportDefaultRegex = /export\s+default\s+(\w+)/g;
    while ((match = exportDefaultRegex.exec(content)) !== null) {
      exports.push({ name: 'default', source: match[1] });
    }
    
    return exports;
  }
  
  /**
   * Extract imports from code content
   * 
   * @param {string} content Code content
   * @param {string} filePath File path
   * @returns {Array<Object>} Extracted imports
   */
  function extractImports(content, filePath) {
    const imports = [];
    
    // import ... from '...'
    const importRegex = /import\s+(?:{([^}]*)}|(\w+))\s+from\s+['"]([^'"]+)['"]/g;
    let match;
    while ((match = importRegex.exec(content)) !== null) {
      const source = match[3];
      
      if (match[2]) {
        // Default import
        imports.push({ name: match[2], source });
      } else if (match[1]) {
        // Named imports
        const namedImports = match[1].split(',');
        for (const namedImport of namedImports) {
          const trimmed = namedImport.trim();
          if (trimmed) {
            // Handle aliased imports
            const aliasMatch = trimmed.match(/(\w+)(?:\s+as\s+(\w+))?/);
            if (aliasMatch) {
              const originalName = aliasMatch[1];
              const alias = aliasMatch[2] || originalName;
              imports.push({ name: alias, originalName, source });
            }
          }
        }
      }
    }
    
    // const ... = require('...')
    const requireRegex = /const\s+(?:{([^}]*)}|(\w+))\s*=\s*require\(['"]([^'"]+)['"]\)/g;
    while ((match = requireRegex.exec(content)) !== null) {
      const source = match[3];
      
      if (match[2]) {
        // Default require
        imports.push({ name: match[2], source });
      } else if (match[1]) {
        // Destructured require
        const namedImports = match[1].split(',');
        for (const namedImport of namedImports) {
          const trimmed = namedImport.trim();
          if (trimmed) {
            // Handle aliased imports
            const aliasMatch = trimmed.match(/(\w+)(?:\s*:\s*(\w+))?/);
            if (aliasMatch) {
              const originalName = aliasMatch[1];
              const alias = aliasMatch[2] || originalName;
              imports.push({ name: alias, originalName, source });
            }
          }
        }
      }
    }
    
    return imports;
  }
  
  /**
   * Search for code entities
   * 
   * @param {string} query The search query
   * @param {Object} options Search options
   * @param {number} options.limit Maximum number of results
   * @returns {Promise<Array<Object>>} The search results
   */
  async function search(query, options = {}) {
    try {
      const { limit = 5 } = options;
      
      // Create embedding for the query
      const embedding = await createEmbedding(query);
      
      // Calculate similarity scores
      const results = graph.nodes.map(node => ({
        ...node,
        score: cosineSimilarity(embedding, node.embedding || [])
      }));
      
      // Sort by score
      results.sort((a, b) => b.score - a.score);
      
      // Take top results
      const topResults = results.slice(0, limit);
      
      // Add related nodes
      for (const result of topResults) {
        // Find edges where this node is source or target
        const relatedEdges = graph.edges.filter(edge => 
          edge.source === result.id || edge.target === result.id
        );
        
        // Get related nodes
        const relatedNodeIds = new Set();
        for (const edge of relatedEdges) {
          if (edge.source === result.id) {
            relatedNodeIds.add(edge.target);
          } else {
            relatedNodeIds.add(edge.source);
          }
        }
        
        // Add related nodes to result
        result.related = Array.from(relatedNodeIds).map(id => {
          const node = graph.nodes.find(n => n.id === id);
          return node ? { id: node.id, name: node.name, type: node.type } : null;
        }).filter(Boolean);
      }
      
      return topResults;
    } catch (error) {
      logger.error(`Failed to search graph: ${error.message}`, { error });
      return [];
    }
  }
  
  /**
   * Calculate cosine similarity between two vectors
   * 
   * @param {Array<number>} vecA First vector
   * @param {Array<number>} vecB Second vector
   * @returns {number} Cosine similarity (-1 to 1)
   */
  function cosineSimilarity(vecA, vecB) {
    // Handle empty vectors
    if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0) {
      return 0;
    }
    
    // Calculate dot product
    let dotProduct = 0;
    for (let i = 0; i < Math.min(vecA.length, vecB.length); i++) {
      dotProduct += vecA[i] * vecB[i];
    }
    
    // Calculate magnitudes
    let magA = 0;
    let magB = 0;
    for (let i = 0; i < vecA.length; i++) {
      magA += vecA[i] * vecA[i];
    }
    for (let i = 0; i < vecB.length; i++) {
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
   * Save the graph to disk
   * 
   * @returns {Promise<void>}
   */
  async function saveGraph() {
    try {
      // Create a simplified graph for storage
      const storableGraph = {
        nodes: graph.nodes.map(node => ({
          id: node.id,
          name: node.name,
          type: node.type,
          filePath: node.filePath,
          targetPath: node.targetPath
          // Skip embedding to save space
        })),
        edges: graph.edges
      };
      
      // Ensure directory exists
      const dir = path.dirname(storePath);
      await fs.mkdir(dir, { recursive: true });
      
      // Save to file
      await fs.writeFile(
        storePath,
        JSON.stringify(storableGraph, null, 2),
        'utf8'
      );
    } catch (error) {
      logger.error(`Failed to save graph: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * Load the graph from disk
   * 
   * @returns {Promise<void>}
   */
  async function loadGraph() {
    try {
      // Check if file exists
      const exists = await fs.access(storePath)
        .then(() => true)
        .catch(() => false);
      
      if (exists) {
        // Load from file
        const data = await fs.readFile(storePath, 'utf8');
        const loadedGraph = JSON.parse(data);
        
        // Replace stored graph
        graph.nodes = loadedGraph.nodes;
        graph.edges = loadedGraph.edges;
        
        logger.info(`Loaded graph with ${graph.nodes.length} nodes and ${graph.edges.length} edges`);
      } else {
        logger.info('Graph file not found, starting with empty graph');
        graph = {
          nodes: [],
          edges: []
        };
      }
    } catch (error) {
      logger.error(`Failed to load graph: ${error.message}`, { error });
      graph = {
        nodes: [],
        edges: []
      };
    }
  }
  
  /**
   * Get GraphRAG status
   * 
   * @returns {Promise<Object>} The status
   */
  async function getStatus() {
    return {
      nodeCount: graph.nodes.length,
      edgeCount: graph.edges.length,
      lastUpdated
    };
  }
  
  /**
   * Clear the graph
   * 
   * @returns {Promise<void>}
   */
  async function clearGraph() {
    try {
      graph = {
        nodes: [],
        edges: []
      };
      lastUpdated = null;
      
      await saveGraph();
    } catch (error) {
      logger.error(`Failed to clear graph: ${error.message}`, { error });
      throw error;
    }
  }
  
  // Load existing graph
  loadGraph();
  
  // Return the GraphRAG module
  return {
    buildGraph,
    updateFileInGraph,
    search,
    getStatus,
    clearGraph
  };
}

module.exports = {
  createGraphRagModule
};