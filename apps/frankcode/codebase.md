# .eslintrc.js

```js
module.exports = {
    env: {
      node: true,
      es6: true,
      jest: true
    },
    extends: [
      'eslint:recommended',
      'plugin:node/recommended',
      'plugin:prettier/recommended'
    ],
    parserOptions: {
      ecmaVersion: 2020
    },
    rules: {
      'no-console': 'off',
      'node/no-unsupported-features/es-syntax': [
        'error',
        { ignores: ['modules'] }
      ],
      'prettier/prettier': 'warn',
      'node/no-unpublished-require': 'off'
    }
  };
```

# .gitignore

```
# Dependency directories
node_modules/
jspm_packages/

# Build outputs
dist/
build/
out/

# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Coverage directory used by tools like istanbul
coverage/
*.lcov

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Editor directories and files
.idea/
.vscode/
*.swp
*.swo
.DS_Store
Thumbs.db

# Temporary files
temp/
tmp/
```

# .prettierrc.js

```js
module.exports = {
    semi: true,
    singleQuote: true,
    trailingComma: 'none',
    printWidth: 80,
    tabWidth: 2,
    arrowParens: 'avoid'
  };
```

# config/default.js

```js
/**
 * Default Configuration for FrankCode
 */

module.exports = {
    // LLM connection settings
    llm: {
      api: 'ollama',           // 'ollama' or 'distributed'
      coordinatorHost: 'localhost',
      coordinatorPort: 11434,  // Default Ollama port
      model: 'llama2',
      temperature: 0.7
    },
    
    // Project context settings
    context: {
      maxTokens: 8192,
      excludeDirs: ['node_modules', 'dist', '.git', 'build', 'vendor'],
      excludeFiles: ['*.lock', '*.log', '*.min.js', '*.min.css', '*.bin', '*.exe'],
      priorityFiles: ['README.md', 'package.json', 'composer.json', 'requirements.txt', 'Makefile']
    },
    
    // UI preferences
    ui: {
      theme: 'dark',
      layout: 'default',
      fontSize: 14
    },
    
    // Logging settings
    logging: {
      level: 'info',
      console: true,
      file: true
    }
  };
```

# config/schema.js

```js
/**
 * Configuration Schema for FrankCode
 * 
 * Defines the schema for configuration validation
 */

module.exports = {
    type: 'object',
    properties: {
      // LLM connection settings
      llm: {
        type: 'object',
        properties: {
          api: {
            type: 'string',
            enum: ['ollama', 'distributed']
          },
          coordinatorHost: {
            type: 'string'
          },
          coordinatorPort: {
            type: 'number',
            minimum: 1,
            maximum: 65535
          },
          model: {
            type: 'string'
          },
          temperature: {
            type: 'number',
            minimum: 0,
            maximum: 1
          }
        },
        required: ['api', 'coordinatorHost', 'coordinatorPort', 'model'],
        additionalProperties: false
      },
      
      // Project context settings
      context: {
        type: 'object',
        properties: {
          maxTokens: {
            type: 'number',
            minimum: 1
          },
          excludeDirs: {
            type: 'array',
            items: {
              type: 'string'
            }
          },
          excludeFiles: {
            type: 'array',
            items: {
              type: 'string'
            }
          },
          priorityFiles: {
            type: 'array',
            items: {
              type: 'string'
            }
          }
        },
        required: ['maxTokens'],
        additionalProperties: false
      },
      
      // UI preferences
      ui: {
        type: 'object',
        properties: {
          theme: {
            type: 'string',
            enum: ['dark', 'light', 'dracula', 'solarized', 'nord']
          },
          layout: {
            type: 'string',
            enum: ['default', 'compact', 'wide']
          },
          fontSize: {
            type: 'number',
            minimum: 8,
            maximum: 24
          }
        },
        additionalProperties: false
      },
      
      // Logging settings
      logging: {
        type: 'object',
        properties: {
          level: {
            type: 'string',
            enum: ['error', 'warn', 'info', 'debug']
          },
          console: {
            type: 'boolean'
          },
          file: {
            type: 'boolean'
          },
          filePath: {
            type: 'string'
          },
          maxSize: {
            type: 'string'
          },
          maxFiles: {
            type: 'number',
            minimum: 1
          }
        },
        additionalProperties: false
      },
      
      // Project root
      projectRoot: {
        type: 'string'
      }
    },
    required: ['llm', 'context', 'ui', 'logging', 'projectRoot'],
    additionalProperties: false
  };
```

# jest.config.js

```js
/**
 * Jest Configuration
 */

module.exports = {
    testEnvironment: 'node',
    testMatch: ['**/__tests__/**/*.js', '**/?(*.)+(spec|test).js'],
    collectCoverage: true,
    coverageDirectory: 'coverage',
    collectCoverageFrom: [
      'src/**/*.js',
      '!**/node_modules/**',
      '!**/vendor/**',
      '!**/coverage/**'
    ],
    coverageReporters: ['text', 'lcov'],
    verbose: true
  };
```

# package.json

```json
{
    "name": "synthbot",
    "version": "0.1.0",
    "description": "Terminal-based coding agent powered by distributed LLMs",
    "main": "src/index.js",
    "bin": {
      "synthbot": "./bin/synthbot.js"
    },
    "scripts": {
      "start": "node src/index.js",
      "test": "jest",
      "lint": "eslint .",
      "format": "prettier --write \"**/*.{js,json}\"",
      "build": "npm run lint && npm run test"
    },
    "keywords": [
      "ai",
      "coding-assistant",
      "terminal",
      "tui",
      "llm",
      "distributed"
    ],
    "author": "",
    "license": "MIT",
    "dependencies": {
      "blessed": "^0.1.81",
      "blessed-contrib": "^4.11.0",
      "chalk": "^4.1.2",
      "commander": "^9.4.1",
      "cosmiconfig": "^8.0.0",
      "diff": "^5.1.0",
      "dotenv": "^16.0.3",
      "fast-glob": "^3.2.12",
      "ignore": "^5.2.4",
      "isomorphic-git": "^1.21.0",
      "keytar": "^7.9.0",
      "node-fetch": "^2.6.9",
      "simple-git": "^3.16.0",
      "tiktoken": "^1.0.3",
      "winston": "^3.8.2",
      "ws": "^8.12.1"
    },
    "devDependencies": {
      "eslint": "^8.34.0",
      "eslint-config-prettier": "^8.6.0",
      "eslint-plugin-node": "^11.1.0",
      "eslint-plugin-prettier": "^4.2.1",
      "jest": "^29.4.3",
      "prettier": "^2.8.4"
    },
    "engines": {
      "node": ">=14.0.0"
    }
  }
```

# README.md

```md
# FrankCode

FrankCode is an open-source terminal-based coding agent powered by distributed large language models. It helps developers write, modify, and understand code directly from their terminal, with full project context awareness.

<p align="center">
  <img src="assets/FrankCode-logo.png" alt="FrankCode Logo" width="200"/>
</p>

## Features

- **Terminal-Based UI**: Beautiful text-based interface that runs in your terminal.
- **Project-Aware**: Continuously scans your project files to maintain context.
- **Direct Code Modifications**: Makes changes to your files with your approval.
- **Context Management**: Monitors token usage and suggests session refreshes.
- **Git-Aware**: Understands git history and project structure.
- **Distributed LLM**: Leverages your local distributed LLM network for privacy and speed.

## Installation

\`\`\`bash
# Install globally
npm install -g frankcode

# Or install locally
npm install FrankCode
\`\`\`

## Usage

Navigate to your project's root directory and run:

\`\`\`bash
frankcode run
\`\`\`

Or if installed locally:

\`\`\`bash
npx frankcode run
\`\`\`

### Command Options

\`\`\`bash
# Start FrankCode with a specific configuration
frankcode run --config path/to/config.js

# Set the coordinator address for the distributed LLM
frankcode run --coordinator 192.168.1.100:5555

# Start with debug logging
frankcode run --verbose

# Show help
frankcode --help
\`\`\`

## Keyboard Shortcuts

Within the TUI:

- `Ctrl+C`: Exit SynthBot
- `Ctrl+S`: Save conversation
- `Ctrl+R`: Refresh project context
- `Ctrl+F`: Find in files
- `Tab`: Switch focus between panels
- `Ctrl+L`: Clear current conversation
- `Ctrl+Z`: Undo last file modification

## Configuration

Create a `.frankcoderc.js` file in your home or project directory:

\`\`\`javascript
module.exports = {
  // LLM connection settings
  llm: {
    coordinatorHost: "localhost",
    coordinatorPort: 5555,
    model: "llama-2-13b",
    temperature: 0.7
  },
  
  // Project context settings
  context: {
    maxTokens: 8192,
    excludeDirs: ["node_modules", "dist", ".git"],
    excludeFiles: ["*.lock", "*.log"],
    priorityFiles: ["README.md", "package.json"]
  },
  
  // UI preferences
  ui: {
    theme: "dark",
    layout: "default",
    fontSize: 14
  }
};
\`\`\`

## Architecture

SynthBot is organized into the following main components:

- **Agent**: Core logic for the LLM interaction and code understanding.
- **API**: Communication with the distributed LLM network.
- **TUI**: Terminal user interface components.
- **Utils**: Helper utilities for filesystem, git, etc.

\`\`\`
frankcode/
├── bin/           # Executable scripts
├── src/
│   ├── agent/     # LLM agent implementation
│   ├── api/       # API for LLM communication
│   ├── tui/       # Terminal UI components
│   └── utils/     # Utility functions
└── config/        # Configuration
\`\`\`

## Integrating with DistributedLLM

SynthBot is designed to work with your local DistributedLLM setup. Make sure your coordinator is running:

\`\`\`bash
# In your DistributedLLM directory
python src/main.py --mode coordinator
\`\`\`

SynthBot will automatically connect to the coordinator and use the distributed LLM network for all inferences.

## Contributing

Contributions welcome! Please check out our [contributing guidelines](CONTRIBUTING.md).

## License

MIT
```

# src/agent/agent.js

```js
/**
 * Core Agent Implementation
 * 
 * Manages interactions with the LLM, processes project context,
 * and handles file modifications.
 */

const path = require('path');
const fs = require('fs').promises;
const { logger } = require('../utils');
const { manageContext } = require('./context');
const { countTokens } = require('./tokenizer');
const { 
  readFile, 
  writeFile, 
  listDirectoryFiles,
  checkFileExists
} = require('./fileManager');

/**
 * Create the agent instance
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.apiClient API client for LLM communication
 * @param {Object} options.tokenMonitor Token usage monitor
 * @param {Array} options.projectFiles Initial list of project files
 * @param {string} options.projectRoot Project root directory
 * @returns {Object} The agent interface
 */
function createAgent({ apiClient, tokenMonitor, projectFiles, projectRoot }) {
  // Initialize conversation history
  let conversationHistory = [];
  
  // Initialize file context
  let fileContexts = new Map();
  
  // Initialize the context manager
  const contextManager = manageContext({
    tokenMonitor,
    maxSize: tokenMonitor.getMaxTokens()
  });
  
  /**
   * Load content of a file and add to context
   * 
   * @param {string} filePath Path to the file
   * @returns {Promise<string>} The file content
   */
  async function loadFileContext(filePath) {
    try {
      // Check if file exists
      const exists = await checkFileExists(filePath);
      if (!exists) {
        throw new Error(`File not found: ${filePath}`);
      }
      
      // Load file content
      const content = await readFile(filePath);
      
      // Calculate tokens
      const tokens = countTokens(content);
      
      // Store in context cache
      fileContexts.set(filePath, {
        content,
        tokens,
        lastModified: new Date()
      });
      
      // Add to context manager
      contextManager.addFileContext(filePath, content);
      
      return content;
    } catch (error) {
      logger.error(`Failed to load file context: ${filePath}`, { error });
      throw error;
    }
  }
  
  /**
   * Modify a file in the project
   * 
   * @param {string} filePath Path to the file
   * @param {string} newContent New file content
   * @returns {Promise<boolean>} Success indicator
   */
  async function modifyFile(filePath, newContent) {
    try {
      const fullPath = path.resolve(projectRoot, filePath);
      
      // Check if file exists
      const exists = await checkFileExists(fullPath);
      if (!exists) {
        throw new Error(`File not found: ${fullPath}`);
      }
      
      // Get original content for diff
      const originalContent = await readFile(fullPath);
      
      // Write new content
      await writeFile(fullPath, newContent);
      
      // Update context
      if (fileContexts.has(filePath)) {
        // Update cache
        fileContexts.set(filePath, {
          content: newContent,
          tokens: countTokens(newContent),
          lastModified: new Date()
        });
        
        // Update context manager
        contextManager.updateFileContext(filePath, newContent);
      }
      
      logger.info(`Modified file: ${filePath}`);
      return true;
    } catch (error) {
      logger.error(`Failed to modify file: ${filePath}`, { error });
      return false;
    }
  }
  
  /**
   * Scan for specific files in the project based on patterns
   * 
   * @param {string} pattern Glob pattern to match files
   * @returns {Promise<Array<string>>} Matching file paths
   */
  async function findFiles(pattern) {
    try {
      // Implementation would use something like fast-glob
      return [];
    } catch (error) {
      logger.error(`Failed to find files: ${pattern}`, { error });
      return [];
    }
  }
  
  /**
   * Process a user message and generate a response
   * 
   * @param {string} message User's message
   * @returns {Promise<Object>} Response object
   */
  async function processMessage(message) {
    try {
      // Add user message to conversation history
      conversationHistory.push({ role: 'user', content: message });
      
      // Get current context
      const context = contextManager.getCurrentContext();
      
      // Generate LLM prompt
      const prompt = generatePrompt(message, context);
      
      // Get response from LLM
      logger.debug('Sending prompt to LLM', { messageLength: message.length });
      const response = await apiClient.generateResponse(prompt);
      
      // Add assistant response to conversation history
      conversationHistory.push({ role: 'assistant', content: response.text });
      
      // Update token usage
      tokenMonitor.updateUsage(countTokens(response.text));
      
      // Check for file modifications in the response
      const fileModifications = parseFileModifications(response.text);
      
      // Return the processed response
      return {
        text: response.text,
        fileModifications,
        tokenUsage: tokenMonitor.getCurrentUsage()
      };
    } catch (error) {
      logger.error('Failed to process message', { error });
      throw error;
    }
  }
  
  /**
   * Generate a prompt for the LLM
   * 
   * @param {string} message User message
   * @param {Object} context Current context
   * @returns {string} Generated prompt
   */
  function generatePrompt(message, context) {
    // Basic prompt with system instruction and context
    const systemPrompt = `You are FrankCode, an AI coding assistant that helps developers understand and modify their codebase. 
You can read files and make changes to them directly. Always be concise and focus on providing practical solutions.
When modifying files, use the format:
\`\`\`file:path/to/file.js
// New content here
\`\`\`

Current project context:
${context.fileContexts.map(fc => `- ${fc.filePath} (${fc.summary})`).join('\n')}

Current conversation:
${conversationHistory.map(msg => `${msg.role.toUpperCase()}: ${msg.content}`).join('\n\n')}`;

    return `${systemPrompt}\n\nUSER: ${message}\n\nASSISTANT:`;
  }
  
  /**
   * Parse file modifications from LLM response
   * 
   * @param {string} response LLM response text
   * @returns {Array<Object>} Array of file modifications
   */
  function parseFileModifications(response) {
    const modifications = [];
    const fileBlockRegex = /\`\`\`file:(.*?)\n([\s\S]*?)\`\`\`/g;
    
    let match;
    while ((match = fileBlockRegex.exec(response)) !== null) {
      const filePath = match[1].trim();
      const content = match[2];
      
      modifications.push({
        filePath,
        content
      });
    }
    
    return modifications;
  }
  
  /**
   * Initialize agent context with project files
   * 
   * @returns {Promise<void>}
   */
  async function initialize() {
    logger.info('Initializing agent context...');
    
    // Load key project files
    for (const file of projectFiles.slice(0, 10)) {  // Start with first 10 for performance
      try {
        await loadFileContext(file);
      } catch (error) {
        // Continue with other files on error
        logger.warn(`Failed to load file: ${file}`, { error });
      }
    }
    
    logger.info('Agent context initialized');
  }
  
  // Initialize immediately
  initialize();
  
  // Return the agent interface
  return {
    processMessage,
    loadFileContext,
    modifyFile,
    findFiles,
    getContextManager: () => contextManager,
    getFileContexts: () => fileContexts,
    getConversationHistory: () => conversationHistory,
    reset: () => {
      conversationHistory = [];
      contextManager.reset();
    }
  };
}

module.exports = {
  createAgent
};
```

# src/agent/context.js

```js
/**
 * Context Management for the Agent
 * 
 * Handles the active context window, ensuring token limits are respected
 * and the most relevant information is included.
 */

const { logger } = require('../utils');
const { countTokens } = require('./tokenizer');

/**
 * Create a context manager
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.tokenMonitor Token usage monitor
 * @param {number} options.maxSize Maximum context size in tokens
 * @returns {Object} The context manager interface
 */
function manageContext({ tokenMonitor, maxSize }) {
  // Context storage
  const fileContexts = new Map();
  
  // Track total tokens used by context
  let totalContextTokens = 0;
  
  // LRU tracking for context eviction
  const fileAccessTimes = new Map();
  
  /**
   * Add a file to the context
   * 
   * @param {string} filePath Path to the file
   * @param {string} content File content
   * @returns {boolean} Success indicator
   */
  function addFileContext(filePath, content) {
    try {
      // Calculate tokens for this file content
      const tokens = countTokens(content);
      
      // Check if this will exceed our max context
      if (totalContextTokens + tokens > maxSize) {
        // Need to make room
        makeRoomInContext(tokens);
      }
      
      // Generate a summary for the file
      const summary = summarizeFile(filePath, content);
      
      // Add to context
      fileContexts.set(filePath, {
        content,
        tokens,
        summary
      });
      
      // Update total tokens
      totalContextTokens += tokens;
      
      // Update access time for LRU tracking
      fileAccessTimes.set(filePath, Date.now());
      
      logger.debug(`Added file to context: ${filePath} (${tokens} tokens)`);
      return true;
    } catch (error) {
      logger.error(`Failed to add file to context: ${filePath}`, { error });
      return false;
    }
  }
  
  /**
   * Update an existing file in context
   * 
   * @param {string} filePath Path to the file
   * @param {string} newContent New file content
   * @returns {boolean} Success indicator
   */
  function updateFileContext(filePath, newContent) {
    try {
      // Check if file exists in context
      if (!fileContexts.has(filePath)) {
        return addFileContext(filePath, newContent);
      }
      
      // Get existing context
      const existing = fileContexts.get(filePath);
      
      // Calculate new tokens
      const newTokens = countTokens(newContent);
      
      // Update total token count
      totalContextTokens = totalContextTokens - existing.tokens + newTokens;
      
      // If we exceed max, need to make room
      if (totalContextTokens > maxSize) {
        makeRoomInContext(0, [filePath]);  // Don't evict the file we're updating
      }
      
      // Generate a new summary
      const summary = summarizeFile(filePath, newContent);
      
      // Update context
      fileContexts.set(filePath, {
        content: newContent,
        tokens: newTokens,
        summary
      });
      
      // Update access time
      fileAccessTimes.set(filePath, Date.now());
      
      logger.debug(`Updated file in context: ${filePath} (${newTokens} tokens)`);
      return true;
    } catch (error) {
      logger.error(`Failed to update file in context: ${filePath}`, { error });
      return false;
    }
  }
  
  /**
   * Remove a file from context
   * 
   * @param {string} filePath Path to the file
   * @returns {boolean} Success indicator
   */
  function removeFileContext(filePath) {
    try {
      // Check if file exists in context
      if (!fileContexts.has(filePath)) {
        return false;
      }
      
      // Get existing context
      const existing = fileContexts.get(filePath);
      
      // Update total token count
      totalContextTokens -= existing.tokens;
      
      // Remove from context
      fileContexts.delete(filePath);
      
      // Remove from access times
      fileAccessTimes.delete(filePath);
      
      logger.debug(`Removed file from context: ${filePath}`);
      return true;
    } catch (error) {
      logger.error(`Failed to remove file from context: ${filePath}`, { error });
      return false;
    }
  }
  
  /**
   * Make room in the context by evicting least recently used files
   * 
   * @param {number} neededTokens Tokens needed for new content
   * @param {Array<string>} excludeFiles Files to exclude from eviction
   */
  function makeRoomInContext(neededTokens, excludeFiles = []) {
    // Continue evicting until we have enough room
    while (totalContextTokens + neededTokens > maxSize && fileContexts.size > 0) {
      // Create array of [filePath, lastAccessTime] for sorting
      const accessEntries = Array.from(fileAccessTimes.entries())
        .filter(([filePath]) => !excludeFiles.includes(filePath));
      
      // Sort by access time (oldest first)
      accessEntries.sort((a, b) => a[1] - b[1]);
      
      // If no files can be evicted, break
      if (accessEntries.length === 0) break;
      
      // Evict the oldest accessed file
      const [oldestFilePath] = accessEntries[0];
      
      // Remove it from context
      const evictedContext = fileContexts.get(oldestFilePath);
      fileContexts.delete(oldestFilePath);
      fileAccessTimes.delete(oldestFilePath);
      
      // Update token count
      totalContextTokens -= evictedContext.tokens;
      
      logger.debug(`Evicted file from context: ${oldestFilePath} (${evictedContext.tokens} tokens)`);
    }
    
    // If we still don't have enough room, log a warning
    if (totalContextTokens + neededTokens > maxSize) {
      logger.warn(`Context window capacity exceeded - some context may be lost`);
    }
  }
  
  /**
   * Get the current context for use in prompts
   * 
   * @returns {Object} The current context
   */
  function getCurrentContext() {
    return {
      fileContexts: Array.from(fileContexts.entries()).map(([filePath, context]) => ({
        filePath,
        summary: context.summary,
        content: context.content
      })),
      totalTokens: totalContextTokens
    };
  }
  
  /**
   * Reset the context manager, clearing all context
   */
  function reset() {
    fileContexts.clear();
    fileAccessTimes.clear();
    totalContextTokens = 0;
    logger.debug('Context manager reset');
  }
  
  /**
   * Generate a brief summary of a file
   * 
   * @param {string} filePath Path to the file
   * @param {string} content File content
   * @returns {string} File summary
   */
  function summarizeFile(filePath, content) {
    // Extract file extension
    const ext = filePath.split('.').pop().toLowerCase();
    
    // Simple summary based on file type
    if (['js', 'ts', 'jsx', 'tsx'].includes(ext)) {
      // JavaScript/TypeScript files: count functions, classes, imports
      const functionCount = (content.match(/function\s+\w+/g) || []).length;
      const arrowFunctionCount = (content.match(/const\s+\w+\s*=\s*(\(.*?\)|[^=]*?)\s*=>/g) || []).length;
      const classCount = (content.match(/class\s+\w+/g) || []).length;
      const importCount = (content.match(/import\s+/g) || []).length;
      
      return `JS file with ${functionCount + arrowFunctionCount} functions, ${classCount} classes, ${importCount} imports`;
    } else if (['json'].includes(ext)) {
      // JSON files: try to identify type from content or keys
      if (content.includes('"dependencies"') && content.includes('"name"')) {
        return 'Package configuration file';
      } else {
        return 'JSON configuration file';
      }
    } else if (['md', 'markdown'].includes(ext)) {
      // Markdown files: count headers
      const h1Count = (content.match(/^#\s+.+$/gm) || []).length;
      const h2Count = (content.match(/^##\s+.+$/gm) || []).length;
      
      return `Markdown document with ${h1Count} main sections, ${h2Count} subsections`;
    } else if (['html', 'htm'].includes(ext)) {
      return 'HTML document';
    } else if (['css', 'scss', 'sass', 'less'].includes(ext)) {
      return 'Stylesheet file';
    } else {
      // Default
      const lines = content.split('\n').length;
      return `File with ${lines} lines`;
    }
  }
  
  // Return the context manager interface
  return {
    addFileContext,
    updateFileContext,
    removeFileContext,
    getCurrentContext,
    reset,
    getTotalTokens: () => totalContextTokens
  };
}

module.exports = {
  manageContext
};
```

# src/agent/fileManager.js

```js
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
      absolute: true,
      dot: true
    });
    
    // Apply ignore rules
    const relativePaths = files.map(f => path.relative(dirPath, f));
    const filteredRelativePaths = ignoreRules.filter(relativePaths);
    
    // Convert back to absolute paths
    const filteredPaths = filteredRelativePaths.map(f => path.resolve(dirPath, f));
    
    return filteredPaths;
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
```

# src/agent/index.js

```js
/**
 * Agent module entry point
 * 
 * Exports the agent creation function and other utilities
 */

const { createAgent } = require('./agent');
const { manageContext } = require('./context');
const { countTokens, truncateToTokens } = require('./tokenizer');
const fileManager = require('./fileManager');
const { logger } = require('../utils');

/**
 * Initialize an agent instance
 * 
 * @param {Object} options Configuration options
 * @returns {Object} The agent instance
 */
function initAgent(options) {
  logger.info('Initializing agent...');
  
  const agent = createAgent(options);
  
  return agent;
}

module.exports = {
  initAgent,
  createAgent,
  manageContext,
  countTokens,
  truncateToTokens,
  fileManager
};
```

# src/agent/tokenizer.js

```js
/**
 * Tokenizer module for token counting
 * 
 * Provides utilities for counting tokens in text using tiktoken
 * This is important for managing context windows with LLMs
 */

const tiktoken = require('tiktoken');
const { logger } = require('../utils');

// Cache encoding to avoid recreating it on each call
let encoding;

/**
 * Initialize the tokenizer
 * 
 * @param {string} model Model name for tokenization (default is cl100k, used by many models)
 * @returns {void}
 */
function initTokenizer(model = 'cl100k_base') {
  try {
    encoding = tiktoken.getEncoding(model);
    logger.debug(`Tokenizer initialized with model: ${model}`);
  } catch (error) {
    logger.error('Failed to initialize tokenizer', { error });
    
    // Fallback to a simple tokenization method
    logger.warn('Using fallback tokenization method');
    encoding = null;
  }
}

/**
 * Count tokens in a text string
 * 
 * @param {string} text Text to count tokens for
 * @returns {number} Number of tokens
 */
function countTokens(text) {
  if (!text) return 0;
  
  try {
    // If encoding is not initialized, initialize it
    if (!encoding) {
      initTokenizer();
    }
    
    // If we have tiktoken encoding, use it
    if (encoding) {
      return encoding.encode(text).length;
    }
    
    // Fallback tokenization (approximate)
    // This is a simplistic approximation - in practice, real tokenization is more complex
    return Math.ceil(text.length / 4);
  } catch (error) {
    logger.error('Error counting tokens', { error });
    
    // Return a conservative estimate
    return Math.ceil(text.length / 3);
  }
}

/**
 * Truncate text to a maximum number of tokens
 * 
 * @param {string} text Text to truncate
 * @param {number} maxTokens Maximum number of tokens
 * @returns {string} Truncated text
 */
function truncateToTokens(text, maxTokens) {
  if (!text) return '';
  
  try {
    // If encoding is not initialized, initialize it
    if (!encoding) {
      initTokenizer();
    }
    
    // If we have tiktoken encoding, use it
    if (encoding) {
      const tokens = encoding.encode(text);
      
      if (tokens.length <= maxTokens) {
        return text;
      }
      
      const truncatedTokens = tokens.slice(0, maxTokens);
      return encoding.decode(truncatedTokens);
    }
    
    // Fallback truncation (approximate)
    const approxTokens = countTokens(text);
    
    if (approxTokens <= maxTokens) {
      return text;
    }
    
    const ratio = maxTokens / approxTokens;
    const charLimit = Math.floor(text.length * ratio);
    
    // Try to cut at a sentence or paragraph boundary
    const truncated = text.substring(0, charLimit);
    
    // Look for a good breakpoint near the end
    const breakpoints = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', '];
    
    for (const breakpoint of breakpoints) {
      const lastIdx = truncated.lastIndexOf(breakpoint);
      if (lastIdx !== -1 && lastIdx > charLimit * 0.8) {
        return truncated.substring(0, lastIdx + breakpoint.length) + '...';
      }
    }
    
    // If no good breakpoint, just cut and add ellipsis
    return truncated + '...';
  } catch (error) {
    logger.error('Error truncating text to tokens', { error });
    
    // Return a conservatively truncated text
    const approxCharPerToken = 4;
    const charLimit = maxTokens * approxCharPerToken;
    return text.substring(0, charLimit) + '...';
  }
}

// Initialize tokenizer on module load
initTokenizer();

module.exports = {
  countTokens,
  truncateToTokens,
  initTokenizer
};
```

# src/api/client.js

```js
/**
 * API Client for LLM Interaction
 * 
 * Supports both distributed LLM networks and Ollama
 */

const fetch = require('node-fetch');
const WebSocket = require('ws');
const { logger } = require('../utils');
const { createQueue } = require('./queue');
const { createWebSocketClient } = require('./websocket');

/**
 * Create an API client
 * 
 * @param {Object} options Configuration options
 * @param {string} options.host Host for the LLM service
 * @param {number} options.port Port for the LLM service
 * @param {string} options.model Model to use
 * @param {number} options.temperature Temperature setting (0-1)
 * @returns {Object} The API client interface
 */
function createClient(options) {
  const {
    host,
    port,
    model,
    temperature = 0.7,
    api = 'ollama' // 'ollama' or 'distributed'
  } = options;
  
  // State tracking
  let isConnected = false;
  let reconnectAttempts = 0;
  const MAX_RECONNECT_ATTEMPTS = 5;
  
  // Create request queue
  const requestQueue = createQueue({
    concurrency: 1, // Process one request at a time
    timeout: 60000   // 60 second timeout
  });
  
  // WebSocket client for distributed LLM
  let wsClient = null;
  
  /**
   * Initialize the client
   * 
   * @returns {Promise<void>}
   */
  async function initialize() {
    if (api === 'distributed') {
      // Initialize WebSocket connection for distributed mode
      wsClient = createWebSocketClient({
        host,
        port,
        model,
        onConnect: () => {
          isConnected = true;
          reconnectAttempts = 0;
          logger.info('Connected to distributed LLM network');
        },
        onDisconnect: () => {
          isConnected = false;
          logger.warn('Disconnected from distributed LLM network');
          
          // Try to reconnect
          if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
            
            logger.info(`Attempting to reconnect in ${delay / 1000} seconds...`);
            setTimeout(() => wsClient.connect(), delay);
          } else {
            logger.error('Max reconnection attempts reached');
          }
        },
        onError: (error) => {
          logger.error('WebSocket error', { error });
        }
      });
      
      await wsClient.connect();
    } else {
      // For Ollama, just test the connection
      try {
        const response = await fetch(`http://${host}:${port}/api/tags`);
        if (response.ok) {
          const data = await response.json();
          isConnected = true;
          
          // Log available models
          const models = data.models || [];
          logger.info(`Connected to Ollama with ${models.length} models available`);
          
          // Check if our model is available
          const modelAvailable = models.some(m => m.name === model);
          if (!modelAvailable) {
            logger.warn(`Model '${model}' not found in Ollama`);
          }
        } else {
          throw new Error(`Ollama API returned status ${response.status}`);
        }
      } catch (error) {
        logger.error('Failed to connect to Ollama API', { error });
        isConnected = false;
      }
    }
  }
  
  /**
   * Generate a response from the LLM
   * 
   * @param {string} prompt Prompt to send to the LLM
   * @param {Object} options Request options
   * @returns {Promise<Object>} Response object
   */
  async function generateResponse(prompt, options = {}) {
    const requestOptions = {
      temperature: options.temperature || temperature,
      maxTokens: options.maxTokens || 2048,
      stop: options.stop || ['\n\nUSER:', '\n\nASSISTANT:']
    };
    
    // Add to request queue
    return requestQueue.add(async () => {
      if (api === 'distributed') {
        return generateDistributedResponse(prompt, requestOptions);
      } else {
        return generateOllamaResponse(prompt, requestOptions);
      }
    });
  }
  
  /**
   * Generate a response using the distributed LLM network
   * 
   * @param {string} prompt Prompt to send
   * @param {Object} options Request options
   * @returns {Promise<Object>} Response object
   */
  async function generateDistributedResponse(prompt, options) {
    if (!isConnected || !wsClient) {
      throw new Error('Not connected to distributed LLM network');
    }
    
    try {
      const response = await wsClient.sendRequest({
        type: 'generate',
        payload: {
          prompt,
          model,
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stop_sequences: options.stop
        }
      });
      
      return {
        text: response.text,
        tokens: response.usage?.total_tokens || 0,
        model: response.model || model
      };
    } catch (error) {
      logger.error('Failed to generate response from distributed LLM', { error });
      throw error;
    }
  }
  
  /**
   * Generate a response using Ollama
   * 
   * @param {string} prompt Prompt to send
   * @param {Object} options Request options
   * @returns {Promise<Object>} Response object
   */
  async function generateOllamaResponse(prompt, options) {
    try {
      const response = await fetch(`http://${host}:${port}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model,
          prompt,
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stop: options.stop
        })
      });
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      
      return {
        text: data.response,
        tokens: data.eval_count || 0,
        model: data.model || model
      };
    } catch (error) {
      logger.error('Failed to generate response from Ollama', { error });
      throw error;
    }
  }
  
  /**
   * Stream a response from the LLM
   * 
   * @param {string} prompt Prompt to send
   * @param {Function} onToken Callback for each token
   * @param {Function} onComplete Callback when complete
   * @param {Object} options Request options
   * @returns {Promise<void>}
   */
  async function streamResponse(prompt, onToken, onComplete, options = {}) {
    const requestOptions = {
      temperature: options.temperature || temperature,
      maxTokens: options.maxTokens || 2048,
      stop: options.stop || ['\n\nUSER:', '\n\nASSISTANT:']
    };
    
    // Add to request queue
    return requestQueue.add(async () => {
      if (api === 'distributed') {
        return streamDistributedResponse(prompt, onToken, onComplete, requestOptions);
      } else {
        return streamOllamaResponse(prompt, onToken, onComplete, requestOptions);
      }
    });
  }
  
  /**
   * Stream a response from the distributed LLM network
   * 
   * @param {string} prompt Prompt to send
   * @param {Function} onToken Callback for each token
   * @param {Function} onComplete Callback when complete
   * @param {Object} options Request options
   * @returns {Promise<void>}
   */
  async function streamDistributedResponse(prompt, onToken, onComplete, options) {
    if (!isConnected || !wsClient) {
      throw new Error('Not connected to distributed LLM network');
    }
    
    try {
      await wsClient.streamRequest({
        type: 'generate_stream',
        payload: {
          prompt,
          model,
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stop_sequences: options.stop
        },
        onToken,
        onComplete
      });
    } catch (error) {
      logger.error('Failed to stream response from distributed LLM', { error });
      throw error;
    }
  }
  
  /**
   * Stream a response from Ollama
   * 
   * @param {string} prompt Prompt to send
   * @param {Function} onToken Callback for each token
   * @param {Function} onComplete Callback when complete
   * @param {Object} options Request options
   * @returns {Promise<void>}
   */
  async function streamOllamaResponse(prompt, onToken, onComplete, options) {
    try {
      const response = await fetch(`http://${host}:${port}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model,
          prompt,
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stop: options.stop,
          stream: true
        })
      });
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      // Parse the stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let totalTokens = 0;
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        // Decode the chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Process complete JSON objects from the buffer
        let newlineIndex;
        while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + 1);
          
          if (line.trim() === '') continue;
          
          try {
            const data = JSON.parse(line);
            
            if (data.response) {
              onToken(data.response);
              totalTokens++;
            }
            
            // Check if done
            if (data.done) {
              onComplete({
                model: data.model || model,
                tokens: totalTokens
              });
              return;
            }
          } catch (error) {
            logger.error('Error parsing Ollama stream', { error, line });
          }
        }
      }
      
      // Final completion if we haven't already called it
      onComplete({
        model,
        tokens: totalTokens
      });
    } catch (error) {
      logger.error('Failed to stream response from Ollama', { error });
      throw error;
    }
  }
  
  /**
   * Execute a shell command using the LLM
   * 
   * @param {string} command Command to execute
   * @returns {Promise<Object>} Execution result
   */
  async function executeCommand(command) {
    // This is handled by the agent, not the LLM directly
    throw new Error('Command execution should be handled by the agent');
  }
  
  /**
   * Check if the client is connected
   * 
   * @returns {boolean} Connection status
   */
  function getConnectionStatus() {
    return isConnected;
  }
  
  /**
   * Get information about the current model
   * 
   * @returns {Promise<Object>} Model information
   */
  async function getModelInfo() {
    try {
      if (api === 'ollama') {
        const response = await fetch(`http://${host}:${port}/api/show`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            name: model
          })
        });
        
        if (!response.ok) {
          throw new Error(`Ollama API returned status ${response.status}`);
        }
        
        const data = await response.json();
        
        return {
          name: data.model || model,
          parameters: data.parameters || {},
          size: data.size || 'unknown',
          family: data.family || 'unknown'
        };
      } else {
        // For distributed LLM
        if (!wsClient) {
          throw new Error('Not connected to distributed LLM network');
        }
        
        const response = await wsClient.sendRequest({
          type: 'model_info',
          payload: { model }
        });
        
        return {
          name: response.name || model,
          parameters: response.parameters || {},
          size: response.size || 'unknown',
          family: response.family || 'unknown'
        };
      }
    } catch (error) {
      logger.error('Failed to get model info', { error });
      throw error;
    }
  }
  
  // Initialize on creation
  initialize().catch(error => {
    logger.error('Failed to initialize API client', { error });
  });
  
  // Return the client interface
  return {
    generateResponse,
    streamResponse,
    executeCommand,
    getConnectionStatus,
    getModelInfo,
    reconnect: initialize
  };
}
```

# src/api/index.js

```js
/**
 * API module entry point
 * 
 * Exports the client creation function and other utilities
 */

const { createClient } = require('./client');
const { createQueue } = require('./queue');
const { createWebSocketClient } = require('./websocket');

module.exports = {
  createClient,
  createQueue,
  createWebSocketClient
};
```

# src/api/queue.js

```js
/**
 * Request Queue for API Calls
 * 
 * Manages a queue of API requests to ensure they are processed
 * in order and with proper concurrency control.
 */

const { logger } = require('../utils');

/**
 * Create a request queue
 * 
 * @param {Object} options Configuration options
 * @param {number} options.concurrency Maximum concurrent requests
 * @param {number} options.timeout Request timeout in milliseconds
 * @returns {Object} The queue interface
 */
function createQueue(options = {}) {
  const {
    concurrency = 1,
    timeout = 60000
  } = options;
  
  // Queue state
  const queue = [];
  let activeCount = 0;
  
  /**
   * Add a task to the queue
   * 
   * @param {Function} task Task function that returns a promise
   * @returns {Promise} Promise that resolves with the task result
   */
  function add(task) {
    return new Promise((resolve, reject) => {
      // Create a timeout handler
      const timeoutId = setTimeout(() => {
        // Find this task in the queue
        const index = queue.findIndex(item => item.task === task);
        
        if (index !== -1) {
          // Remove from queue
          queue.splice(index, 1);
          reject(new Error('Task timed out'));
        }
      }, timeout);
      
      // Add to queue
      queue.push({
        task,
        resolve: (result) => {
          clearTimeout(timeoutId);
          resolve(result);
        },
        reject: (error) => {
          clearTimeout(timeoutId);
          reject(error);
        }
      });
      
      // Process queue
      processQueue();
    });
  }
  
  /**
   * Process the next items in the queue
   */
  function processQueue() {
    // Check if we can process more tasks
    if (activeCount >= concurrency) {
      return;
    }
    
    // Process as many as we can up to concurrency limit
    while (queue.length > 0 && activeCount < concurrency) {
      const { task, resolve, reject } = queue.shift();
      
      // Increment active count
      activeCount++;
      
      // Execute the task
      Promise.resolve().then(() => task())
        .then(result => {
          // Task completed successfully
          activeCount--;
          resolve(result);
          
          // Process next items
          processQueue();
        })
        .catch(error => {
          // Task failed
          activeCount--;
          reject(error);
          
          // Process next items
          processQueue();
        });
    }
  }
  
  /**
   * Get the current queue size
   * 
   * @returns {number} Queue size
   */
  function size() {
    return queue.length;
  }
  
  /**
   * Get the current active count
   * 
   * @returns {number} Active count
   */
  function active() {
    return activeCount;
  }
  
  /**
   * Clear the queue, rejecting all pending tasks
   */
  function clear() {
    // Reject all pending tasks
    queue.forEach(item => {
      item.reject(new Error('Queue cleared'));
    });
    
    // Clear the queue
    queue.length = 0;
  }
  
  // Return the queue interface
  return {
    add,
    size,
    active,
    clear
  };
}

module.exports = {
  createQueue
};
```

# src/api/websocket.js

```js
/**
 * WebSocket Client for Distributed LLM Communication
 * 
 * Manages WebSocket connections to the distributed LLM network
 */

const WebSocket = require('ws');
const { logger } = require('../utils');

/**
 * Create a WebSocket client
 * 
 * @param {Object} options Configuration options
 * @param {string} options.host Host for WebSocket connection
 * @param {number} options.port Port for WebSocket connection
 * @param {Function} options.onConnect Connection callback
 * @param {Function} options.onDisconnect Disconnect callback
 * @param {Function} options.onError Error callback
 * @returns {Object} The WebSocket client interface
 */
function createWebSocketClient(options) {
  const {
    host,
    port,
    model,
    onConnect = () => {},
    onDisconnect = () => {},
    onError = () => {}
  } = options;
  
  // WebSocket instance
  let ws = null;
  
  // Request handling
  let nextRequestId = 1;
  const pendingRequests = new Map();
  
  /**
   * Connect to the WebSocket server
   * 
   * @returns {Promise<void>}
   */
  function connect() {
    return new Promise((resolve, reject) => {
      try {
        // Close existing connection if any
        if (ws) {
          ws.terminate();
        }
        
        // Create new WebSocket
        ws = new WebSocket(`ws://${host}:${port}`);
        
        // Set up event handlers
        ws.on('open', () => {
          logger.info(`WebSocket connected to ${host}:${port}`);
          onConnect();
          resolve();
        });
        
        ws.on('close', () => {
          logger.info('WebSocket connection closed');
          onDisconnect();
        });
        
        ws.on('error', (error) => {
          logger.error('WebSocket error', { error });
          onError(error);
          reject(error);
        });
        
        ws.on('message', (data) => {
          try {
            const message = JSON.parse(data);
            handleMessage(message);
          } catch (error) {
            logger.error('Error parsing WebSocket message', { error });
          }
        });
      } catch (error) {
        logger.error('Error connecting to WebSocket', { error });
        reject(error);
      }
    });
  }
  
  /**
   * Handle incoming messages
   * 
   * @param {Object} message Message object
   */
  function handleMessage(message) {
    // Check for response ID
    if (message.id && pendingRequests.has(message.id)) {
      const { resolve, reject, streamHandlers } = pendingRequests.get(message.id);
      
      // Handle streaming responses
      if (message.type === 'stream_token' && streamHandlers) {
        streamHandlers.onToken(message.token);
        return;
      }
      
      // Handle stream completion
      if (message.type === 'stream_end' && streamHandlers) {
        streamHandlers.onComplete(message.payload);
        pendingRequests.delete(message.id);
        return;
      }
      
      // Handle errors
      if (message.error) {
        reject(new Error(message.error));
        pendingRequests.delete(message.id);
        return;
      }
      
      // Handle normal responses
      resolve(message.payload);
      pendingRequests.delete(message.id);
    }
  }
  
  /**
   * Send a request to the server
   * 
   * @param {Object} request Request object
   * @returns {Promise<Object>} Response object
   */
  function sendRequest(request) {
    return new Promise((resolve, reject) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket not connected'));
        return;
      }
      
      // Generate request ID
      const id = nextRequestId++;
      
      // Store the promise callbacks
      pendingRequests.set(id, { resolve, reject });
      
      // Send the request
      ws.send(JSON.stringify({
        id,
        ...request
      }));
      
      // Set timeout
      setTimeout(() => {
        if (pendingRequests.has(id)) {
          pendingRequests.delete(id);
          reject(new Error('Request timed out'));
        }
      }, 60000); // 60 second timeout
    });
  }
  
  /**
   * Stream a request to the server
   * 
   * @param {Object} request Request object
   * @param {Function} onToken Token callback
   * @param {Function} onComplete Completion callback
   * @returns {Promise<void>}
   */
  function streamRequest({ type, payload, onToken, onComplete }) {
    return new Promise((resolve, reject) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket not connected'));
        return;
      }
      
      // Generate request ID
      const id = nextRequestId++;
      
      // Store callbacks
      pendingRequests.set(id, {
        resolve,
        reject,
        streamHandlers: { onToken, onComplete }
      });
      
      // Send the request
      ws.send(JSON.stringify({
        id,
        type,
        payload,
        stream: true
      }));
      
      // Request is resolved when we start receiving the stream
      resolve();
      
      // Set timeout
      setTimeout(() => {
        if (pendingRequests.has(id)) {
          pendingRequests.delete(id);
          reject(new Error('Stream request timed out'));
        }
      }, 60000); // 60 second timeout
    });
  }
  
  /**
   * Close the WebSocket connection
   */
  function close() {
    if (ws) {
      ws.close();
      ws = null;
    }
  }
  
  /**
   * Check if connected
   * 
   * @returns {boolean} Connection status
   */
  function isConnected() {
    return ws && ws.readyState === WebSocket.OPEN;
  }
  
  // Return the client interface
  return {
    connect,
    sendRequest,
    streamRequest,
    close,
    isConnected
  };
}

module.exports = {
  createWebSocketClient
};
```

# src/index.js

```js
/**
 * SynthBot - Main application entry point
 * 
 * This module initializes and coordinates the different components:
 * - Terminal UI
 * - Agent
 * - Distributed LLM API client
 */

const { initAgent } = require('./agent');
const { createClient } = require('./api');
const { createApp } = require('./tui');
const { 
  logger, 
  scanProjectFiles, 
  setupLogging,
  createTokenMonitor
} = require('./utils');

/**
 * Start the SynthBot application
 * @param {Object} config Configuration object
 */
async function startApp(config) {
  try {
    // Setup logging based on configuration
    setupLogging(config.logging);
    
    logger.info('Starting SynthBot...');
    logger.debug('Configuration loaded', { config });
    
    // Initial project scan
    logger.info('Scanning project files...');
    const projectFiles = await scanProjectFiles(config.projectRoot, {
      exclude: config.context.excludeDirs.concat(config.context.excludeFiles),
      prioritize: config.context.priorityFiles
    });
    
    logger.info(`Found ${projectFiles.length} files in project`);
    
    // Initialize API client
    logger.info('Connecting to DistributedLLM coordinator...');
    const apiClient = createClient({
      host: config.llm.coordinatorHost,
      port: config.llm.coordinatorPort,
      model: config.llm.model,
      temperature: config.llm.temperature
    });
    
    // Create token monitor
    const tokenMonitor = createTokenMonitor({
      maxTokens: config.context.maxTokens,
      warningThreshold: 0.8 // Warn at 80% usage
    });
    
    // Initialize agent
    logger.info('Initializing agent...');
    const agent = initAgent({
      apiClient,
      tokenMonitor,
      projectFiles,
      projectRoot: config.projectRoot
    });
    
    // Create and start the TUI
    logger.info('Starting Terminal UI...');
    const app = createApp({
      agent,
      apiClient,
      tokenMonitor,
      config: config.ui,
      projectRoot: config.projectRoot
    });
    
    // Handle application shutdown
    function shutdown() {
      logger.info('Shutting down FrankCode...');
      app.destroy();
      process.exit(0);
    }
    
    // Handle signals
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
    
    // Start the TUI
    app.start();
    
    logger.info('SynthBot started successfully');
    
  } catch (error) {
    logger.error('Failed to start FrankCode', { error });
    console.error('Failed to start FrankCode:', error.message);
    process.exit(1);
  }
}

module.exports = {
  startApp
};
```

# src/tui/app.js

```js
/**
 * Main Terminal UI Application
 * 
 * Creates and manages the terminal interface using blessed/blessed-contrib
 */

const blessed = require('blessed');
const contrib = require('blessed-contrib');
const { applyTheme } = require('./themes');
const { createInputHandler } = require('./input');
const { createOutputRenderer } = require('./output');
const { createStatusBar } = require('./statusBar');
const { createFileTree } = require('./fileTree');
const { logger } = require('../utils');

/**
 * Create the TUI application
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.agent The agent instance
 * @param {Object} options.apiClient The API client
 * @param {Object} options.tokenMonitor Token monitoring utility
 * @param {Object} options.config UI configuration
 * @param {string} options.projectRoot Project root directory
 * @returns {Object} The TUI application object
 */
function createApp({ agent, apiClient, tokenMonitor, config, projectRoot }) {
  // Create a screen object
  const screen = blessed.screen({
    smartCSR: true,
    title: 'FrankCode',
    fullUnicode: true,
    dockBorders: true,
    autoPadding: true
  });
  
  // Apply theme
  applyTheme(screen, config.theme);
  
  // Create layout grid
  const grid = new contrib.grid({ rows: 12, cols: 12, screen });
  
  // Create file tree panel (left side, 2/12 of width)
  const fileTreePanel = grid.set(0, 0, 10, 2, contrib.tree, {
    label: 'Project Files',
    style: {
      selected: {
        fg: 'white',
        bg: 'blue'
      }
    },
    template: {
      lines: true
    },
    tags: true
  });
  
  // Create conversation panel (right side, 10/12 of width)
  const conversationPanel = grid.set(0, 2, 8, 10, blessed.log, {
    label: 'Conversation',
    scrollable: true,
    alwaysScroll: true,
    scrollbar: {
      ch: ' ',
      style: {
        bg: 'blue'
      },
      track: {
        bg: 'black'
      }
    },
    mouse: true,
    tags: true
  });
  
  // Create input box (bottom right, span 10/12 of width)
  const inputBox = grid.set(8, 2, 2, 10, blessed.textarea, {
    label: 'Command',
    inputOnFocus: true,
    padding: {
      top: 1,
      left: 2
    },
    style: {
      focus: {
        border: {
          fg: 'blue'
        }
      }
    },
    border: {
      type: 'line'
    }
  });
  
  // Create status bar (bottom of screen, full width)
  const statusBar = grid.set(10, 0, 2, 12, blessed.box, {
    tags: true,
    content: ' {bold}Status:{/bold} Ready',
    style: {
      fg: 'white',
      bg: 'blue'
    },
    padding: {
      left: 1,
      right: 1
    }
  });
  
  // Initialize components
  const fileTree = createFileTree({
    widget: fileTreePanel,
    projectRoot,
    agent
  });
  
  const outputRenderer = createOutputRenderer({
    widget: conversationPanel,
    tokenMonitor
  });
  
  const statusBarController = createStatusBar({
    widget: statusBar,
    apiClient,
    tokenMonitor
  });
  
  const inputHandler = createInputHandler({
    widget: inputBox,
    outputRenderer,
    agent,
    fileTree,
    screen
  });
  
  // Set up key bindings
  screen.key(['C-c'], () => {
    return process.exit(0);
  });
  
  screen.key(['tab'], () => {
    if (screen.focused === inputBox) {
      fileTreePanel.focus();
    } else {
      inputBox.focus();
    }
  });
  
  screen.key(['C-r'], () => {
    fileTree.refresh();
    statusBarController.update('Refreshing project files...');
  });
  
  screen.key(['C-l'], () => {
    conversationPanel.setContent('');
    screen.render();
    statusBarController.update('Conversation cleared');
  });
  
  screen.key(['C-s'], () => {
    // Save conversation implementation
    statusBarController.update('Conversation saved');
  });
  
  // Focus input by default
  inputBox.focus();
  
  // Initialize file tree
  fileTree.init();
  
  // Return the application object
  return {
    screen,
    start: () => {
      // Initial rendering
      screen.render();
      
      // Welcome message
      outputRenderer.addSystemMessage('Welcome to FrankCode! Type your question or command below.');
      statusBarController.update('Ready');
    },
    destroy: () => {
      screen.destroy();
    }
  };
}

module.exports = {
  createApp
};
```

# src/tui/fileTree.js

```js
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
```

# src/tui/index.js

```js
/**
 * TUI module entry point
 * 
 * Exports the TUI creation function and components
 */

const { createApp } = require('./app');
const { createFileTree } = require('./fileTree');
const { createInputHandler } = require('./input');
const { createOutputRenderer } = require('./output');
const { createStatusBar } = require('./statusBar');
const { applyTheme, getAvailableThemes } = require('./themes');

module.exports = {
  createApp,
  createFileTree,
  createInputHandler,
  createOutputRenderer,
  createStatusBar,
  applyTheme,
  getAvailableThemes
};
```

# src/tui/input.js

```js
/**
 * Input Handler for TUI
 * 
 * Manages user input and command processing
 */

const { logger } = require('../utils');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);
const path = require('path');

/**
 * Create an input handler
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.widget The input widget
 * @param {Object} options.outputRenderer Output renderer
 * @param {Object} options.agent Agent instance
 * @param {Object} options.fileTree File tree component
 * @param {Object} options.screen Blessed screen
 * @returns {Object} The input handler interface
 */
function createInputHandler({ widget, outputRenderer, agent, fileTree, screen }) {
  // Command history
  const history = [];
  let historyIndex = -1;
  
  // File modification tracking
  const pendingModifications = [];
  let currentModificationIndex = -1;
  
  // Initialize
  function init() {
    // Handle input submission
    widget.key('enter', async () => {
      const input = widget.getValue().trim();
      
      if (input === '') {
        return;
      }
      
      // Add to history
      history.push(input);
      historyIndex = history.length;
      
      // Clear input
      widget.setValue('');
      widget.screen.render();
      
      // Process the input
      await processInput(input);
    });
    
    // Handle history navigation
    widget.key('up', () => {
      if (historyIndex > 0) {
        historyIndex--;
        widget.setValue(history[historyIndex]);
        widget.screen.render();
      }
    });
    
    widget.key('down', () => {
      if (historyIndex < history.length - 1) {
        historyIndex++;
        widget.setValue(history[historyIndex]);
      } else {
        historyIndex = history.length;
        widget.setValue('');
      }
      widget.screen.render();
    });
    
    // Handle tab completion
    widget.key('tab', () => {
      // Implement tab completion later
    });
  }
  
  /**
   * Process user input
   * 
   * @param {string} input User input
   */
  async function processInput(input) {
    try {
      // Check for command prefixes
      if (input.startsWith('/')) {
        await processCommand(input.slice(1));
        return;
      }
      
      // Display user input
      outputRenderer.addUserMessage(input);
      
      // Send to agent for processing
      const response = await agent.processMessage(input);
      
      // Display assistant response
      outputRenderer.addAssistantMessage(response.text);
      
      // Check for file modifications
      if (response.fileModifications && response.fileModifications.length > 0) {
        await handleFileModifications(response.fileModifications);
      }
    } catch (error) {
      logger.error('Failed to process input', { error });
      outputRenderer.addErrorMessage(`Error processing input: ${error.message}`);
    }
  }
  
  /**
   * Process a command
   * 
   * @param {string} command Command string
   */
  async function processCommand(command) {
    try {
      // Split command and arguments
      const parts = command.split(/\s+/);
      const cmd = parts[0].toLowerCase();
      const args = parts.slice(1);
      
      // Process different commands
      switch (cmd) {
        case 'help':
          showHelp();
          break;
        
        case 'clear':
          outputRenderer.clear();
          break;
        
        case 'exit':
        case 'quit':
          process.exit(0);
          break;
        
        case 'refresh':
          await fileTree.refresh();
          outputRenderer.addSystemMessage('File tree refreshed');
          break;
        
        case 'exec':
        case 'shell':
          await executeShellCommand(args.join(' '));
          break;
        
        case 'load':
          await loadFile(args[0]);
          break;
        
        case 'save':
          await saveConversation(args[0]);
          break;
        
        case 'reset':
          agent.reset();
          outputRenderer.addSystemMessage('Agent context and conversation reset');
          break;
        
        case 'yes':
        case 'y':
          await approveModification(true);
          break;
        
        case 'no':
        case 'n':
          await approveModification(false);
          break;
        
        case 'yesall':
        case 'ya':
          await approveAllModifications();
          break;
        
        case 'noall':
        case 'na':
          await rejectAllModifications();
          break;
        
        default:
          outputRenderer.addSystemMessage(`Unknown command: ${cmd}. Type /help for available commands.`);
      }
    } catch (error) {
      logger.error('Failed to process command', { error });
      outputRenderer.addErrorMessage(`Error processing command: ${error.message}`);
    }
  }
  
  /**
   * Show help text
   */
  function showHelp() {
    const helpText = `
Available commands:
  /help             - Show this help text
  /clear            - Clear the conversation
  /exit, /quit      - Exit the application
  /refresh          - Refresh the file tree
  /exec <command>   - Execute a shell command
  /shell <command>  - Same as /exec
  /load <file>      - Load a file into context
  /save <file>      - Save the conversation to a file
  /reset            - Reset agent context and conversation
  /yes, /y          - Approve the current file modification
  /no, /n           - Reject the current file modification
  /yesall, /ya      - Approve all pending file modifications
  /noall, /na       - Reject all pending file modifications
    `;
    
    outputRenderer.addSystemMessage(helpText);
  }
  
  /**
   * Handle file modifications
   * 
   * @param {Array<Object>} modifications File modifications
   */
  async function handleFileModifications(modifications) {
    if (modifications.length === 0) {
      return;
    }
    
    // Store modifications
    pendingModifications.length = 0;
    pendingModifications.push(...modifications);
    currentModificationIndex = 0;
    
    // Display the first modification
    displayCurrentModification();
  }
  
  /**
   * Display the current modification
   */
  function displayCurrentModification() {
    if (pendingModifications.length === 0 || currentModificationIndex >= pendingModifications.length) {
      return;
    }
    
    const mod = pendingModifications[currentModificationIndex];
    
    // Show the file modification
    outputRenderer.addSystemMessage(`Proposed file modification (${currentModificationIndex + 1}/${pendingModifications.length}):`);
    outputRenderer.addCodeBlock(`File: ${mod.filePath}\n\n${mod.content}`, 'javascript');
    outputRenderer.addSystemMessage('Type /yes (or /y) to approve, /no (or /n) to reject, /yesall (or /ya) to approve all, /noall (or /na) to reject all');
  }
  
  /**
   * Approve or reject a modification
   * 
   * @param {boolean} approve Whether to approve the modification
   */
  async function approveModification(approve) {
    if (pendingModifications.length === 0 || currentModificationIndex >= pendingModifications.length) {
      outputRenderer.addSystemMessage('No pending file modifications.');
      return;
    }
    
    const mod = pendingModifications[currentModificationIndex];
    
    if (approve) {
      try {
        // Apply the modification
        const fullPath = path.resolve(path.join(screen.cwd, mod.filePath));
        await agent.modifyFile(fullPath, mod.content);
        
        outputRenderer.addSystemMessage(`✅ Applied changes to ${mod.filePath}`);
      } catch (error) {
        logger.error(`Failed to apply modification to ${mod.filePath}`, { error });
        outputRenderer.addErrorMessage(`Error applying modification: ${error.message}`);
      }
    } else {
      outputRenderer.addSystemMessage(`❌ Rejected changes to ${mod.filePath}`);
    }
    
    // Move to next modification
    currentModificationIndex++;
    
    // Display next modification if any
    if (currentModificationIndex < pendingModifications.length) {
      displayCurrentModification();
    } else {
      outputRenderer.addSystemMessage('All modifications processed.');
    }
  }
  
  /**
   * Approve all pending modifications
   */
  async function approveAllModifications() {
    if (pendingModifications.length === 0) {
      outputRenderer.addSystemMessage('No pending file modifications.');
      return;
    }
    
    const total = pendingModifications.length;
    let applied = 0;
    let failed = 0;
    
    for (const mod of pendingModifications) {
      try {
        // Apply the modification
        const fullPath = path.resolve(path.join(screen.cwd, mod.filePath));
        await agent.modifyFile(fullPath, mod.content);
        applied++;
      } catch (error) {
        logger.error(`Failed to apply modification to ${mod.filePath}`, { error });
        failed++;
      }
    }
    
    outputRenderer.addSystemMessage(`Applied ${applied}/${total} modifications (${failed} failed).`);
    
    // Clear pending modifications
    pendingModifications.length = 0;
    currentModificationIndex = 0;
  }
  
  /**
   * Reject all pending modifications
   */
  function rejectAllModifications() {
    if (pendingModifications.length === 0) {
      outputRenderer.addSystemMessage('No pending file modifications.');
      return;
    }
    
    const total = pendingModifications.length;
    outputRenderer.addSystemMessage(`Rejected all ${total} pending modifications.`);
    
    // Clear pending modifications
    pendingModifications.length = 0;
    currentModificationIndex = 0;
  }
  
  /**
   * Execute a shell command
   * 
   * @param {string} command Command to execute
   */
  async function executeShellCommand(command) {
    if (!command) {
      outputRenderer.addErrorMessage('No command specified');
      return;
    }
    
    try {
      outputRenderer.addSystemMessage(`Executing: ${command}`);
      
      const { stdout, stderr } = await execPromise(command, {
        cwd: screen.cwd,
        shell: true
      });
      
      if (stdout) {
        outputRenderer.addCodeBlock(stdout, 'shell');
      }
      
      if (stderr) {
        outputRenderer.addErrorMessage(stderr);
      }
    } catch (error) {
      outputRenderer.addErrorMessage(`Command failed: ${error.message}`);
      if (error.stdout) {
        outputRenderer.addCodeBlock(error.stdout, 'shell');
      }
      if (error.stderr) {
        outputRenderer.addErrorMessage(error.stderr);
      }
    }
  }
  
  /**
   * Load a file into context
   * 
   * @param {string} filePath Path to the file
   */
  async function loadFile(filePath) {
    if (!filePath) {
      outputRenderer.addErrorMessage('No file specified');
      return;
    }
    
    try {
      // Resolve path
      const fullPath = path.resolve(path.join(screen.cwd, filePath));
      
      // Load into agent context
      await agent.loadFileContext(fullPath);
      
      outputRenderer.addSystemMessage(`Loaded file into context: ${filePath}`);
    } catch (error) {
      logger.error(`Failed to load file: ${filePath}`, { error });
      outputRenderer.addErrorMessage(`Error loading file: ${error.message}`);
    }
  }
  
  /**
   * Save conversation to a file
   * 
   * @param {string} filePath Path to the file
   */
  async function saveConversation(filePath) {
    if (!filePath) {
      outputRenderer.addErrorMessage('No file specified');
      return;
    }
    
    try {
      // Resolve path
      const fullPath = path.resolve(path.join(screen.cwd, filePath));
      
      // Get conversation history
      const history = agent.getConversationHistory();
      
      // Format conversation
      const formatted = history.map(msg => `${msg.role.toUpperCase()}: ${msg.content}`).join('\n\n');
      
      // Write to file
      const fs = require('fs').promises;
      await fs.writeFile(fullPath, formatted, 'utf8');
      
      outputRenderer.addSystemMessage(`Conversation saved to: ${filePath}`);
    } catch (error) {
      logger.error(`Failed to save conversation: ${filePath}`, { error });
      outputRenderer.addErrorMessage(`Error saving conversation: ${error.message}`);
    }
  }
  
  // Initialize immediately
  init();
  
  // Return the input handler interface
  return {
    processInput,
    processCommand,
    handleFileModifications
  };
}
```

# src/tui/output.js

```js
/**
 * Output Renderer for TUI
 * 
 * Handles rendering of messages in the conversation panel
 */

const chalk = require('chalk');
const { logger } = require('../utils');
const { formatTimestamp } = require('../utils/formatters');

/**
 * Create an output renderer
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.widget The output widget
 * @param {Object} options.tokenMonitor Token monitoring utility
 * @returns {Object} The output renderer interface
 */
function createOutputRenderer({ widget, tokenMonitor }) {
  /**
   * Add a user message to the conversation
   * 
   * @param {string} message Message text
   */
  function addUserMessage(message) {
    const timestamp = formatTimestamp(new Date());
    
    // Format and add the message
    widget.log(chalk.bold.green(`YOU [${timestamp}]:`));
    widget.log(message);
    widget.log(''); // Empty line for spacing
    
    // Render the screen
    widget.screen.render();
  }
  
  /**
   * Add an assistant message to the conversation
   * 
   * @param {string} message Message text
   */
  function addAssistantMessage(message) {
    const timestamp = formatTimestamp(new Date());
    
    // Format and add the message
    widget.log(chalk.bold.blue(`FRANKCODE [${timestamp}]:`));
    
    // Process message for code blocks
    const processedMessage = processMessageForFormatting(message);
    
    // Add each line
    processedMessage.forEach(part => {
      if (part.type === 'text') {
        widget.log(part.content);
      } else if (part.type === 'code') {
        addCodeBlock(part.content, part.language);
      }
    });
    
    widget.log(''); // Empty line for spacing
    
    // Render the screen
    widget.screen.render();
  }
  
  /**
   * Add a system message to the conversation
   * 
   * @param {string} message Message text
   */
  function addSystemMessage(message) {
    // Format and add the message
    widget.log(chalk.bold.yellow('SYSTEM:'));
    widget.log(chalk.yellow(message));
    widget.log(''); // Empty line for spacing
    
    // Render the screen
    widget.screen.render();
  }
  
  /**
   * Add an error message to the conversation
   * 
   * @param {string} message Error message
   */
  function addErrorMessage(message) {
    // Format and add the message
    widget.log(chalk.bold.red('ERROR:'));
    widget.log(chalk.red(message));
    widget.log(''); // Empty line for spacing
    
    // Render the screen
    widget.screen.render();
  }
  
  /**
   * Add a code block to the conversation
   * 
   * @param {string} code Code text
   * @param {string} language Programming language
   */
  function addCodeBlock(code, language = '') {
    // Format the code block
    widget.log(chalk.bold(`\`\`\`${language}`));
    
    // Add the code with line numbers
    const lines = code.split('\n');
    lines.forEach((line, index) => {
      // Add line numbers for longer code blocks
      if (lines.length > 5) {
        const lineNum = String(index + 1).padStart(3, ' ');
        widget.log(chalk.dim(`${lineNum} |`) + ' ' + line);
      } else {
        widget.log(line);
      }
    });
    
    widget.log(chalk.bold('\`\`\`'));
    widget.log(''); // Empty line for spacing
    
    // Render the screen
    widget.screen.render();
  }
  
  /**
   * Process a message to extract code blocks
   * 
   * @param {string} message Message text
   * @returns {Array<Object>} Processed message parts
   */
  function processMessageForFormatting(message) {
    const parts = [];
    const codeBlockRegex = /\`\`\`(\w*)\n([\s\S]*?)\`\`\`/g;
    
    let lastIndex = 0;
    let match;
    
    // Find all code blocks
    while ((match = codeBlockRegex.exec(message)) !== null) {
      // Add text before the code block
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: message.slice(lastIndex, match.index)
        });
      }
      
      // Add the code block
      parts.push({
        type: 'code',
        language: match[1],
        content: match[2]
      });
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (lastIndex < message.length) {
      parts.push({
        type: 'text',
        content: message.slice(lastIndex)
      });
    }
    
    // If no code blocks were found, return the whole message as text
    if (parts.length === 0) {
      parts.push({
        type: 'text',
        content: message
      });
    }
    
    return parts;
  }
  
  /**
   * Process file modification code blocks
   * 
   * @param {string} message Message text
   * @returns {Array<Object>} File modifications
   */
  function processFileModifications(message) {
    const modifications = [];
    const fileBlockRegex = /\`\`\`file:(.*?)\n([\s\S]*?)\`\`\`/g;
    
    let match;
    while ((match = fileBlockRegex.exec(message)) !== null) {
      const filePath = match[1].trim();
      const content = match[2];
      
      modifications.push({
        filePath,
        content
      });
    }
    
    return modifications;
  }
  
  /**
   * Clear the conversation
   */
  function clear() {
    widget.setContent('');
    widget.screen.render();
  }
  
  // Return the output renderer interface
  return {
    addUserMessage,
    addAssistantMessage,
    addSystemMessage,
    addErrorMessage,
    addCodeBlock,
    processFileModifications,
    clear
  };
}

module.exports = {
  createOutputRenderer
};
```

# src/tui/statusBar.js

```js
/**
 * Status Bar Component
 * 
 * Displays information about the current state of the application
 */

const chalk = require('chalk');
const { logger } = require('../utils');

/**
 * Create a status bar component
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.widget The status bar widget
 * @param {Object} options.apiClient API client
 * @param {Object} options.tokenMonitor Token monitoring utility
 * @returns {Object} The status bar interface
 */
function createStatusBar({ widget, apiClient, tokenMonitor }) {
  // Status message
  let statusMessage = 'Ready';
  
  // Update interval handle
  let updateInterval = null;
  
  /**
   * Initialize the status bar
   */
  function init() {
    // Update immediately
    updateStatus();
    
    // Set up update interval (every 5 seconds)
    updateInterval = setInterval(() => {
      updateStatus();
    }, 5000);
  }
  
  /**
   * Update the status bar
   */
  function updateStatus() {
    try {
      // Get connection status
      const connected = apiClient.getConnectionStatus();
      
      // Get token usage
      const tokenUsage = tokenMonitor.getCurrentUsage();
      const tokenMax = tokenMonitor.getMaxTokens();
      const tokenPercentage = Math.floor((tokenUsage / tokenMax) * 100);
      
      // Format the token meter
      const tokenMeter = formatTokenMeter(tokenUsage, tokenMax);
      
      // Create status bar content
      const content = [
        // Left side
        `${chalk.bold('Status:')} ${statusMessage}`,
        
        // Center
        `Model: ${apiClient.getConnectionStatus() ? chalk.green('Connected') : chalk.red('Disconnected')}`,
        
        // Right side
        `Tokens: ${tokenMeter} ${tokenUsage}/${tokenMax} (${tokenPercentage}%)`
      ].join('    ');
      
      // Set widget content
      widget.setContent(content);
      
      // Check if token usage is nearing limit
      if (tokenPercentage > 80) {
        widget.style.bg = 'red';
      } else if (tokenPercentage > 60) {
        widget.style.bg = 'yellow';
      } else {
        widget.style.bg = 'blue';
      }
      
      // Render the screen
      widget.screen.render();
    } catch (error) {
      logger.error('Failed to update status bar', { error });
    }
  }
  
  /**
   * Format a token usage meter
   * 
   * @param {number} used Tokens used
   * @param {number} max Maximum tokens
   * @returns {string} Formatted meter
   */
  function formatTokenMeter(used, max) {
    const width = 10;
    const filled = Math.floor((used / max) * width);
    const empty = width - filled;
    
    let color = chalk.green;
    if (used / max > 0.8) {
      color = chalk.red;
    } else if (used / max > 0.6) {
      color = chalk.yellow;
    }
    
    const meter = color('[' + '='.repeat(filled) + ' '.repeat(empty) + ']');
    
    return meter;
  }
  
  /**
   * Update the status message
   * 
   * @param {string} message New status message
   */
  function update(message) {
    statusMessage = message;
    updateStatus();
  }
  
  /**
   * Destroy the status bar component
   */
  function destroy() {
    if (updateInterval) {
      clearInterval(updateInterval);
      updateInterval = null;
    }
  }
  
  // Initialize immediately
  init();
  
  // Return the status bar interface
  return {
    update,
    updateStatus,
    destroy
  };
}

module.exports = {
  createStatusBar
};
```

# src/tui/themes.js

```js
/**
 * Terminal UI Themes
 * 
 * Defines color themes for the TUI
 */

const chalk = require('chalk');

// Theme definitions
const themes = {
  dark: {
    bg: 'black',
    fg: 'white',
    border: {
      fg: 'blue',
      bg: 'black'
    },
    focus: {
      border: {
        fg: 'green',
        bg: 'black'
      }
    },
    label: {
      fg: 'white',
      bg: 'blue'
    },
    scrollbar: {
      bg: 'blue'
    }
  },
  light: {
    bg: 'white',
    fg: 'black',
    border: {
      fg: 'blue',
      bg: 'white'
    },
    focus: {
      border: {
        fg: 'green',
        bg: 'white'
      }
    },
    label: {
      fg: 'black',
      bg: 'cyan'
    },
    scrollbar: {
      bg: 'cyan'
    }
  },
  dracula: {
    bg: '#282a36',
    fg: '#f8f8f2',
    border: {
      fg: '#bd93f9',
      bg: '#282a36'
    },
    focus: {
      border: {
        fg: '#50fa7b',
        bg: '#282a36'
      }
    },
    label: {
      fg: '#f8f8f2',
      bg: '#6272a4'
    },
    scrollbar: {
      bg: '#bd93f9'
    }
  },
  solarized: {
    bg: '#002b36',
    fg: '#839496',
    border: {
      fg: '#2aa198',
      bg: '#002b36'
    },
    focus: {
      border: {
        fg: '#cb4b16',
        bg: '#002b36'
      }
    },
    label: {
      fg: '#fdf6e3',
      bg: '#073642'
    },
    scrollbar: {
      bg: '#586e75'
    }
  },
  nord: {
    bg: '#2e3440',
    fg: '#d8dee9',
    border: {
      fg: '#88c0d0',
      bg: '#2e3440'
    },
    focus: {
      border: {
        fg: '#a3be8c',
        bg: '#2e3440'
      }
    },
    label: {
      fg: '#eceff4',
      bg: '#3b4252'
    },
    scrollbar: {
      bg: '#5e81ac'
    }
  }
};

/**
 * Apply a theme to the screen
 * 
 * @param {Object} screen Blessed screen
 * @param {string} themeName Theme name
 */
function applyTheme(screen, themeName = 'dark') {
  // Get the theme
  const theme = themes[themeName] || themes.dark;
  
  // Apply to screen
  screen.style = {
    bg: theme.bg,
    fg: theme.fg
  };
  
  // Apply to all elements
  applyThemeToElements(screen, theme);
}

/**
 * Apply theme to all screen elements recursively
 * 
 * @param {Object} element Blessed element
 * @param {Object} theme Theme object
 */
function applyThemeToElements(element, theme) {
  // Set element styles
  if (element.style) {
    element.style.bg = theme.bg;
    element.style.fg = theme.fg;
    
    if (element.style.border) {
      element.style.border.fg = theme.border.fg;
      element.style.border.bg = theme.border.bg;
    }
    
    if (element.style.focus) {
      element.style.focus.border = {
        fg: theme.focus.border.fg,
        bg: theme.focus.border.bg
      };
    }
    
    if (element.style.scrollbar) {
      element.style.scrollbar.bg = theme.scrollbar.bg;
    }
    
    if (element.style.label) {
      element.style.label.fg = theme.label.fg;
      element.style.label.bg = theme.label.bg;
    }
  }
  
  // Apply to children
  if (element.children) {
    element.children.forEach(child => {
      applyThemeToElements(child, theme);
    });
  }
}

/**
 * Get available theme names
 * 
 * @returns {Array<string>} Theme names
 */
function getAvailableThemes() {
  return Object.keys(themes);
}

module.exports = {
  applyTheme,
  getAvailableThemes,
  themes
};
```

# src/utils/config.js

```js
/**
 * Configuration Utilities
 * 
 * Loads and manages application configuration
 */

const path = require('path');
const fs = require('fs');
const { cosmiconfig } = require('cosmiconfig');
const dotenv = require('dotenv');
const { logger } = require('./logger');

// Default configuration
const defaultConfig = {
  // LLM connection settings
  llm: {
    api: 'ollama',           // 'ollama' or 'distributed'
    coordinatorHost: 'localhost',
    coordinatorPort: 11434,  // Default Ollama port
    model: 'llama2',
    temperature: 0.7
  },
  
  // Project context settings
  context: {
    maxTokens: 8192,
    excludeDirs: ['node_modules', 'dist', '.git', 'build', 'vendor'],
    excludeFiles: ['*.lock', '*.log', '*.min.js', '*.min.css', '*.bin', '*.exe'],
    priorityFiles: ['README.md', 'package.json', 'composer.json', 'requirements.txt', 'Makefile']
  },
  
  // UI preferences
  ui: {
    theme: 'dark',
    layout: 'default',
    fontSize: 14
  },
  
  // Logging settings
  logging: {
    level: 'info',
    console: true,
    file: true
  }
};

/**
 * Load configuration
 * 
 * @param {Object} options Configuration options
 * @param {string} options.configPath Path to configuration file
 * @returns {Promise<Object>} Loaded configuration
 */
async function loadConfig(options = {}) {
  try {
    // Load .env file if it exists
    dotenv.config();
    
    // Load configuration using cosmiconfig
    const explorer = cosmiconfig('frankcode', {
      searchPlaces: [
        '.frankcoderc',
        '.frankcoderc.json',
        '.frankcoderc.yaml',
        '.frankcoderc.yml',
        '.frankcoderc.js',
        'frankcode.config.js',
        'package.json'
      ]
    });
    
    // Try to load the configuration
    let result;
    
    if (options.configPath) {
      // Load from specified path
      result = await explorer.load(options.configPath);
    } else {
      // Search for configuration
      result = await explorer.search();
    }
    
    // Get configuration object
    const config = result ? result.config : {};
    
    // Merge with default configuration
    const mergedConfig = mergeConfigs(defaultConfig, config);
    
    // Override with environment variables
    applyEnvironmentVariables(mergedConfig);
    
    // Override with command line options
    applyCommandLineOptions(mergedConfig, options);
    
    // Set project root
    mergedConfig.projectRoot = options.projectRoot || process.cwd();
    
    return mergedConfig;
  } catch (error) {
    logger.error('Failed to load configuration', { error });
    
    // Return default configuration
    return {
      ...defaultConfig,
      projectRoot: options.projectRoot || process.cwd()
    };
  }
}

/**
 * Merge configurations
 * 
 * @param {Object} target Target configuration
 * @param {Object} source Source configuration
 * @returns {Object} Merged configuration
 */
function mergeConfigs(target, source) {
  const result = { ...target };
  
  for (const key in source) {
    if (source[key] === null || source[key] === undefined) {
      continue;
    }
    
    if (typeof source[key] === 'object' && !Array.isArray(source[key])) {
      result[key] = mergeConfigs(result[key] || {}, source[key]);
    } else {
      result[key] = source[key];
    }
  }
  
  return result;
}

/**
 * Apply environment variables to configuration
 * 
 * @param {Object} config Configuration object
 */
function applyEnvironmentVariables(config) {
  // LLM configuration
  if (process.env.FRANKCODE_LLM_API) {
    config.llm.api = process.env.FRANKCODE_LLM_API;
  }
  
  if (process.env.FRANKCODE_LLM_HOST) {
    config.llm.coordinatorHost = process.env.FRANKCODE_LLM_HOST;
  }
  
  if (process.env.FRANKCODE_LLM_PORT) {
    config.llm.coordinatorPort = parseInt(process.env.FRANKCODE_LLM_PORT, 10);
  }
  
  if (process.env.FRANKCODE_LLM_MODEL) {
    config.llm.model = process.env.FRANKCODE_LLM_MODEL;
  }
  
  if (process.env.FRANKCODE_LLM_TEMPERATURE) {
    config.llm.temperature = parseFloat(process.env.FRANKCODE_LLM_TEMPERATURE);
  }
  
  // Context configuration
  if (process.env.FRANKCODE_CONTEXT_MAX_TOKENS) {
    config.context.maxTokens = parseInt(process.env.FRANKCODE_CONTEXT_MAX_TOKENS, 10);
  }
  
  // UI configuration
  if (process.env.FRANKCODE_UI_THEME) {
    config.ui.theme = process.env.FRANKCODE_UI_THEME;
  }
  
  // Logging configuration
  if (process.env.FRANKCODE_LOG_LEVEL) {
    config.logging.level = process.env.FRANKCODE_LOG_LEVEL;
  }
}

/**
 * Apply command line options to configuration
 * 
 * @param {Object} config Configuration object
 * @param {Object} options Command line options
 */
function applyCommandLineOptions(config, options) {
  // LLM configuration
  if (options.api) {
    config.llm.api = options.api;
  }
  
  if (options.host) {
    config.llm.coordinatorHost = options.host;
  }
  
  if (options.port) {
    config.llm.coordinatorPort = parseInt(options.port, 10);
  }
  
  if (options.model) {
    config.llm.model = options.model;
  }
  
  if (options.temperature !== undefined) {
    config.llm.temperature = parseFloat(options.temperature);
  }
  
  // Context configuration
  if (options.maxTokens) {
    config.context.maxTokens = parseInt(options.maxTokens, 10);
  }
  
  // UI configuration
  if (options.theme) {
    config.ui.theme = options.theme;
  }
  
  // Logging configuration
  if (options.verbose) {
    config.logging.level = 'debug';
  } else if (options.quiet) {
    config.logging.level = 'error';
  } else if (options.logLevel) {
    config.logging.level = options.logLevel;
  }
}

/**
 * Save configuration to file
 * 
 * @param {Object} config Configuration object
 * @param {string} filePath File path
 * @returns {Promise<boolean>} Success indicator
 */
async function saveConfig(config, filePath) {
  try {
    // Determine file format based on extension
    const ext = path.extname(filePath).toLowerCase();
    let content;
    
    if (ext === '.json') {
      content = JSON.stringify(config, null, 2);
    } else if (ext === '.js') {
      content = `module.exports = ${JSON.stringify(config, null, 2)};`;
    } else {
      // Default to JSON
      content = JSON.stringify(config, null, 2);
    }
    
    // Write to file
    await fs.promises.writeFile(filePath, content, 'utf8');
    
    logger.info(`Configuration saved to ${filePath}`);
    return true;
  } catch (error) {
    logger.error(`Failed to save configuration to ${filePath}`, { error });
    return false;
  }
}

module.exports = {
  loadConfig,
  saveConfig,
  defaultConfig
};
```

# src/utils/fileSystem.js

```js
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
```

# src/utils/formatters.js

```js
/**
 * Formatting utilities
 * 
 * Functions for formatting data in the UI
 */

/**
 * Format a timestamp
 * 
 * @param {Date} date Date object
 * @returns {string} Formatted timestamp
 */
function formatTimestamp(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  
  /**
   * Format a file size
   * 
   * @param {number} bytes Size in bytes
   * @param {number} decimals Decimal places
   * @returns {string} Formatted size
   */
  function formatFileSize(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }
  
  /**
   * Format a duration
   * 
   * @param {number} ms Duration in milliseconds
   * @returns {string} Formatted duration
   */
  function formatDuration(ms) {
    if (ms < 1000) {
      return `${ms}ms`;
    }
    
    const seconds = Math.floor(ms / 1000);
    
    if (seconds < 60) {
      return `${seconds}s`;
    }
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes < 60) {
      return `${minutes}m ${remainingSeconds}s`;
    }
    
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    
    return `${hours}h ${remainingMinutes}m`;
  }
  
  /**
   * Format a percentage
   * 
   * @param {number} value Value
   * @param {number} total Total
   * @param {number} decimals Decimal places
   * @returns {string} Formatted percentage
   */
  function formatPercentage(value, total, decimals = 1) {
    if (total === 0) return '0%';
    
    const percentage = (value / total) * 100;
    return percentage.toFixed(decimals) + '%';
  }
  
  /**
   * Truncate a string
   * 
   * @param {string} str String to truncate
   * @param {number} length Maximum length
   * @param {string} suffix Suffix to add
   * @returns {string} Truncated string
   */
  function truncate(str, length = 50, suffix = '...') {
    if (!str) return '';
    if (str.length <= length) return str;
    
    return str.substring(0, length - suffix.length) + suffix;
  }
  
  /**
   * Format a code snippet for display
   * 
   * @param {string} code Code snippet
   * @param {number} maxLines Maximum lines
   * @returns {string} Formatted code
   */
  function formatCodeSnippet(code, maxLines = 10) {
    if (!code) return '';
    
    const lines = code.split('\n');
    
    if (lines.length <= maxLines) {
      return code;
    }
    
    return lines.slice(0, maxLines).join('\n') + `\n... (${lines.length - maxLines} more lines)`;
  }
  
  module.exports = {
    formatTimestamp,
    formatFileSize,
    formatDuration,
    formatPercentage,
    truncate,
    formatCodeSnippet
  };
```

# src/utils/git.js

```js
/**
 * Git utilities
 * 
 * Provides Git operations for the application
 */

const simpleGit = require('simple-git');
const { logger } = require('./logger');

/**
 * Create a Git utility instance
 * 
 * @param {string} repositoryPath Repository path
 * @returns {Object} Git utility interface
 */
function createGitUtils(repositoryPath) {
  // Create simple-git instance
  const git = simpleGit(repositoryPath);
  
  /**
   * Check if the directory is a Git repository
   * 
   * @returns {Promise<boolean>} Whether it's a Git repository
   */
  async function isGitRepository() {
    try {
      return await git.checkIsRepo();
    } catch (error) {
      logger.error('Failed to check if directory is a Git repository', { error });
      return false;
    }
  }
  
  /**
   * Get the current branch
   * 
   * @returns {Promise<string>} Current branch name
   */
  async function getCurrentBranch() {
    try {
      return await git.revparse(['--abbrev-ref', 'HEAD']);
    } catch (error) {
      logger.error('Failed to get current branch', { error });
      return '';
    }
  }
  
  /**
   * Get Git status
   * 
   * @returns {Promise<Object>} Git status
   */
  async function getStatus() {
    try {
      return await git.status();
    } catch (error) {
      logger.error('Failed to get Git status', { error });
      return null;
    }
  }
  
  /**
   * Get repository info
   * 
   * @returns {Promise<Object>} Repository info
   */
  async function getRepositoryInfo() {
    try {
      // Check if it's a Git repository
      const isRepo = await isGitRepository();
      
      if (!isRepo) {
        return { isGitRepository: false };
      }
      
      // Get information
      const [branch, status, remotes] = await Promise.all([
        getCurrentBranch(),
        getStatus(),
        git.getRemotes(true)
      ]);
      
      // Get current remote URL (usually origin)
      let remoteUrl = '';
      
      if (remotes.length > 0) {
        const origin = remotes.find(r => r.name === 'origin') || remotes[0];
        remoteUrl = origin.refs.fetch || '';
      }
      
      return {
        isGitRepository: true,
        branch,
        status,
        remoteUrl,
        modified: status ? status.modified : [],
        untracked: status ? status.not_added : [],
        staged: status ? status.staged : []
      };
    } catch (error) {
      logger.error('Failed to get repository info', { error });
      return { isGitRepository: false };
    }
  }
  
  /**
   * Get file history
   * 
   * @param {string} filePath File path
   * @param {number} maxEntries Maximum number of entries
   * @returns {Promise<Array<Object>>} File history
   */
  async function getFileHistory(filePath, maxEntries = 10) {
    try {
      // Get log for the file
      const log = await git.log({
        file: filePath,
        maxCount: maxEntries
      });
      
      return log.all.map(entry => ({
        hash: entry.hash,
        date: entry.date,
        message: entry.message,
        author: entry.author_name
      }));
    } catch (error) {
      logger.error(`Failed to get file history: ${filePath}`, { error });
      return [];
    }
  }
  
  /**
   * Get file diff
   * 
   * @param {string} filePath File path
   * @returns {Promise<string>} File diff
   */
  async function getFileDiff(filePath) {
    try {
      // Get diff for the file
      return await git.diff(['--', filePath]);
    } catch (error) {
      logger.error(`Failed to get file diff: ${filePath}`, { error });
      return '';
    }
  }
  
  /**
   * Get modified files with their diffs
   * 
   * @returns {Promise<Array<Object>>} Modified files with diffs
   */
  async function getModifiedFilesWithDiffs() {
    try {
      // Get status
      const status = await getStatus();
      
      if (!status) {
        return [];
      }
      
      // Get modified files
      const modifiedFiles = [...status.modified, ...status.not_added];
      
      // Get diffs for each file
      const result = [];
      
      for (const file of modifiedFiles) {
        try {
          const diff = await getFileDiff(file);
          
          result.push({
            path: file,
            diff
          });
        } catch (error) {
          logger.error(`Failed to get diff for file: ${file}`, { error });
        }
      }
      
      return result;
    } catch (error) {
      logger.error('Failed to get modified files with diffs', { error });
      return [];
    }
  }
  
  /**
   * Get the content of a file at a specific revision
   * 
   * @param {string} filePath File path
   * @param {string} revision Revision (default is HEAD)
   * @returns {Promise<string>} File content
   */
  async function getFileAtRevision(filePath, revision = 'HEAD') {
    try {
      return await git.show([`${revision}:${filePath}`]);
    } catch (error) {
      logger.error(`Failed to get file at revision: ${filePath}@${revision}`, { error });
      return '';
    }
  }
  
  /**
   * Add a file to the staging area
   * 
   * @param {string} filePath File path
   * @returns {Promise<boolean>} Success indicator
   */
  async function addFile(filePath) {
    try {
      await git.add(filePath);
      return true;
    } catch (error) {
      logger.error(`Failed to add file: ${filePath}`, { error });
      return false;
    }
  }
  
  /**
   * Commit changes
   * 
   * @param {string} message Commit message
   * @returns {Promise<boolean>} Success indicator
   */
  async function commit(message) {
    try {
      await git.commit(message);
      return true;
    } catch (error) {
      logger.error(`Failed to commit: ${message}`, { error });
      return false;
    }
  }
  
  /**
   * Push changes to the remote
   * 
   * @param {string} remote Remote name
   * @param {string} branch Branch name
   * @returns {Promise<boolean>} Success indicator
   */
  async function push(remote = 'origin', branch = '') {
    try {
      await git.push(remote, branch);
      return true;
    } catch (error) {
      logger.error(`Failed to push to ${remote}/${branch}`, { error });
      return false;
    }
  }
  
  // Return the Git utility interface
  return {
    isGitRepository,
    getCurrentBranch,
    getStatus,
    getRepositoryInfo,
    getFileHistory,
    getFileDiff,
    getModifiedFilesWithDiffs,
    getFileAtRevision,
    addFile,
    commit,
    push
  };
}

module.exports = {
  createGitUtils
};
```

# src/utils/index.js

```js
/**
 * Utilities module entry point
 * 
 * Exports utility functions and objects
 */

const { logger, setupLogging } = require('./logger');
const { loadConfig, saveConfig } = require('./config');
const { scanProjectFiles, ensureDir, pathExists, getStats, getFileType } = require('./fileSystem');
const { createGitUtils } = require('./git');
const { formatTimestamp, formatFileSize, formatDuration, formatPercentage, truncate } = require('./formatters');
const { createTokenMonitor } = require('./tokenMonitor');

module.exports = {
  logger,
  setupLogging,
  loadConfig,
  saveConfig,
  scanProjectFiles,
  ensureDir,
  pathExists,
  getStats,
  getFileType,
  createGitUtils,
  formatTimestamp,
  formatFileSize,
  formatDuration,
  formatPercentage,
  truncate,
  createTokenMonitor
};
```

# src/utils/logger.js

```js
/**
 * Logger module
 * 
 * Provides logging capabilities for the application
 */

const winston = require('winston');
const fs = require('fs');
const path = require('path');

// Custom log levels
const levels = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3
};

// Create logs directory if it doesn't exist
const logsDir = path.join(process.cwd(), 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir);
}

// Create winston logger
const logger = winston.createLogger({
  levels,
  format: winston.format.combine(
    winston.format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    winston.format.errors({ stack: true }),
    winston.format.splat(),
    winston.format.json()
  ),
  defaultMeta: { service: 'frankcode' },
  transports: [
    // Write logs with level 'error' and below to error.log
    new winston.transports.File({
      filename: path.join(logsDir, 'error.log'),
      level: 'error'
    }),
    
    // Write logs with level 'info' and below to combined.log
    new winston.transports.File({
      filename: path.join(logsDir, 'combined.log')
    })
  ]
});

/**
 * Set up logging based on configuration
 * 
 * @param {Object} config Logging configuration
 */
function setupLogging(config = {}) {
  // Default configuration
  const {
    level = 'info',
    console = true,
    file = true,
    filePath,
    maxSize = '10m',
    maxFiles = 5
  } = config;
  
  // Set log level
  logger.level = level;
  
  // Add console transport if enabled
  if (console) {
    logger.add(new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }));
  }
  
  // Configure file transports if enabled
  if (file) {
    // Remove existing file transports
    logger.transports = logger.transports.filter(
      t => t.name !== 'file'
    );
    
    // Custom log path or default
    const logPath = filePath || logsDir;
    
    // Ensure log directory exists
    if (!fs.existsSync(logPath)) {
      fs.mkdirSync(logPath, { recursive: true });
    }
    
    // Add configured file transports
    logger.add(new winston.transports.File({
      filename: path.join(logPath, 'error.log'),
      level: 'error',
      maxsize: maxSize,
      maxFiles
    }));
    
    logger.add(new winston.transports.File({
      filename: path.join(logPath, 'combined.log'),
      maxsize: maxSize,
      maxFiles
    }));
  }
  
  logger.info('Logging initialized', { level });
}
```

# src/utils/tokenMonitor.js

```js
/**
 * Token Monitor
 * 
 * Tracks token usage for context management
 */

const { logger } = require('./logger');
const { EventEmitter } = require('events');

/**
 * Create a token monitor
 * 
 * @param {Object} options Configuration options
 * @param {number} options.maxTokens Maximum tokens
 * @param {number} options.warningThreshold Warning threshold (0-1)
 * @returns {Object} The token monitor interface
 */
function createTokenMonitor(options = {}) {
  const {
    maxTokens = 8192,
    warningThreshold = 0.8
  } = options;
  
  // Current token usage
  let currentTokens = 0;
  
  // Create event emitter
  const events = new EventEmitter();
  
  /**
   * Update token usage
   * 
   * @param {number} tokens Number of tokens
   */
  function updateUsage(tokens) {
    // Calculate new total
    const newTotal = currentTokens + tokens;
    
    // Check if we're going over the limit
    if (newTotal > maxTokens) {
      logger.warn(`Token limit exceeded: ${newTotal}/${maxTokens}`);
      events.emit('limit-exceeded', { current: newTotal, max: maxTokens });
    } else if (newTotal / maxTokens >= warningThreshold && currentTokens / maxTokens < warningThreshold) {
      // Emit warning if we cross the threshold
      logger.warn(`Token usage approaching limit: ${newTotal}/${maxTokens}`);
      events.emit('warning-threshold', { current: newTotal, max: maxTokens });
    }
    
    // Update current tokens
    currentTokens = newTotal;
    
    // Emit update event
    events.emit('usage-updated', { current: currentTokens, max: maxTokens });
  }
  
  /**
   * Set token usage directly
   * 
   * @param {number} tokens Number of tokens
   */
  function setUsage(tokens) {
    // Check if we're going over the limit
    if (tokens > maxTokens) {
      logger.warn(`Token limit exceeded: ${tokens}/${maxTokens}`);
      events.emit('limit-exceeded', { current: tokens, max: maxTokens });
    } else if (tokens / maxTokens >= warningThreshold && currentTokens / maxTokens < warningThreshold) {
      // Emit warning if we cross the threshold
      logger.warn(`Token usage approaching limit: ${tokens}/${maxTokens}`);
      events.emit('warning-threshold', { current: tokens, max: maxTokens });
    }
    
    // Update current tokens
    currentTokens = tokens;
    
    // Emit update event
    events.emit('usage-updated', { current: currentTokens, max: maxTokens });
  }
  
  /**
   * Reset token usage
   */
  function reset() {
    currentTokens = 0;
    events.emit('usage-updated', { current: currentTokens, max: maxTokens });
    logger.debug('Token usage reset');
  }
  
  /**
   * Get current token usage
   * 
   * @returns {number} Current token usage
   */
  function getCurrentUsage() {
    return currentTokens;
  }
  
  /**
   * Get maximum tokens
   * 
   * @returns {number} Maximum tokens
   */
  function getMaxTokens() {
    return maxTokens;
  }
  
  /**
   * Get usage ratio
   * 
   * @returns {number} Usage ratio (0-1)
   */
  function getUsageRatio() {
    return currentTokens / maxTokens;
  }
  
  /**
   * Check if nearing limit
   * 
   * @returns {boolean} Whether nearing limit
   */
  function isNearingLimit() {
    return getUsageRatio() >= warningThreshold;
  }
  
  /**
   * Add an event listener
   * 
   * @param {string} event Event name
   * @param {Function} listener Event listener
   */
  function on(event, listener) {
    events.on(event, listener);
  }
  
  /**
   * Remove an event listener
   * 
   * @param {string} event Event name
   * @param {Function} listener Event listener
   */
  function off(event, listener) {
    events.off(event, listener);
  }
  
  // Return the token monitor interface
  return {
    updateUsage,
    setUsage,
    reset,
    getCurrentUsage,
    getMaxTokens,
    getUsageRatio,
    isNearingLimit,
    on,
    off
  };
}

module.exports = {
  createTokenMonitor
};
```

