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

# conversation.txt

```txt
USER:
hello

---

ASSISTANT:
<think>
Alright, so the user just sent me a message asking for help. Let's see, I'm supposed to act as FrankCode, an AI assistant for developers. Okay, first thing that comes to mind is that they're probably starting fresh with their project and need some guidance.

Looking at their current context, they have a JSON file in the apps directory but no functions or classes. They just sent two lines of text, "hello" twice. Hmm, maybe they want me to help them write something for this JSON file? But since I can't modify that file directly, perhaps I should suggest adding code that connects it to their project.

I remember there's a tool called tsconfig.json used in Node.js projects. Maybe they need to set up some configuration files like tsconfig.json and tsconfig.jsonb (for TypeScript). That way, my assistant can generate the necessary configuration for running this file as a server or something else.

Also, if I were part of their codebase, they might have specific modules or imports that are missing. I should consider advising them on how to check out or modify any existing dependencies in their project and ensure all required files are included.

I wonder if there's more context from their end beyond the initial messages. Maybe they're encountering a problem that can't be resolved with my general advice, so it's worth asking them about what they need help with specifically.
</think>

To assist you effectively, could you clarify or provide specific details about your project and any issues you're facing?


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
    "name": "frankcode",
    "version": "0.1.0",
    "description": "Terminal-based coding agent powered by distributed LLMs",
    "main": "src/index.js",
    "bin": {
        "frankcode": "./bin/frankcode.js"
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
        "cors": "^2.8.5",
        "cosmiconfig": "^8.0.0",
        "diff": "^5.1.0",
        "dotenv": "^16.4.7",
        "express": "^4.21.2",
        "fast-glob": "^3.2.12",
        "ignore": "^5.2.4",
        "isomorphic-git": "^1.21.0",
        "keytar": "^7.9.0",
        "ngrok": "^5.0.0-beta.2",
        "node-fetch": "^2.7.0",
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

# scripts/setup-embeddings.js

```js
#!/usr/bin/env node

/**
 * Setup Script for Embeddings Dependencies
 * 
 * This script installs the necessary Python dependencies for 
 * generating embeddings with Jina Embeddings.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Colors for output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
};

// Log with color
function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

// Check if Python is installed
function checkPython() {
  try {
    const version = execSync('python --version').toString().trim();
    log(`Found ${version}`, colors.green);
    return true;
  } catch (error) {
    try {
      const version = execSync('python3 --version').toString().trim();
      log(`Found ${version}`, colors.green);
      return true;
    } catch (error) {
      log('Python not found. Please install Python 3.8 or higher.', colors.red);
      return false;
    }
  }
}

// Check if pip is installed
function checkPip() {
  try {
    const version = execSync('pip --version').toString().trim();
    log(`Found pip: ${version}`, colors.green);
    return true;
  } catch (error) {
    try {
      const version = execSync('pip3 --version').toString().trim();
      log(`Found pip3: ${version}`, colors.green);
      return true;
    } catch (error) {
      log('pip not found. Please install pip.', colors.red);
      return false;
    }
  }
}

// Install Python dependencies
function installDependencies() {
  log('Installing Python dependencies...', colors.cyan);
  
  try {
    let cmd = 'pip install sentence-transformers torch numpy';
    
    // Use pip3 on some systems
    if (execSync('pip --version').toString().indexOf('python 2') !== -1) {
      cmd = 'pip3 install sentence-transformers torch numpy';
    }
    
    log(`Running: ${cmd}`, colors.yellow);
    execSync(cmd, { stdio: 'inherit' });
    
    log('Dependencies installed successfully!', colors.green);
    return true;
  } catch (error) {
    log(`Failed to install dependencies: ${error.message}`, colors.red);
    return false;
  }
}

// Create Python script directory and ensure script exists
function createScriptDir() {
  try {
    const scriptsDir = path.join(__dirname, '../src/utils');
    
    if (!fs.existsSync(scriptsDir)) {
      fs.mkdirSync(scriptsDir, { recursive: true });
      log(`Created scripts directory: ${scriptsDir}`, colors.green);
    }
    
    // Copy the embeddings.py script to the utils directory if it doesn't exist
    const targetScript = path.join(scriptsDir, 'embeddings.py');
    
    if (!fs.existsSync(targetScript)) {
      log('embeddings.py script not found. Creating it...', colors.yellow);
      
      // Create a simple default script if it doesn't exist
      // This is a placeholder - the actual script should be created separately
      const defaultScript = `#!/usr/bin/env python3
"""
Embeddings Generator using Jina Embeddings v3 or other Hugging Face models.
"""
import argparse
import json
import sys

print("Please place the actual embeddings.py script in src/utils/")
sys.exit(1)
`;
      
      fs.writeFileSync(targetScript, defaultScript);
      fs.chmodSync(targetScript, 0o755); // Make executable
      
      log('Created placeholder embeddings.py script.', colors.yellow);
      log('Please replace it with the actual implementation.', colors.yellow);
    } else {
      log('embeddings.py script found.', colors.green);
      // Make sure it's executable
      fs.chmodSync(targetScript, 0o755);
    }
    
    return true;
  } catch (error) {
    log(`Failed to create scripts directory: ${error.message}`, colors.red);
    return false;
  }
}

// Create temp directory for embedding processing
function createTempDir() {
  try {
    const tempDir = path.join(__dirname, '../temp');
    
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
      log(`Created temp directory: ${tempDir}`, colors.green);
    } else {
      log(`Temp directory already exists: ${tempDir}`, colors.green);
    }
    
    return true;
  } catch (error) {
    log(`Failed to create temp directory: ${error.message}`, colors.red);
    return false;
  }
}

// Main function
function main() {
  log('Setting up embedding dependencies for FrankCode...', colors.cyan);
  
  const pythonOk = checkPython();
  if (!pythonOk) {
    log('Python check failed. Please install Python 3.8 or higher.', colors.red);
    return false;
  }
  
  const pipOk = checkPip();
  if (!pipOk) {
    log('pip check failed. Please install pip.', colors.red);
    return false;
  }
  
  const depsOk = installDependencies();
  if (!depsOk) {
    log('Failed to install dependencies.', colors.red);
    return false;
  }
  
  const scriptDirOk = createScriptDir();
  if (!scriptDirOk) {
    log('Failed to create scripts directory.', colors.red);
    return false;
  }
  
  const tempDirOk = createTempDir();
  if (!tempDirOk) {
    log('Failed to create temp directory.', colors.red);
    return false;
  }
  
  log('\nSetup complete! You can now use Jina Embeddings with FrankCode.', colors.green);
  log('\nIf you encounter any issues with embeddings, check the following:', colors.cyan);
  log('1. Make sure Python and pip are properly installed', colors.cyan);
  log('2. Make sure the embeddings.py script is in src/utils/', colors.cyan);
  log('3. Make sure the temp directory exists and is writable', colors.cyan);
  
  return true;
}

// Run main function
if (main()) {
  process.exit(0);
} else {
  process.exit(1);
}
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
   * Add system context to the conversation
   * 
   * @param {string} context System context to add
   */
  function addSystemContext(context) {
    try {
      // Add a system message to the conversation history
      conversationHistory.push({ 
        role: 'system', 
        content: context 
      });
      
      logger.debug('Added system context to conversation');
    } catch (error) {
      logger.error('Failed to add system context', { error });
    }
  }
  // Update the reset function to optionally preserve system context
  /**
   * Reset the agent state
   * 
   * @param {boolean} preserveSystemContext Whether to preserve system context messages
   */
  function reset(preserveSystemContext = false) {
    try {
      if (preserveSystemContext) {
        // Filter out non-system messages
        const systemMessages = conversationHistory.filter(msg => msg.role === 'system');
        conversationHistory = systemMessages;
      } else {
        conversationHistory = [];
      }
      
      contextManager.reset();
      logger.debug('Agent state reset');
    } catch (error) {
      logger.error('Failed to reset agent state', { error });
      // Ensure we at least clear the conversation if there's an error
      conversationHistory = [];
      contextManager.reset();
    }
  }
  
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
    addSystemContext,  // Add this
    reset,  // This should replace the existing reset function
    setApiClient: (newClient) => {
      apiClient = newClient;
    }
  };
}

module.exports = {
  createAgent
};
```

# src/agent/agentEnhancer.js

```js
/**
 * AutoGen Integration for FrankCode
 * 
 * This module provides integration with AutoGen concepts to enhance agent capabilities.
 */

const { logger } = require('../utils/logger');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);
const path = require('path');
const fs = require('fs').promises;

/**
 * Create an AutoGen-inspired agent enhancer
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.agent Agent instance
 * @param {Object} options.outputRenderer Output renderer
 * @param {Object} options.apiClient API client
 * @returns {Object} The agent enhancer interface
 */
function createAgentEnhancer({ agent, outputRenderer, apiClient }) {
  /**
   * Add planning capabilities to break tasks into steps
   * 
   * @param {string} task The task to plan
   * @returns {Promise<Array<Object>>} The planned steps
   */
  async function planTask(task) {
    try {
      outputRenderer.addSystemMessage(`🧠 Planning task: ${task}`);
      
      const prompt = `You are FrankCode, an AI coding assistant that helps with complex programming tasks.
I need you to break down the following task into a clear step-by-step plan:

TASK: ${task}

Your task is to break this down into smaller steps. For each step:
1. Specify the type of action required (read_file, search_files, update_file, create_file, generate_code, analyze)
2. Provide a clear description of what needs to be done
3. List any specific files that need to be examined or modified
4. Explain the reasoning for this step

FORMAT YOUR RESPONSE AS:
STEP 1:
Type: [action_type]
Description: [clear description]
Files: [file paths, if relevant]
Reasoning: [brief explanation]

STEP 2:
...

Ensure your plan is comprehensive yet concise. Identify 3-7 key steps to make the task manageable.`;
      
      const response = await apiClient.generateResponse(prompt, {
        temperature: 0.3,
        maxTokens: 1024
      });
      
      const steps = parseSteps(response.text);
      
      // Display the plan
      outputRenderer.addSystemMessage(`📋 Task Plan (${steps.length} steps):`);
      
      steps.forEach((step, index) => {
        outputRenderer.addSystemMessage(`${index + 1}. ${step.type.toUpperCase()}: ${step.description}`);
      });
      
      return steps;
    } catch (error) {
      logger.error(`Failed to plan task: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Error planning task: ${error.message}`);
      return [];
    }
  }
  
  /**
   * Parse the steps from the LLM response
   */
  function parseSteps(response) {
    const steps = [];
    const stepRegex = /STEP\s+(\d+):\s*\n([\s\S]*?)(?=STEP\s+\d+:|$)/g;
    
    let match;
    while ((match = stepRegex.exec(response)) !== null) {
      const stepContent = match[2].trim();
      
      // Parse step details
      const typeMatch = /Type:\s*(\w+)/.exec(stepContent);
      const descMatch = /Description:\s*(.+?)(?=\n\w+:|$)/s.exec(stepContent);
      const filesMatch = /Files:\s*(.+?)(?=\n\w+:|$)/s.exec(stepContent);
      const reasoningMatch = /Reasoning:\s*(.+?)(?=\n\w+:|$)/s.exec(stepContent);
      
      if (typeMatch && descMatch) {
        steps.push({
          type: typeMatch[1].toLowerCase(),
          description: descMatch[1].trim(),
          files: filesMatch ? 
            filesMatch[1].trim().split(/,\s*/).filter(f => f !== 'N/A' && f !== '') : 
            [],
          reasoning: reasoningMatch ? reasoningMatch[1].trim() : ''
        });
      }
    }
    
    // If no steps were parsed but we have content, create a fallback step
    if (steps.length === 0 && response.trim().length > 0) {
      steps.push({
        type: 'analyze',
        description: 'Analyze the task and provide guidance',
        files: [],
        reasoning: 'No structured plan could be created, falling back to general analysis'
      });
    }
    
    return steps;
  }
  
  /**
   * Execute a task with planning and step-by-step execution
   * 
   * @param {string} task The task to execute
   * @returns {Promise<Object>} The execution result
   */
  async function executeTask(task) {
    try {
      // Plan the task
      const steps = await planTask(task);
      
      if (steps.length === 0) {
        outputRenderer.addErrorMessage('Failed to create a plan for this task.');
        return { status: 'error', error: 'Failed to create plan' };
      }
      
      // Execute each step
      for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        
        // Display current step
        outputRenderer.addSystemMessage(`\n▶️ Step ${i + 1}/${steps.length}: ${step.description}`);
        
        // Generate a prompt for this step
        const prompt = `You are FrankCode, an AI coding assistant.
I need your help with this step in a larger task:

TASK: ${task}
STEP: ${step.description}
${step.files && step.files.length > 0 ? `FILES: ${step.files.join(', ')}` : ''}
${step.reasoning ? `CONTEXT: ${step.reasoning}` : ''}

Please provide a detailed and helpful response to complete this step.`;
        
        // Execute the step using the agent
        const response = await agent.processMessage(prompt);
        
        // Display assistant response (this should happen automatically in processMessage)
        // Check for file modifications if any
        if (response && response.fileModifications && response.fileModifications.length > 0) {
          // Handle file modifications - these will be handled separately
        }
        
        // Give user a chance to review/cancel
        if (i < steps.length - 1) {
          outputRenderer.addSystemMessage('\nPress Enter to continue to the next step, or type "cancel" to stop.');
        }
      }
      
      // Task completed successfully
      outputRenderer.addSystemMessage(`\n✅ Task completed successfully!`);
      
      return { 
        status: 'success',
        steps: steps.length
      };
    } catch (error) {
      logger.error(`Failed to execute task: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Error executing task: ${error.message}`);
      
      return { 
        status: 'error', 
        error: error.message
      };
    }
  }
  
  /**
   * Create a file in the specified location
   * 
   * @param {string} filePath File path
   * @param {string} content File content
   * @returns {Promise<boolean>} Success indicator
   */
  async function createFile(filePath, content) {
    try {
      // Get absolute path
      const fullPath = path.isAbsolute(filePath) ? filePath : path.join(process.cwd(), filePath);
      
      // Ensure directory exists
      const dirPath = path.dirname(fullPath);
      await fs.mkdir(dirPath, { recursive: true }).catch(() => {});
      
      // Write file
      await fs.writeFile(fullPath, content, 'utf8');
      
      outputRenderer.addSystemMessage(`✅ Created file: ${filePath}`);
      return true;
    } catch (error) {
      logger.error(`Failed to create file: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Error creating file: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Execute code in a safe environment
   * 
   * @param {string} code The code to execute
   * @param {string} language The language (python, javascript, etc.)
   * @returns {Promise<Object>} The execution result
   */
  async function executeCode(code, language = 'python') {
    try {
      // Create a temporary file
      const extension = language === 'python' ? '.py' : 
                       language === 'javascript' ? '.js' : 
                       language === 'bash' ? '.sh' : '.txt';
      
      const tempFile = `temp_${Date.now()}${extension}`;
      await fs.writeFile(tempFile, code, 'utf8');
      
      // Execute the code
      let command;
      switch(language) {
        case 'python':
          command = `python ${tempFile}`;
          break;
        case 'javascript':
          command = `node ${tempFile}`;
          break;
        case 'bash':
          command = `bash ${tempFile}`;
          break;
        default:
          throw new Error(`Unsupported language: ${language}`);
      }
      
      outputRenderer.addSystemMessage(`⚙️ Executing ${language} code...`);
      
      const { stdout, stderr } = await execPromise(command);
      
      // Clean up
      await fs.unlink(tempFile).catch(() => {});
      
      // Display result
      if (stdout) {
        outputRenderer.addSystemMessage('🔹 Output:');
        outputRenderer.addCodeBlock(stdout, 'bash');
      }
      
      if (stderr) {
        outputRenderer.addErrorMessage('Error output:');
        outputRenderer.addCodeBlock(stderr, 'bash');
      }
      
      return { 
        success: !stderr, 
        stdout, 
        stderr 
      };
    } catch (error) {
      logger.error(`Failed to execute code: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Error executing code: ${error.message}`);
      
      // Try to clean up
      try {
        await fs.unlink(tempFile).catch(() => {});
      } catch {}
      
      return { 
        success: false, 
        error: error.message 
      };
    }
  }
  
  // Return the agent enhancer interface
  return {
    planTask,
    executeTask,
    createFile,
    executeCode
  };
}

module.exports = {
  createAgentEnhancer
};
```

# src/agent/agentUtils.js

```js
/**
 * Agent Utilities
 * 
 * Helper functions for integrating agent capabilities with the FrankCode TUI
 */

const { logger } = require('../utils/logger');
const { createAgentRunner } = require('./runner');

/**
 * Create an agent command processor
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.agent Agent instance
 * @param {Object} options.llm LLM client
 * @param {Object} options.screen Blessed screen object
 * @param {Object} options.outputRenderer Output renderer
 * @returns {Object} Command processor interface
 */
function createAgentCommandProcessor({ agent, llm, screen, outputRenderer }) {
  // Create agent runner
  const runner = createAgentRunner({ 
    llm, 
    screen, 
    outputRenderer,
    agent
  });

  // Command patterns
  const taskCommandPattern = /^(implement|create|build|add|refactor|fix|optimize)\s+(.+)$/i;
  const analyzeCommandPattern = /^(analyze|examine|check|review)\s+(.+)$/i;
  const explainCommandPattern = /^(explain|how|why)\s+(.+)$/i;
  
  /**
   * Process a user command and detect if it's an agent task
   * 
   * @param {string} command User command
   * @returns {Promise<boolean>} Whether the command was handled
   */
  async function processCommand(command) {
    try {
      // Normalize command
      const normalizedCommand = command.trim();
      
      // Check if this is a task command
      const taskMatch = taskCommandPattern.exec(normalizedCommand);
      if (taskMatch) {
        const verb = taskMatch[1].toLowerCase();
        const task = taskMatch[2];
        
        // Execute the task
        await runner.executeTask(`${verb} ${task}`);
        return true;
      }
      
      // Check if this is an analyze command
      const analyzeMatch = analyzeCommandPattern.exec(normalizedCommand);
      if (analyzeMatch) {
        const verb = analyzeMatch[1].toLowerCase();
        const target = analyzeMatch[2];
        
        // Execute analysis action
        await runner.executeAction({
          type: runner.ActionType.ANALYZE,
          description: `${verb} ${target}`,
          reasoning: 'User requested analysis'
        });
        return true;
      }
      
      // Check if this is an explain command
      const explainMatch = explainCommandPattern.exec(normalizedCommand);
      if (explainMatch) {
        const verb = explainMatch[1].toLowerCase();
        const question = explainMatch[2];
        
        // Execute question answering action
        await runner.executeAction({
          type: runner.ActionType.ANSWER_QUESTION,
          description: `${verb} ${question}`,
          reasoning: 'User asked a question'
        });
        return true;
      }
      
      // Check for explicit file operations
      if (normalizedCommand.startsWith('read ')) {
        const filePath = normalizedCommand.substring(5).trim();
        await runner.executeAction({
          type: runner.ActionType.READ_FILE,
          description: `Read the file ${filePath}`,
          files: [filePath]
        });
        return true;
      }
      
      if (normalizedCommand.startsWith('find ')) {
        const pattern = normalizedCommand.substring(5).trim();
        await runner.executeAction({
          type: runner.ActionType.SEARCH_FILES,
          description: `Find files matching ${pattern}`,
          files: [pattern]
        });
        return true;
      }
      
      // Not an agent command
      return false;
    } catch (error) {
      logger.error('Error processing agent command', { error });
      outputRenderer.addErrorMessage(`Error: ${error.message}`);
      return true; // Mark as handled to prevent further processing
    }
  }
  
  /**
   * Check if a command looks like it might be a task for the agent
   * 
   * @param {string} command User command
   * @returns {boolean} Whether it looks like an agent task
   */
  function isPotentialAgentTask(command) {
    const normalizedCommand = command.trim().toLowerCase();
    
    // Check against command patterns
    if (taskCommandPattern.test(normalizedCommand)) return true;
    if (analyzeCommandPattern.test(normalizedCommand)) return true;
    if (explainCommandPattern.test(normalizedCommand)) return true;
    
    // Check for explicit file operations
    if (normalizedCommand.startsWith('read ')) return true;
    if (normalizedCommand.startsWith('find ')) return true;
    
    // Look for programming-related terms that might indicate a task
    const programmingTerms = [
      'function', 'component', 'class', 'method',
      'module', 'interface', 'feature', 'bug',
      'performance', 'memory', 'leak', 'api',
      'endpoint', 'database', 'schema', 'model',
      'route', 'controller', 'view', 'template',
      'test', 'unit test', 'integration test',
      'authentication', 'authorization', 'security',
      'deploy', 'pipeline', 'build', 'docker',
      'caching', 'logging', 'error handling'
    ];
    
    return programmingTerms.some(term => normalizedCommand.includes(term));
  }
  
  /**
   * Suggest using agent capabilities
   * 
   * @param {string} command User command
   * @returns {string|null} Suggestion or null if none
   */
  function suggestAgentCapability(command) {
    const normalizedCommand = command.trim().toLowerCase();
    
    // Detect questions
    if (normalizedCommand.includes('how do i') || 
        normalizedCommand.includes('how to') ||
        normalizedCommand.includes('what is') ||
        normalizedCommand.includes('why does')) {
      return "It looks like you're asking a question. Try starting with 'explain' or 'how' for detailed explanations.";
    }
    
    // Detect potential tasks
    if (normalizedCommand.length > 15 && isPotentialAgentTask(normalizedCommand)) {
      return "This seems like a task I could help with. Try using task verbs like 'implement', 'create', 'fix', or 'refactor'.";
    }
    
    return null;
  }
  
  /**
   * Get a list of example agent commands
   * 
   * @returns {Array<string>} Example commands
   */
  function getExampleCommands() {
    return [
      "implement a function to sort users by creation date",
      "create a new React component for displaying user profiles",
      "fix the memory leak in the connection manager",
      "refactor the authentication module to use JWT",
      "analyze the performance bottlenecks in the data processing pipeline",
      "explain how the routing system works in this codebase",
      "read src/utils/formatters.js",
      "find all files related to authentication"
    ];
  }
  
  /**
   * Execute a specific agent task
   * 
   * @param {string} task Task description
   * @returns {Promise<Object>} Task result
   */
  async function executeTask(task) {
    return await runner.executeTask(task);
  }
  
  // Return the command processor interface
  return {
    processCommand,
    isPotentialAgentTask,
    suggestAgentCapability,
    getExampleCommands,
    executeTask,
    runner
  };
}

module.exports = {
  createAgentCommandProcessor
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

# src/agent/conversationManager.js

```js
/**
 * Improved Conversation Management Module for FrankCode
 * 
 * Provides utilities for managing conversation history,
 * including clearing, compacting, and summarizing.
 */

const { logger } = require('../utils/logger');
const fs = require('fs').promises;
const path = require('path');

/**
 * Create a conversation manager
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.agent Agent instance
 * @param {Object} options.outputRenderer Output renderer
 * @param {Object} options.llm LLM client
 * @returns {Object} The conversation manager interface
 */
function createConversationManager({ agent, outputRenderer, llm }) {
  /**
   * Clear the conversation history
   * 
   * @returns {Promise<void>}
   */
  async function clearConversation() {
    try {
      // Reset the agent's conversation history
      agent.reset();
      
      // Clear the output renderer
      outputRenderer.clear();
      
      // Display confirmation message
      outputRenderer.addSystemMessage('Conversation history cleared.');
      
      logger.info('Conversation history cleared by user');
    } catch (error) {
      logger.error('Failed to clear conversation', { error });
      outputRenderer.addErrorMessage(`Error clearing conversation: ${error.message}`);
    }
  }
  
  /**
   * Compact the conversation by summarizing and clearing history
   * 
   * @returns {Promise<void>}
   */
  async function compactConversation() {
    try {
      // Get the current conversation history
      const history = agent.getConversationHistory();
      
      if (!history || history.length === 0) {
        outputRenderer.addSystemMessage('No conversation history to compact.');
        return;
      }
      
      outputRenderer.addSystemMessage('Compacting conversation...');
      
      // Generate a summary of the conversation
      const summary = await generateConversationSummary(history);
      
      // Reset the agent's conversation history but preserve system messages
      agent.reset(true);
      
      // Add the summary as system context
      agent.addSystemContext(summary);
      
      // Clear the output display
      outputRenderer.clear();
      
      // Display the summary with proper formatting
      outputRenderer.addSystemMessage('***Session Summary***');
      
      // Split the summary into lines and add with proper formatting
      const summaryLines = summary.split('\n');
      for (const line of summaryLines) {
        outputRenderer.addSystemMessage(line);
      }
      
      outputRenderer.addSystemMessage('\nConversation history has been compacted. The summary above has been retained in context.');
      
      logger.info('Conversation compacted with summary');
    } catch (error) {
      logger.error('Failed to compact conversation', { error });
      outputRenderer.addErrorMessage(`Error compacting conversation: ${error.message}`);
    }
  }
  
  /**
   * Generate a summary of the conversation history
   * 
   * @param {Array<Object>} history Conversation history
   * @returns {Promise<string>} Summary text
   */
  async function generateConversationSummary(history) {
    try {
      if (!history || history.length === 0) {
        return "No conversation history to summarize.";
      }
      
      // Filter out system messages and thinking tags
      const filteredHistory = history.filter(msg => 
        msg.role !== 'system'
      ).map(msg => ({
        role: msg.role,
        content: msg.content.replace(/<think>[\s\S]*?<\/think>/g, '') // Remove thinking tags
      }));
      
      if (filteredHistory.length === 0) {
        return "No user-assistant interactions to summarize.";
      }
      
      // Format the conversation history
      const formattedHistory = filteredHistory.map(msg => 
        `${msg.role.toUpperCase()}: ${msg.content.substring(0, 500)}${msg.content.length > 500 ? '...' : ''}`
      ).join('\n\n');
      
      // Create a prompt for the LLM to generate a summary
      const prompt = `I need you to create a structured summary of the following conversation between a user and FrankCode, an AI coding assistant. Format the summary like this example:

We've been working on a chemical process simulation application, focusing on fixing display and calculation issues in the heat exchanger components. Here's what we've accomplished:

**Main Issues Fixed:**

1. **Heat Exchanger Duty Display**
  - Fixed an issue where the heat exchanger duty wasn't updating correctly
  - Added forced numeric conversion to ensure duty values display correctly
  - Enhanced logging to trace calculation steps and verify values

2. **Property Display Improvements**
  - Added "Show Properties on Flowsheet" toggle in component configuration modals
  - Reverted flow rate formatting to maintain readability for large values
  - Fixed precision issues in numeric displays

**Main Files Modified:**

1. **/src/lib/simulation.ts**
  - Enhanced calculateHeatDuty() with better logging
  - Fixed the HeatExchangerNode.calculate() method
  - Added deep object cloning to avoid reference issues

**Next Steps:**

1. Test heat exchanger calculations with different flow rates
2. Verify connection configuration modal works as expected
3. Look into other components that may need similar display improvements

The most critical fixes were ensuring that changing boundary inflow properly propagates through the simulation.

Here's the conversation to summarize:

${formattedHistory}

Create a similar summary that:
1. Identifies the main topics and tasks discussed
2. Lists specific problems solved or features implemented 
3. Mentions key files or components that were modified or discussed
4. Suggests logical next steps based on the conversation
5. Highlights the most important achievements or insights

Format with clear sections using **bold** for headers and bullet points for details. Keep the summary concise but comprehensive.`;
      
      // Generate the summary using the LLM
      const response = await llm.generateResponse(prompt, {
        temperature: 0.3,  // Lower temperature for more factual summaries
        maxTokens: 1024    // Allow enough tokens for a detailed summary
      });
      
      return response.text;
    } catch (error) {
      logger.error('Failed to generate conversation summary', { error });
      return 'Failed to generate summary. The conversation history has still been cleared.';
    }
  }
  
  /**
   * Save the conversation to a file
   * 
   * @param {string} filePath File path
   * @returns {Promise<boolean>} Success indicator
   */
  async function saveConversation(filePath) {
    try {
      // Get the conversation history
      const history = agent.getConversationHistory();
      
      if (!history || history.length === 0) {
        outputRenderer.addSystemMessage('No conversation history to save.');
        return false;
      }
      
      // Format the conversation for saving
      const formattedHistory = history.map(msg => {
        // Remove thinking tags for cleaner export
        const cleanContent = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
        return `${msg.role.toUpperCase()}:\n${cleanContent}\n\n---\n\n`;
      }).join('');
      
      // Ensure directory exists
      const dirPath = path.dirname(filePath);
      await fs.mkdir(dirPath, { recursive: true }).catch(() => {});
      
      // Write to file
      await fs.writeFile(filePath, formattedHistory, 'utf8');
      
      outputRenderer.addSystemMessage(`Conversation saved to ${filePath}`);
      logger.info(`Conversation saved to ${filePath}`);
      
      return true;
    } catch (error) {
      logger.error('Failed to save conversation', { error });
      outputRenderer.addErrorMessage(`Error saving conversation: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Export conversation with summary
   * 
   * @param {string} filePath File path
   * @returns {Promise<boolean>} Success indicator
   */
  async function exportConversationWithSummary(filePath) {
    try {
      // Get the conversation history
      const history = agent.getConversationHistory();
      
      if (!history || history.length === 0) {
        outputRenderer.addSystemMessage('No conversation history to export.');
        return false;
      }
      
      // Generate a summary
      const summary = await generateConversationSummary(history);
      
      // Format the conversation for saving (remove thinking tags)
      const formattedHistory = history.map(msg => {
        const cleanContent = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
        return `${msg.role.toUpperCase()}:\n${cleanContent}\n\n---\n\n`;
      }).join('');
      
      // Combine summary and full history
      const content = `# Conversation Summary\n\n${summary}\n\n# Full Conversation\n\n${formattedHistory}`;
      
      // Ensure directory exists
      const dirPath = path.dirname(filePath);
      await fs.mkdir(dirPath, { recursive: true }).catch(() => {});
      
      // Write to file
      await fs.writeFile(filePath, content, 'utf8');
      
      outputRenderer.addSystemMessage(`Conversation with summary exported to ${filePath}`);
      logger.info(`Conversation exported to ${filePath}`);
      
      return true;
    } catch (error) {
      logger.error('Failed to export conversation', { error });
      outputRenderer.addErrorMessage(`Error exporting conversation: ${error.message}`);
      return false;
    }
  }
  
  // Return the conversation manager interface
  return {
    clearConversation,
    compactConversation,
    saveConversation,
    exportConversationWithSummary,
    generateConversationSummary
  };
}

module.exports = {
  createConversationManager
};
```

# src/agent/enhanced/embeddings.js

```js
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
```

# src/agent/enhanced/enhanced-agent-controller.js

```js
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
```

# src/agent/enhanced/enhanced-agent.js

```js
/**
 * Enhanced Agent for FrankCode
 * 
 * This module integrates various AI capabilities (RAG, GraphRAG, Memory)
 * to create a more capable autonomous code editing agent.
 */

const path = require('path');
const fs = require('fs').promises;
const { createCodeRAG } = require('./rag');
const { createGraphRAG } = require('./graphrag');
const { createMemoryController } = require('./memory-controller');
const { createVectorStore } = require('./embeddings');
const { logger } = require('../utils/logger');
const { countTokens } = require('./tokenizer');

/**
 * Create an enhanced agent for code editing
 * 
 * @param {Object} options Configuration options
 * @param {string} options.projectRoot Project root directory
 * @param {Object} options.apiClient API client for LLM
 * @param {Object} options.tokenMonitor Token usage monitor
 * @returns {Object} The enhanced agent interface
 */
function createEnhancedAgent({ projectRoot, apiClient, tokenMonitor }) {
  // Set up RAG systems
  const storageDir = path.join(projectRoot, '.frankcode');
  
  // Create RAG systems
  const codeVectorStore = createVectorStore({
    storePath: path.join(storageDir, 'code-vectors.json')
  });
  
  const codeRAG = createCodeRAG({
    projectRoot,
    vectorStore: codeVectorStore
  });
  
  const graphRAG = createGraphRAG({
    projectRoot,
    vectorStore: codeVectorStore
  });
  
  // Create memory controller
  const memoryController = createMemoryController({
    memoryPath: path.join(storageDir, 'memory.json'),
    maxMemories: 1000,
    relevanceThreshold: 0.7
  });
  
  // Track indexing status
  let isIndexed = false;
  let isIndexing = false;
  
  /**
   * Initialize the enhanced agent
   * 
   * @returns {Promise<void>}
   */
  async function initialize() {
    try {
      // Create storage directory
      await fs.mkdir(storageDir, { recursive: true });
      
      // Try to load existing GraphRAG data
      const graphLoaded = await graphRAG.loadGraph(path.join(storageDir, 'code-graph.json'));
      
      if (graphLoaded) {
        isIndexed = true;
        logger.info('GraphRAG data loaded from disk');
      }
    } catch (error) {
      logger.error(`Failed to initialize enhanced agent: ${error.message}`, { error });
    }
  }
  
  /**
   * Index the project for RAG and GraphRAG
   * 
   * @param {Object} options Indexing options
   * @returns {Promise<boolean>} Success indicator
   */
  async function indexProject(options = {}) {
    if (isIndexing) {
      logger.warn('Project indexing already in progress');
      return false;
    }
    
    try {
      isIndexing = true;
      
      // Index with CodeRAG
      logger.info('Indexing project with CodeRAG...');
      const codeRAGCount = await codeRAG.indexProject(options);
      
      // Index with GraphRAG
      logger.info('Indexing project with GraphRAG...');
      const graphRAGCount = await graphRAG.indexProject(options);
      
      isIndexed = true;
      isIndexing = false;
      
      logger.info(`Indexing complete. Indexed ${codeRAGCount} files with CodeRAG and ${graphRAGCount} files with GraphRAG`);
      return true;
    } catch (error) {
      isIndexing = false;
      logger.error(`Failed to index project: ${error.message}`, { error });
      return false;
    }
  }
  
  /**
   * Process a user message with context augmentation
   * 
   * @param {string} message User's message
   * @returns {Promise<Object>} Response object
   */
  async function processMessage(message) {
    try {
      // Check if we need to index the project
      if (!isIndexed && !isIndexing && !message.startsWith('/')) {
        logger.info('Project not indexed. Starting background indexing...');
        // Start indexing in the background
        indexProject().catch(error => {
          logger.error(`Background indexing failed: ${error.message}`, { error });
        });
      }
      
      // Extract the task from the message
      const task = extractTask(message);
      
      // Skip context augmentation for commands
      if (message.startsWith('/')) {
        return { 
          text: message,
          augmented: false
        };
      }
      
      // Get relevant memories
      logger.debug('Retrieving relevant memories...');
      const relevantMemories = await memoryController.getRelevantMemories(task);
      
      // Get relevant code context
      logger.debug('Retrieving relevant code context...');
      const codeContext = isIndexed ? 
        await codeRAG.retrieveContext(task, { maxResults: 3 }) : 
        [];
      
      // Get relevant graph context if task involves code relationships
      logger.debug('Retrieving relevant graph context...');
      const graphContext = isIndexed ? 
        await graphRAG.search(task, { maxResults: 3 }) : 
        [];
      
      // Format contexts
      const memoriesText = memoryController.formatMemoriesForPrompt(relevantMemories);
      const codeContextText = codeRAG.formatContextForPrompt(codeContext);
      const graphContextText = graphRAG.formatSearchResultsForPrompt(graphContext);
      
      // Combine contexts
      let augmentedContext = '';
      
      if (memoriesText) augmentedContext += memoriesText + '\n\n';
      if (codeContextText) augmentedContext += codeContextText + '\n\n';
      if (graphContextText) augmentedContext += graphContextText + '\n\n';
      
      // Truncate to avoid token limits
      if (augmentedContext.length > 0) {
        const maxContextTokens = 3000; // Should be adjusted based on model
        const tokenCount = countTokens(augmentedContext);
        
        if (tokenCount > maxContextTokens) {
          logger.debug(`Truncating context from ${tokenCount} to ${maxContextTokens} tokens`);
          augmentedContext = truncateToTokens(augmentedContext, maxContextTokens);
        }
        
        // Add the augmented context to the message
        augmentedContext += `\n\nUser message: ${message}`;
        
        logger.debug('Message augmented with context');
        return {
          text: augmentedContext,
          originalMessage: message,
          augmented: true
        };
      }
      
      // No context to add
      return {
        text: message,
        augmented: false
      };
    } catch (error) {
      logger.error(`Failed to process message: ${error.message}`, { error });
      return {
        text: message,
        error: error.message,
        augmented: false
      };
    }
  }
  
  /**
   * Extract a task from a message
   * 
   * @param {string} message User message
   * @returns {string} Extracted task
   */
  function extractTask(message) {
    // For now, just return the message as is
    // In a more sophisticated implementation, this could extract
    // the core task or intent from the message
    return message;
  }
  
  /**
   * Learn from user feedback
   * 
   * @param {string} context The context or task
   * @param {string} feedback User's feedback
   * @param {Object} metadata Additional metadata
   * @returns {Promise<boolean>} Success indicator
   */
  async function learnFromFeedback(context, feedback, metadata = {}) {
    try {
      // Determine memory type based on context
      let memoryType = '';
      
      if (context.toLowerCase().includes('code pattern') || 
          context.toLowerCase().includes('structure') ||
          context.toLowerCase().includes('format')) {
        await memoryController.rememberCodePattern(context, feedback, metadata);
      } else if (context.toLowerCase().includes('prefer') ||
                 context.toLowerCase().includes('like') ||
                 context.toLowerCase().includes('want')) {
        await memoryController.rememberUserPreference(context, feedback, metadata);
      } else {
        await memoryController.rememberProblemSolution(context, feedback, metadata);
      }
      
      logger.info('Learned from user feedback');
      return true;
    } catch (error) {
      logger.error(`Failed to learn from feedback: ${error.message}`, { error });
      return false;
    }
  }
  
  /**
   * Generate an LLM prompt with appropriate context
   * 
   * @param {string} task The task description
   * @param {Object} options Prompt options
   * @returns {Promise<string>} Generated prompt
   */
  async function generatePrompt(task, options = {}) {
    try {
      const { 
        includeMemories = true,
        includeCodeContext = true,
        includeGraphContext = true,
        systemInstruction = null
      } = options;
      
      // Start with system instruction if provided
      let prompt = '';
      
      if (systemInstruction) {
        prompt += systemInstruction + '\n\n';
      } else {
        prompt += 'You are FrankCode, an autonomous code editing agent. You can help users write, modify, and understand code. Please follow these guidelines:\n\n';
        prompt += '1. When modifying files, clearly show the changes and ask for confirmation\n';
        prompt += '2. When creating new files, show the full content and ask for confirmation\n';
        prompt += '3. Use consistent code patterns that match the existing codebase\n';
        prompt += '4. Provide explanations for your changes\n';
        prompt += '5. Remember user preferences for future interactions\n\n';
      }
      
      // Add context
      let contextAdded = false;
      
      // Add memories if available and requested
      if (includeMemories) {
        const relevantMemories = await memoryController.getRelevantMemories(task);
        if (relevantMemories.length > 0) {
          prompt += memoryController.formatMemoriesForPrompt(relevantMemories) + '\n\n';
          contextAdded = true;
        }
      }
      
      // Add code context if available and requested
      if (includeCodeContext && isIndexed) {
        const codeContext = await codeRAG.retrieveContext(task, { maxResults: 3 });
        if (codeContext.length > 0) {
          prompt += codeRAG.formatContextForPrompt(codeContext) + '\n\n';
          contextAdded = true;
        }
      }
      
      // Add graph context if available and requested
      if (includeGraphContext && isIndexed) {
        const graphContext = await graphRAG.search(task, { maxResults: 3 });
        if (graphContext.length > 0) {
          prompt += graphRAG.formatSearchResultsForPrompt(graphContext) + '\n\n';
          contextAdded = true;
        }
      }
      
      // Add the task
      if (contextAdded) {
        prompt += 'Based on the context above, please address the following:\n\n';
      }
      
      prompt += task;
      
      return prompt;
    } catch (error) {
      logger.error(`Failed to generate prompt: ${error.message}`, { error });
      return task; // Fallback to just the task
    }
  }
  
  /**
   * Process a file modification request
   * 
   * @param {Object} modification File modification data
   * @param {string} modification.filePath Path to the file
   * @param {string} modification.content New content
   * @returns {Promise<Object>} Processing result
   */
  async function processFileModification(modification) {
    try {
      // Get original file content if it exists
      let originalContent = '';
      try {
        originalContent = await fs.readFile(path.join(projectRoot, modification.filePath), 'utf8');
      } catch (error) {
        // File doesn't exist, treat as a new file
        originalContent = '';
      }
      
      // Check if this is a pattern we should remember
      if (originalContent) {
        // This is an existing file being modified
        await memoryController.rememberCodePattern(
          `Code pattern for file: ${modification.filePath}`,
          originalContent,
          { filePath: modification.filePath, type: 'file_structure' }
        );
      }
      
      // Return the modification with additional context
      return {
        ...modification,
        isNewFile: !originalContent,
        originalContent
      };
    } catch (error) {
      logger.error(`Failed to process file modification: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * Truncate text to fit within token limit
   * 
   * @param {string} text Text to truncate
   * @param {number} maxTokens Maximum number of tokens
   * @returns {string} Truncated text
   */
  function truncateToTokens(text, maxTokens) {
    // Simple truncation by removing content from the middle
    const tokens = countTokens(text);
    
    if (tokens <= maxTokens) {
      return text;
    }
    
    // Split into lines
    const lines = text.split('\n');
    
    // If very few lines, just return the first part
    if (lines.length <= 5) {
      // Estimate tokens per character and truncate
      const tokensPerChar = tokens / text.length;
      const maxChars = Math.floor(maxTokens / tokensPerChar);
      return text.substring(0, maxChars) + '\n\n[Context truncated due to length]';
    }
    
    // Keep the beginning and end parts
    const keepRatio = maxTokens / tokens;
    const linesToKeepStart = Math.floor(lines.length * keepRatio * 0.6); // More from beginning
    const linesToKeepEnd = Math.floor(lines.length * keepRatio * 0.4); // Less from end
    
    const beginningLines = lines.slice(0, linesToKeepStart);
    const endLines = lines.slice(-linesToKeepEnd);
    
    return beginningLines.join('\n') + 
           '\n\n[... Context truncated due to length ...]\n\n' + 
           endLines.join('\n');
  }
  
  // Initialize
  initialize();
  
  // Return the enhanced agent interface
  return {
    processMessage,
    indexProject,
    learnFromFeedback,
    generatePrompt,
    processFileModification,
    codeRAG,
    graphRAG,
    memoryController
  };
}

module.exports = {
  createEnhancedAgent
};
```

# src/agent/enhanced/graphrag-module.js

```js
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
```

# src/agent/enhanced/index.js

```js
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
```

# src/agent/enhanced/memory-controller.js

```js
/**
 * Memory Controller for FrankCode
 * 
 * This module provides a way for FrankCode to learn from past experiences
 * and remember code patterns, solving approaches, and user preferences.
 */

const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const { createEmbedding, cosineSimilarity } = require('./embeddings');
const { logger } = require('../utils/logger');

/**
 * Create a Memory Controller
 * 
 * @param {Object} options Configuration options
 * @param {string} options.memoryPath Path to store memories
 * @param {number} options.maxMemories Maximum number of memories to keep
 * @param {number} options.relevanceThreshold Threshold for considering memories relevant
 * @returns {Object} The memory controller interface
 */
function createMemoryController({ memoryPath, maxMemories = 1000, relevanceThreshold = 0.7 }) {
  // Memory types
  const MemoryType = {
    CODE_PATTERN: 'code_pattern',     // Remembered code structures and patterns
    USER_PREFERENCE: 'user_preference', // User's preferred way of doing things
    PROBLEM_SOLUTION: 'problem_solution', // How a problem was solved
    FILE_STRUCTURE: 'file_structure',   // Structure of a particular file
    TASK_INSIGHT: 'task_insight',      // Insight for a particular task
    ERROR_RESOLUTION: 'error_resolution' // How an error was fixed
  };
  
  // In-memory store
  let memories = [];
  
  /**
   * Initialize the memory controller
   * 
   * @returns {Promise<void>}
   */
  async function initialize() {
    try {
      // Ensure directory exists
      await fs.mkdir(path.dirname(memoryPath), { recursive: true });
      
      // Check if memory file exists
      try {
        await fs.access(memoryPath);
        
        // Load memories
        const data = await fs.readFile(memoryPath, 'utf8');
        memories = JSON.parse(data);
        logger.info(`Loaded ${memories.length} memories`);
      } catch (error) {
        // File doesn't exist, start with empty memories
        memories = [];
        logger.info('Starting with empty memories');
      }
    } catch (error) {
      logger.error(`Failed to initialize memory controller: ${error.message}`, { error });
      memories = [];
    }
  }
  
  /**
   * Save the current memories to disk
   * 
   * @returns {Promise<boolean>} Success indicator
   */
  async function saveMemories() {
    try {
      // Ensure directory exists
      await fs.mkdir(path.dirname(memoryPath), { recursive: true });
      
      // Write to file
      await fs.writeFile(memoryPath, JSON.stringify(memories, null, 2), 'utf8');
      logger.debug(`Saved ${memories.length} memories to ${memoryPath}`);
      return true;
    } catch (error) {
      logger.error(`Failed to save memories: ${error.message}`, { error });
      return false;
    }
  }
  
  /**
   * Add a new memory
   * 
   * @param {Object} memory Memory object
   * @param {string} memory.type Memory type
   * @param {string} memory.task Description of the task or context
   * @param {string} memory.insight The insight or information to remember
   * @param {Object} memory.metadata Additional metadata
   * @returns {Promise<string>} Memory ID
   */
  async function addMemory(memory) {
    try {
      // Generate an ID for the memory
      const id = crypto.randomUUID();
      
      // Generate embeddings
      const taskEmbedding = await createEmbedding(memory.task);
      const insightEmbedding = await createEmbedding(memory.insight);
      
      // Create full memory object
      const newMemory = {
        id,
        ...memory,
        created: Date.now(),
        lastAccessed: Date.now(),
        accessCount: 0,
        taskEmbedding,
        insightEmbedding
      };
      
      // Add to memories
      memories.push(newMemory);
      
      // Limit size if needed
      if (memories.length > maxMemories) {
        // Remove oldest memory
        memories.sort((a, b) => a.lastAccessed - b.lastAccessed);
        memories.shift();
      }
      
      // Save memories
      await saveMemories();
      
      logger.info(`Added new ${memory.type} memory: ${id}`);
      return id;
    } catch (error) {
      logger.error(`Failed to add memory: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * Retrieve memories relevant to a task
   * 
   * @param {string} task Task description
   * @param {Object} options Query options
   * @param {Array<string>} options.types Memory types to include
   * @param {number} options.limit Maximum number of results
   * @param {number} options.threshold Relevance threshold
   * @returns {Promise<Array<Object>>} Relevant memories
   */
  async function getRelevantMemories(task, options = {}) {
    try {
      const {
        types = null,
        limit = 5,
        threshold = relevanceThreshold
      } = options;
      
      // Generate embedding for the task
      const taskEmbedding = await createEmbedding(task);
      
      // Find relevant memories
      const relevantMemories = [];
      
      for (const memory of memories) {
        // Filter by type if specified
        if (types && !types.includes(memory.type)) {
          continue;
        }
        
        // Calculate relevance score
        const relevance = cosineSimilarity(taskEmbedding, memory.taskEmbedding);
        
        // Add if relevant
        if (relevance >= threshold) {
          relevantMemories.push({
            ...memory,
            relevance,
            taskEmbedding: undefined, // Don't include embedding in results
            insightEmbedding: undefined
          });
        }
      }
      
      // Sort by relevance
      relevantMemories.sort((a, b) => b.relevance - a.relevance);
      
      // Update access info for retrieved memories
      for (const memory of relevantMemories.slice(0, limit)) {
        const original = memories.find(m => m.id === memory.id);
        if (original) {
          original.lastAccessed = Date.now();
          original.accessCount++;
        }
      }
      
      // Save memory access info
      await saveMemories();
      
      logger.debug(`Retrieved ${relevantMemories.length} relevant memories for task`);
      return relevantMemories.slice(0, limit);
    } catch (error) {
      logger.error(`Failed to retrieve relevant memories: ${error.message}`, { error });
      return [];
    }
  }
  
  /**
   * Remove a memory by ID
   * 
   * @param {string} id Memory ID
   * @returns {Promise<boolean>} Success indicator
   */
  async function removeMemory(id) {
    try {
      // Find memory index
      const index = memories.findIndex(m => m.id === id);
      
      if (index === -1) {
        logger.warn(`Memory not found: ${id}`);
        return false;
      }
      
      // Remove memory
      memories.splice(index, 1);
      
      // Save memories
      await saveMemories();
      
      logger.info(`Removed memory: ${id}`);
      return true;
    } catch (error) {
      logger.error(`Failed to remove memory: ${error.message}`, { error });
      return false;
    }
  }
  
  /**
   * Clear all memories
   * 
   * @returns {Promise<boolean>} Success indicator
   */
  async function clearMemories() {
    try {
      // Clear memories
      memories = [];
      
      // Save empty memories
      await saveMemories();
      
      logger.info('Cleared all memories');
      return true;
    } catch (error) {
      logger.error(`Failed to clear memories: ${error.message}`, { error });
      return false;
    }
  }
  
  /**
   * Format memories for use in a prompt
   * 
   * @param {Array<Object>} memories Memories to format
   * @returns {string} Formatted memories
   */
  function formatMemoriesForPrompt(memories) {
    if (!memories || memories.length === 0) {
      return '';
    }
    
    let formatted = '--- RELEVANT MEMORIES ---\n\n';
    
    for (const memory of memories) {
      formatted += `Type: ${memory.type}\n`;
      formatted += `Relevance: ${memory.relevance ? memory.relevance.toFixed(2) : 'N/A'}\n`;
      formatted += `Context: ${memory.task}\n`;
      formatted += `Insight: ${memory.insight}\n`;
      
      if (memory.metadata) {
        formatted += 'Metadata:\n';
        for (const [key, value] of Object.entries(memory.metadata)) {
          formatted += `- ${key}: ${value}\n`;
        }
      }
      
      formatted += '\n';
    }
    
    return formatted;
  }
  
  /**
   * Add a memory about a code pattern
   * 
   * @param {string} pattern Description of the pattern
   * @param {string} example Example of the pattern
   * @param {Object} metadata Additional metadata
   * @returns {Promise<string>} Memory ID
   */
  async function rememberCodePattern(pattern, example, metadata = {}) {
    return await addMemory({
      type: MemoryType.CODE_PATTERN,
      task: pattern,
      insight: example,
      metadata
    });
  }
  
  /**
   * Add a memory about a user preference
   * 
   * @param {string} context The context of the preference
   * @param {string} preference The user's preference
   * @param {Object} metadata Additional metadata
   * @returns {Promise<string>} Memory ID
   */
  async function rememberUserPreference(context, preference, metadata = {}) {
    return await addMemory({
      type: MemoryType.USER_PREFERENCE,
      task: context,
      insight: preference,
      metadata
    });
  }
  
  /**
   * Add a memory about a problem solution
   * 
   * @param {string} problem Description of the problem
   * @param {string} solution The solution
   * @param {Object} metadata Additional metadata
   * @returns {Promise<string>} Memory ID
   */
  async function rememberProblemSolution(problem, solution, metadata = {}) {
    return await addMemory({
      type: MemoryType.PROBLEM_SOLUTION,
      task: problem,
      insight: solution,
      metadata
    });
  }
  
  /**
   * Get memories about code patterns
   * 
   * @param {string} context The context to find patterns for
   * @param {Object} options Query options
   * @returns {Promise<Array<Object>>} Relevant code patterns
   */
  async function getCodePatterns(context, options = {}) {
    return await getRelevantMemories(context, {
      ...options,
      types: [MemoryType.CODE_PATTERN]
    });
  }
  
  /**
   * Get memories about user preferences
   * 
   * @param {string} context The context to find preferences for
   * @param {Object} options Query options
   * @returns {Promise<Array<Object>>} Relevant user preferences
   */
  async function getUserPreferences(context, options = {}) {
    return await getRelevantMemories(context, {
      ...options,
      types: [MemoryType.USER_PREFERENCE]
    });
  }
  
  /**
   * Get memories about problem solutions
   * 
   * @param {string} problem The problem to find solutions for
   * @param {Object} options Query options
   * @returns {Promise<Array<Object>>} Relevant problem solutions
   */
  async function getProblemSolutions(problem, options = {}) {
    return await getRelevantMemories(problem, {
      ...options,
      types: [MemoryType.PROBLEM_SOLUTION]
    });
  }
  
  // Initialize the memory controller
  initialize();
  
  // Return the memory controller interface
  return {
    addMemory,
    getRelevantMemories,
    removeMemory,
    clearMemories,
    formatMemoriesForPrompt,
    rememberCodePattern,
    rememberUserPreference,
    rememberProblemSolution,
    getCodePatterns,
    getUserPreferences,
    getProblemSolutions,
    MemoryType
  };
}

module.exports = {
  createMemoryController
};
```

# src/agent/enhanced/rag-implementation.js

```js
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
      formattedContext += '\`\`\`\n';
      formattedContext += ctx.content;
      formattedContext += '\n\`\`\`\n\n';
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

# src/agent/runner.js

```js
/**
 * Agent Runner for FrankCode
 * 
 * This module is responsible for executing agent actions,
 * coordinating between the LLM, tools, and user interface.
 */

const { logger } = require('../utils/logger');
const fileOperations = require('./tools/fileOperations');
const { confirmOperation } = require('../tui/confirmation');
const { createTaskPlanner } = require('./taskPlanner');

/**
 * Agent Action Types
 */
const ActionType = {
  READ_FILE: 'read_file',
  SEARCH_FILES: 'search_files',
  UPDATE_FILE: 'update_file',
  CREATE_FILE: 'create_file',
  DELETE_FILE: 'delete_file',
  GENERATE_CODE: 'generate_code',
  ANALYZE: 'analyze',
  ANSWER_QUESTION: 'answer_question'
};

/**
 * Create an agent runner
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.llm LLM client
 * @param {Object} options.screen Blessed screen object
 * @param {Object} options.outputRenderer Output renderer
 * @param {Object} options.agent Agent instance
 * @returns {Object} Agent runner interface
 */
function createAgentRunner({ llm, screen, outputRenderer, agent }) {
  // Store preferences
  let autoConfirm = false;
  
  // Create task planner
  const taskPlanner = createTaskPlanner({ llm, agent });
  
  // Track the current task execution
  let currentTask = null;
  let currentPlan = [];
  let currentStepIndex = 0;
  
  /**
   * Execute a complex task by breaking it down and sequentially processing steps
   * 
   * @param {string} taskDescription Description of the task
   * @returns {Promise<Object>} Task result
   */
  async function executeTask(taskDescription) {
    try {
      // Initialize task tracking
      currentTask = taskDescription;
      currentStepIndex = 0;
      
      // Add system message
      outputRenderer.addSystemMessage(`🧠 Analyzing task: ${taskDescription}`);
      
      // Get the current context from the agent
      const context = agent.getContextManager().getCurrentContext();
      
      // Create a plan for the task
      currentPlan = await taskPlanner.planTask(taskDescription, context);
      
      // If planning failed, return error
      if (currentPlan.length === 0 || (currentPlan.length === 1 && currentPlan[0].type === 'error')) {
        outputRenderer.addErrorMessage('Failed to create a plan for this task. Please try being more specific.');
        return { status: 'error', error: 'Failed to create plan' };
      }
      
      // Display the plan
      outputRenderer.addSystemMessage(`📋 Task Plan (${currentPlan.length} steps):`);
      
      currentPlan.forEach((step, index) => {
        outputRenderer.addSystemMessage(`${index + 1}. ${step.type.toUpperCase()}: ${step.description}`);
      });
      
      // Assess task difficulty
      const difficulty = taskPlanner.assessTaskDifficulty(taskDescription, currentPlan);
      outputRenderer.addSystemMessage(`⏱️ Estimated time: ~${difficulty.estimatedMinutes} minute${difficulty.estimatedMinutes > 1 ? 's' : ''} (${difficulty.difficulty} complexity)`);
      
      // Execute each step in the plan
      for (currentStepIndex = 0; currentStepIndex < currentPlan.length; currentStepIndex++) {
        const step = currentPlan[currentStepIndex];
        
        // Display current step
        outputRenderer.addSystemMessage(`\n▶️ Step ${currentStepIndex + 1}/${currentPlan.length}: ${step.description}`);
        
        // Execute the step
        const stepResult = await executeStep(step);
        
        // Handle step failure
        if (stepResult.status === 'error') {
          outputRenderer.addErrorMessage(`Failed to complete step ${currentStepIndex + 1}: ${stepResult.error}`);
          
          // Ask the LLM for recovery suggestions
          const recoveryResult = await suggestRecovery(step, stepResult.error);
          outputRenderer.addSystemMessage(`💡 Recovery suggestion: ${recoveryResult.suggestion}`);
          
          // We may want to continue with next steps even if one fails
          if (recoveryResult.fatal) {
            outputRenderer.addErrorMessage('Cannot continue with the task due to critical error.');
            return { status: 'error', error: stepResult.error, completedSteps: currentStepIndex };
          }
        } else if (stepResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Completed step ${currentStepIndex + 1}`);
        }
        
        // Force render after each step
        screen.render();
      }
      
      // Task completed successfully
      outputRenderer.addSystemMessage(`\n✅ Task completed successfully!`);
      
      return { 
        status: 'success',
        steps: currentPlan.length,
        completedSteps: currentStepIndex
      };
    } catch (error) {
      logger.error(`Failed to execute task: ${taskDescription}`, { error });
      outputRenderer.addErrorMessage(`Error executing task: ${error.message}`);
      
      return { 
        status: 'error', 
        error: error.message,
        completedSteps: currentStepIndex
      };
    } finally {
      // Reset the task tracking
      currentTask = null;
      currentPlan = [];
      currentStepIndex = 0;
    }
  }
  
  /**
   * Execute a single step from the plan
   * 
   * @param {Object} step Step to execute
   * @returns {Promise<Object>} Step result
   */
  async function executeStep(step) {
    try {
      logger.debug(`Executing step: ${step.type}`, { step });
      
      switch (step.type) {
        case ActionType.READ_FILE:
          return await executeReadFile(step);
        
        case ActionType.SEARCH_FILES:
          return await executeSearchFiles(step);
        
        case ActionType.UPDATE_FILE:
          return await executeUpdateFile(step);
        
        case ActionType.CREATE_FILE:
          return await executeCreateFile(step);
        
        case ActionType.DELETE_FILE:
          return await executeDeleteFile(step);
        
        case ActionType.GENERATE_CODE:
          return await executeGenerateCode(step);
        
        case ActionType.ANALYZE:
          return await executeAnalyze(step);
        
        case ActionType.ANSWER_QUESTION:
          return await executeAnswerQuestion(step);
        
        default:
          return {
            type: step.type,
            status: 'error',
            error: `Unknown step type: ${step.type}`
          };
      }
    } catch (error) {
      logger.error(`Failed to execute step: ${step.type}`, { error });
      return {
        type: step.type,
        status: 'error',
        error: error.message
      };
    }
  }
  
  /**
   * Execute read file action
   * 
   * @param {Object} step Read file step
   * @returns {Promise<Object>} Action result
   */
  async function executeReadFile(step) {
    // Check if we have specific files to read
    if (!step.files || step.files.length === 0) {
      outputRenderer.addErrorMessage('No files specified for reading');
      return { status: 'error', error: 'No files specified' };
    }
    
    const results = [];
    
    // Read each file
    for (const filePath of step.files) {
      // Display action in UI
      outputRenderer.addSystemMessage(`📄 Reading file: ${filePath}`);
      
      // Execute the action
      const result = await fileOperations.readFile(filePath);
      
      // Display result in UI
      if (result.status === 'success') {
        outputRenderer.addCodeBlock(result.rawContent, getFileLanguage(filePath));
        results.push(result);
      } else {
        outputRenderer.addErrorMessage(`Failed to read file: ${result.error}`);
        return { status: 'error', error: result.error };
      }
    }
    
    return { status: 'success', results };
  }
  
  /**
   * Execute search files action
   * 
   * @param {Object} step Search files step
   * @returns {Promise<Object>} Action result
   */
  async function executeSearchFiles(step) {
    let pattern = '*';
    let baseDir = '';
    
    // Extract pattern from description or use files
    if (step.files && step.files.length > 0) {
      pattern = step.files[0];
    } else {
      // Try to extract pattern from description
      const patternMatch = /pattern[:\s]+([^\s]+)/i.exec(step.description);
      if (patternMatch) {
        pattern = patternMatch[1];
      }
    }
    
    // Display action in UI
    outputRenderer.addSystemMessage(`🔍 Searching for files: ${pattern}${baseDir ? ` in ${baseDir}` : ''}`);
    
    // Execute the action
    const result = await fileOperations.searchFiles(pattern, baseDir);
    
    // Display result in UI
    if (result.status === 'success') {
      if (result.matches.length > 0) {
        outputRenderer.addSystemMessage(`Found ${result.matches.length} matching files:`);
        outputRenderer.addCodeBlock(result.matches.join('\n'), 'bash');
      } else {
        outputRenderer.addSystemMessage('No matching files found.');
      }
    } else {
      outputRenderer.addErrorMessage(`Failed to search files: ${result.error}`);
      return { status: 'error', error: result.error };
    }
    
    return { status: 'success', result };
  }
  
  /**
   * Execute update file action
   * 
   * @param {Object} step Update file step
   * @returns {Promise<Object>} Action result
   */
  async function executeUpdateFile(step) {
    // For update operations, we'll need to query the LLM
    // to generate the actual changes based on step description
    if (!step.files || step.files.length === 0) {
      outputRenderer.addErrorMessage('No files specified for update');
      return { status: 'error', error: 'No files specified' };
    }
    
    // Go through each file to update
    for (const filePath of step.files) {
      // First, read the file
      outputRenderer.addSystemMessage(`📄 Reading file for update: ${filePath}`);
      const readResult = await fileOperations.readFile(filePath);
      
      if (readResult.status !== 'success') {
        outputRenderer.addErrorMessage(`Failed to read file: ${readResult.error}`);
        return { status: 'error', error: readResult.error };
      }
      
      // Display current content
      outputRenderer.addCodeBlock(readResult.rawContent, getFileLanguage(filePath));
      
      // Generate changes using the LLM
      outputRenderer.addSystemMessage(`🧠 Generating updates for: ${filePath}`);
      
      const updatePrompt = `You are helping update code in the file: ${filePath}
Current content:
\`\`\`
${readResult.rawContent}
\`\`\`

Update needed: ${step.description}

Provide the complete updated file content, with appropriate changes to satisfy the requirement.
Only return the updated code without explanations or markdown.
`;
      
      const generatedUpdate = await llm.generateResponse(updatePrompt);
      
      // Extract the code (remove any markdown or explanation the LLM might add)
      const newContent = extractCodeContent(generatedUpdate.text);
      
      // Display action in UI
      outputRenderer.addSystemMessage(`✏️ Update file: ${filePath}`);
      
      // Execute the action
      const updateResult = await fileOperations.updateFile(filePath, newContent);
      
      // Handle errors
      if (updateResult.status === 'error') {
        outputRenderer.addErrorMessage(`Failed to update file: ${updateResult.error}`);
        return { status: 'error', error: updateResult.error };
      }
      
      // Show a message about the changes
      outputRenderer.addSystemMessage(`Changes: ${updateResult.additions} additions, ${updateResult.removals} removals`);
      
      // Auto-confirm if enabled
      if (autoConfirm) {
        const confirmResult = await fileOperations.confirmUpdate(updateResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Updated ${filePath} with ${updateResult.additions} additions and ${updateResult.removals} removals`);
        } else {
          outputRenderer.addErrorMessage(`Failed to update file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      }
      
      // Request confirmation from user
      const confirmation = await confirmOperation(screen, updateResult);
      
      // Handle confirmation result
      if (confirmation.choice === 'yes' || confirmation.choice === 'yes-to-all') {
        // Set auto-confirm if yes-to-all
        if (confirmation.choice === 'yes-to-all') {
          autoConfirm = true;
        }
        
        // Confirm the update
        const confirmResult = await fileOperations.confirmUpdate(updateResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Updated ${filePath} with ${updateResult.additions} additions and ${updateResult.removals} removals`);
        } else {
          outputRenderer.addErrorMessage(`Failed to update file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      } else if (confirmation.choice === 'custom') {
        // Apply custom content
        const customResult = await fileOperations.updateFile(filePath, confirmation.customContent);
        const confirmResult = await fileOperations.confirmUpdate(customResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Updated ${filePath} with custom changes`);
        } else {
          outputRenderer.addErrorMessage(`Failed to update file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      } else {
        // Cancelled or rejected
        outputRenderer.addSystemMessage(`❌ Update cancelled for ${filePath}`);
        return { status: 'cancelled', filePath };
      }
    }
    
    return { status: 'error', error: 'No files were processed' };
  }
  
  /**
   * Execute create file action
   * 
   * @param {Object} step Create file step
   * @returns {Promise<Object>} Action result
   */
  async function executeCreateFile(step) {
    // For create operations, we'll need to query the LLM
    // to generate the content based on step description
    if (!step.files || step.files.length === 0) {
      outputRenderer.addErrorMessage('No files specified for creation');
      return { status: 'error', error: 'No files specified' };
    }
    
    // Go through each file to create
    for (const filePath of step.files) {
      // Generate content using the LLM
      outputRenderer.addSystemMessage(`🧠 Generating content for new file: ${filePath}`);
      
      const createPrompt = `You are helping create a new file: ${filePath}

File purpose: ${step.description}

Please generate appropriate content for this file. Consider the file extension and purpose.
Only return the code without explanations or markdown.
`;
      
      const generatedContent = await llm.generateResponse(createPrompt);
      
      // Extract the code (remove any markdown or explanation the LLM might add)
      const content = extractCodeContent(generatedContent.text);
      
      // Display action in UI
      outputRenderer.addSystemMessage(`🔨 Creating file: ${filePath}`);
      outputRenderer.addCodeBlock(content, getFileLanguage(filePath));
      
      // Execute the action
      const createResult = await fileOperations.createFile(filePath, content);
      
      // Handle errors
      if (createResult.status === 'error') {
        outputRenderer.addErrorMessage(`Failed to create file: ${createResult.error}`);
        return { status: 'error', error: createResult.error };
      }
      
      // Auto-confirm if enabled
      if (autoConfirm) {
        const confirmResult = await fileOperations.confirmCreate(createResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Created ${filePath}`);
        } else {
          outputRenderer.addErrorMessage(`Failed to create file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      }
      
      // Request confirmation from user
      const confirmation = await confirmOperation(screen, createResult);
      
      // Handle confirmation result
      if (confirmation.choice === 'yes' || confirmation.choice === 'yes-to-all') {
        // Set auto-confirm if yes-to-all
        if (confirmation.choice === 'yes-to-all') {
          autoConfirm = true;
        }
        
        // Confirm the creation
        const confirmResult = await fileOperations.confirmCreate(createResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Created ${filePath}`);
        } else {
          outputRenderer.addErrorMessage(`Failed to create file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      } else if (confirmation.choice === 'custom') {
        // Apply custom content
        const customResult = await fileOperations.createFile(filePath, confirmation.customContent);
        const confirmResult = await fileOperations.confirmCreate(customResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Created ${filePath} with custom content`);
        } else {
          outputRenderer.addErrorMessage(`Failed to create file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      } else {
        // Cancelled or rejected
        outputRenderer.addSystemMessage(`❌ Creation cancelled for ${filePath}`);
        return { status: 'cancelled', filePath };
      }
    }
    
    return { status: 'error', error: 'No files were processed' };
  }
  
  /**
   * Execute delete file action
   * 
   * @param {Object} step Delete file step
   * @returns {Promise<Object>} Action result
   */
  async function executeDeleteFile(step) {
    if (!step.files || step.files.length === 0) {
      outputRenderer.addErrorMessage('No files specified for deletion');
      return { status: 'error', error: 'No files specified' };
    }
    
    // Go through each file to delete
    for (const filePath of step.files) {
      // Display action in UI
      outputRenderer.addSystemMessage(`🗑️ Delete file: ${filePath}`);
      
      // Execute the action
      const deleteResult = await fileOperations.deleteFile(filePath);
      
      // Handle errors
      if (deleteResult.status === 'error') {
        outputRenderer.addErrorMessage(`Failed to delete file: ${deleteResult.error}`);
        return { status: 'error', error: deleteResult.error };
      }
      
      // Request confirmation from user (never auto-confirm deletion)
      const confirmation = await confirmOperation(screen, deleteResult);
      
      // Handle confirmation result
      if (confirmation.choice === 'yes') {
        // Confirm the deletion
        const confirmResult = await fileOperations.confirmDelete(deleteResult);
        
        if (confirmResult.status === 'success') {
          outputRenderer.addSystemMessage(`✅ Deleted ${filePath}`);
        } else {
          outputRenderer.addErrorMessage(`Failed to delete file: ${confirmResult.error}`);
          return { status: 'error', error: confirmResult.error };
        }
        
        return { status: 'success', filePath, confirmResult };
      } else {
        // Cancelled or rejected
        outputRenderer.addSystemMessage(`❌ Deletion cancelled for ${filePath}`);
        return { status: 'cancelled', filePath };
      }
    }
    
    return { status: 'error', error: 'No files were processed' };
  }
  
  /**
   * Execute generate code action
   * 
   * @param {Object} step Generate code step
   * @returns {Promise<Object>} Action result
   */
  async function executeGenerateCode(step) {
    try {
      // Display action in UI
      outputRenderer.addSystemMessage(`💻 Generating code: ${step.description}`);
      
      // Generate code using the LLM
      const prompt = `You are FrankCode, an AI programming assistant.
Generate high-quality code based on the following requirements:

${step.description}

${step.files && step.files.length > 0 ? `The code should be for file: ${step.files[0]}` : ''}

${step.reasoning ? `Additional context: ${step.reasoning}` : ''}

Provide only the code without any explanations or markdown formatting.`;
      
      const response = await llm.generateResponse(prompt);
      
      // Display the generated code
      const code = extractCodeContent(response.text);
      outputRenderer.addCodeBlock(code, step.files && step.files.length > 0 ? getFileLanguage(step.files[0]) : 'javascript');
      
      // If we have a file specified, offer to save it
      if (step.files && step.files.length > 0) {
        outputRenderer.addSystemMessage(`Would you like to save this code to ${step.files[0]}?`);
        outputRenderer.addSystemMessage(`Type /yes to save, or /no to just keep the code in the conversation.`);
        
        // The actual saving will be handled by the input handler
        return { 
          status: 'success', 
          code,
          pendingAction: {
            type: 'save_code',
            filePath: step.files[0],
            content: code
          }
        };
      }
      
      return { status: 'success', code };
    } catch (error) {
      logger.error(`Failed to generate code: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Failed to generate code: ${error.message}`);
      return { status: 'error', error: error.message };
    }
  }
  
  /**
   * Execute analyze action
   * 
   * @param {Object} step Analyze step
   * @returns {Promise<Object>} Action result
   */
  async function executeAnalyze(step) {
    try {
      // Display action in UI
      outputRenderer.addSystemMessage(`🔍 Analyzing: ${step.description}`);
      
      // Build context based on the files
      let fileContents = '';
      if (step.files && step.files.length > 0) {
        for (const filePath of step.files) {
          const readResult = await fileOperations.readFile(filePath);
          if (readResult.status === 'success') {
            fileContents += `File: ${filePath}\n\n${readResult.rawContent}\n\n`;
          }
        }
      }
      
      // Generate analysis using the LLM
      const prompt = `You are FrankCode, an AI programming assistant.
Analyze the following code and provide insights based on this request:

${step.description}

${fileContents ? `Code to analyze:\n${fileContents}` : ''}

${step.reasoning ? `Analysis focus: ${step.reasoning}` : ''}

Provide a detailed analysis addressing the request.`;
      
      const response = await llm.generateResponse(prompt, {
        temperature: 0.3 // Lower temperature for more analytical responses
      });
      
      // Display the analysis
      outputRenderer.addAssistantMessage(response.text);
      
      return { status: 'success', analysis: response.text };
    } catch (error) {
      logger.error(`Failed to analyze: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Failed to analyze: ${error.message}`);
      return { status: 'error', error: error.message };
    }
  }
  
  /**
   * Execute answer question action
   * 
   * @param {Object} step Answer question step
   * @returns {Promise<Object>} Action result
   */
  async function executeAnswerQuestion(step) {
    try {
      // Display action in UI
      outputRenderer.addSystemMessage(`❓ Answering: ${step.description}`);
      
      // Build context based on the files
      let fileContents = '';
      if (step.files && step.files.length > 0) {
        for (const filePath of step.files) {
          const readResult = await fileOperations.readFile(filePath);
          if (readResult.status === 'success') {
            fileContents += `File: ${filePath}\n\n${readResult.rawContent}\n\n`;
          }
        }
      }
      
      // Generate answer using the LLM
      const prompt = `You are FrankCode, an AI programming assistant.
Answer the following question:

${step.description}

${fileContents ? `Related code for context:\n${fileContents}` : ''}

${step.reasoning ? `Additional context: ${step.reasoning}` : ''}

Provide a clear and comprehensive answer.`;
      
      const response = await llm.generateResponse(prompt);
      
      // Display the answer
      outputRenderer.addAssistantMessage(response.text);
      
      return { status: 'success', answer: response.text };
    } catch (error) {
      logger.error(`Failed to answer question: ${error.message}`, { error });
      outputRenderer.addErrorMessage(`Failed to answer question: ${error.message}`);
      return { status: 'error', error: error.message };
    }
  }
  
  /**
   * Suggest recovery options for a failed step
   * 
   * @param {Object} step Failed step
   * @param {string} error Error message
   * @returns {Promise<Object>} Recovery suggestions
   */
  async function suggestRecovery(step, error) {
    try {
      const prompt = `You are FrankCode, an AI programming assistant.
A step in my task execution has failed. Please suggest how to recover:

Failed step: ${step.type} - ${step.description}
${step.files && step.files.length > 0 ? `Files involved: ${step.files.join(', ')}` : ''}
Error message: ${error}

Provide:
1. A brief explanation of what might have gone wrong
2. A specific suggestion on how to recover or work around this issue
3. Indicate if this is a fatal error that should halt the entire task (true/false)`;
      
      const response = await llm.generateResponse(prompt, {
        temperature: 0.3 // Lower temperature for more precise recovery suggestions
      });
      
      // Try to extract whether it's fatal
      const fatalMatch = /fatal:?\s*(true|false)/i.exec(response.text);
      const isFatal = fatalMatch ? fatalMatch[1].toLowerCase() === 'true' : false;
      
      return {
        suggestion: response.text,
        fatal: isFatal
      };
    } catch (error) {
      logger.error(`Failed to suggest recovery: ${error.message}`, { error });
      return {
        suggestion: "Failed to generate recovery suggestions. You may need to modify your task or approach.",
        fatal: true
      };
    }
  }
  
  /**
   * Extract code content from LLM response, removing markdown and explanations
   * 
   * @param {string} text LLM response text
   * @returns {string} Extracted code content
   */
  function extractCodeContent(text) {
    // Check for code block
    const codeBlockMatch = /\`\`\`(?:\w*\n)?([\s\S]*?)\`\`\`/g.exec(text);
    if (codeBlockMatch) {
      return codeBlockMatch[1].trim();
    }
    
    // No code block, strip any explanations (look for patterns like explanations followed by actual code)
    const lines = text.split('\n');
    
    // Look for patterns indicating explanations
    const codeStartIndex = lines.findIndex((line, index) => {
      // Common patterns that indicate the start of code after explanations
      const codeStartPatterns = [
        /^(class|function|import|const|let|var|#|\/\/|\*|\/\*|<!)/,
        /^[a-zA-Z0-9_]+\s*\(\s*\)\s*{/,
        /^export /,
        /^public /,
        /^private /,
        /^module\./
      ];
      
      return index > 0 && codeStartPatterns.some(pattern => pattern.test(line));
    });
    
    if (codeStartIndex > 0) {
      return lines.slice(codeStartIndex).join('\n').trim();
    }
    
    // If no obvious code start found, return the original text
    return text.trim();
  }
  
  /**
   * Get the language for syntax highlighting based on file extension
   * 
   * @param {string} filePath File path
   * @returns {string} Language for syntax highlighting
   */
  function getFileLanguage(filePath) {
    const extension = filePath.split('.').pop().toLowerCase();
    
    const languageMap = {
      'js': 'javascript',
      'jsx': 'javascript',
      'ts': 'typescript',
      'tsx': 'typescript',
      'py': 'python',
      'rb': 'ruby',
      'go': 'go',
      'java': 'java',
      'c': 'c',
      'cpp': 'cpp',
      'h': 'c',
      'hpp': 'cpp',
      'cs': 'csharp',
      'php': 'php',
      'html': 'html',
      'css': 'css',
      'scss': 'scss',
      'sass': 'scss',
      'json': 'json',
      'md': 'markdown',
      'xml': 'xml',
      'sh': 'bash',
      'bash': 'bash',
      'yml': 'yaml',
      'yaml': 'yaml',
      'sql': 'sql',
      'rs': 'rust',
      'swift': 'swift'
    };
    
    return languageMap[extension] || 'text';
  }
  
  /**
   * Execute a direct agent action
   * 
   * @param {Object} action Action to execute
   * @returns {Promise<Object>} Action result
   */
  async function executeAction(action) {
    try {
      // Create a step from the action
      const step = {
        type: action.type,
        description: action.description || '',
        files: action.files || [],
        reasoning: action.reasoning || ''
      };
      
      // Execute the step
      return await executeStep(step);
    } catch (error) {
      logger.error(`Failed to execute action: ${action.type}`, { error });
      return {
        type: action.type,
        status: 'error',
        error: error.message
      };
    }
  }
  
  // Return the agent runner interface
  return {
    executeTask,
    executeAction,
    ActionType
  };
}

module.exports = {
  createAgentRunner
};
```

# src/agent/taskPlanner.js

```js
/**
 * Task Planning Module for FrankCode
 * 
 * This module is responsible for breaking down complex tasks into
 * manageable steps and creating an execution plan.
 */

const { logger } = require('../utils/logger');
const { countTokens } = require('./tokenizer');

/**
 * Create a task planner
 * 
 * @param {Object} options Configuration options
 * @param {Object} options.llm LLM client for generating plans
 * @param {Object} options.agent Agent instance
 * @returns {Object} The task planner interface
 */
function createTaskPlanner({ llm, agent }) {
  /**
   * Break down a complex task into steps
   * 
   * @param {string} task The high-level task description
   * @param {Object} context Current project context
   * @returns {Promise<Array<Object>>} Planned steps
   */
  async function planTask(task, context) {
    try {
      logger.info(`Planning task: ${task}`);
      
      // Generate the planning prompt
      const prompt = generatePlanningPrompt(task, context);
      
      // Use the LLM to generate a plan
      const response = await llm.generateResponse(prompt, {
        temperature: 0.3, // Lower temperature for more consistent planning
        maxTokens: 1024   // Allow for detailed plans
      });
      
      // Parse the plan from the response
      const steps = parsePlanSteps(response.text);
      
      logger.info(`Task plan created with ${steps.length} steps`);
      return steps;
    } catch (error) {
      logger.error(`Failed to plan task: ${error.message}`, { error });
      return [{ 
        type: 'error', 
        description: `Failed to create plan: ${error.message}`,
        action: 'notify'
      }];
    }
  }
  
  /**
   * Generate a prompt for task planning
   * 
   * @param {string} task The task description
   * @param {Object} context Current project context
   * @returns {string} The planning prompt
   */
  function generatePlanningPrompt(task, context) {
    // Include only the most relevant files to keep context manageable
    const relevantFiles = getRelevantFilesForTask(task, context);
    
    return `You are FrankCode, an AI coding assistant that helps with complex programming tasks.
I need you to break down the following task into a clear step-by-step plan:

TASK: ${task}

Current project context:
${relevantFiles.map(file => `- ${file.filePath}: ${file.summary}`).join('\n')}

Your task is to break this down into smaller steps. For each step:
1. Specify the type of action required (read_file, search_files, update_file, create_file, generate_code, analyze)
2. Provide a clear description of what needs to be done
3. List any specific files that need to be examined or modified
4. Explain the reasoning for this step

FORMAT YOUR RESPONSE AS:
STEP 1:
Type: [action_type]
Description: [clear description]
Files: [file paths, if relevant]
Reasoning: [brief explanation]

STEP 2:
...

Ensure your plan is comprehensive yet concise. Identify 3-7 key steps to make the task manageable.`;
  }
  
  /**
   * Parse the steps from the LLM response
   * 
   * @param {string} response The LLM response text
   * @returns {Array<Object>} Parsed steps
   */
  function parsePlanSteps(response) {
    const steps = [];
    const stepRegex = /STEP\s+(\d+):\s*\n([\s\S]*?)(?=STEP\s+\d+:|$)/g;
    
    let match;
    while ((match = stepRegex.exec(response)) !== null) {
      const stepContent = match[2].trim();
      
      // Parse step details
      const typeMatch = /Type:\s*(\w+)/.exec(stepContent);
      const descMatch = /Description:\s*(.+?)(?=\n\w+:|$)/s.exec(stepContent);
      const filesMatch = /Files:\s*(.+?)(?=\n\w+:|$)/s.exec(stepContent);
      const reasoningMatch = /Reasoning:\s*(.+?)(?=\n\w+:|$)/s.exec(stepContent);
      
      if (typeMatch && descMatch) {
        steps.push({
          type: typeMatch[1].toLowerCase(),
          description: descMatch[1].trim(),
          files: filesMatch ? 
            filesMatch[1].trim().split(/,\s*/).filter(f => f !== 'N/A' && f !== '') : 
            [],
          reasoning: reasoningMatch ? reasoningMatch[1].trim() : ''
        });
      }
    }
    
    // If no steps were parsed but we have content, create a fallback step
    if (steps.length === 0 && response.trim().length > 0) {
      steps.push({
        type: 'analyze',
        description: 'Analyze the task and provide guidance',
        files: [],
        reasoning: 'No structured plan could be created, falling back to general analysis'
      });
    }
    
    return steps;
  }
  
  /**
   * Get relevant files for a task based on content and keywords
   * 
   * @param {string} task The task description
   * @param {Object} context Current project context
   * @returns {Array<Object>} Relevant files
   */
  function getRelevantFilesForTask(task, context) {
    // Simple relevance scoring based on keywords in the task
    const taskWords = task.toLowerCase().split(/\W+/).filter(w => w.length > 3);
    const relevanceScores = new Map();
    
    // Score each file based on matching keywords
    for (const file of context.fileContexts) {
      let score = 0;
      
      // File name relevance (weighted higher)
      const fileName = file.filePath.toLowerCase();
      taskWords.forEach(word => {
        if (fileName.includes(word)) {
          score += 5;
        }
      });
      
      // Content relevance
      const content = file.content.toLowerCase();
      taskWords.forEach(word => {
        const regex = new RegExp(word, 'g');
        const matches = content.match(regex);
        if (matches) {
          score += matches.length;
        }
      });
      
      relevanceScores.set(file, score);
    }
    
    // Sort files by relevance score and take top 5
    return Array.from(context.fileContexts)
      .sort((a, b) => relevanceScores.get(b) - relevanceScores.get(a))
      .slice(0, 5);
  }
  
  /**
   * Estimate the difficulty and time required for a task
   * 
   * @param {string} task The task description
   * @param {Array<Object>} steps The planned steps
   * @returns {Object} Difficulty assessment
   */
  function assessTaskDifficulty(task, steps) {
    // Count total files to be modified
    const uniqueFiles = new Set();
    steps.forEach(step => {
      step.files.forEach(file => uniqueFiles.add(file));
    });
    
    // Assess based on number of steps and files
    const filesToModify = uniqueFiles.size;
    const stepCount = steps.length;
    
    // Count complexity factors in the task description
    const complexityFactors = [
      /refactor/i, 
      /optimize/i, 
      /architecture/i, 
      /redesign/i, 
      /performance/i, 
      /security/i
    ];
    
    const complexityScore = complexityFactors.reduce((score, regex) => 
      regex.test(task) ? score + 1 : score, 0);
    
    let difficulty;
    if (stepCount <= 2 && filesToModify <= 1 && complexityScore === 0) {
      difficulty = 'easy';
    } else if (stepCount <= 5 && filesToModify <= 3 && complexityScore <= 1) {
      difficulty = 'medium';
    } else {
      difficulty = 'complex';
    }
    
    // Estimate time (very rough estimate)
    let estimatedMinutes;
    switch (difficulty) {
      case 'easy':
        estimatedMinutes = 1;
        break;
      case 'medium':
        estimatedMinutes = 3;
        break;
      case 'complex':
        estimatedMinutes = 5 + (stepCount - 5) * 2;
        break;
    }
    
    return {
      difficulty,
      estimatedMinutes,
      stepCount,
      filesToModify
    };
  }
  
  // Return the task planner interface
  return {
    planTask,
    assessTaskDifficulty
  };
}

module.exports = {
  createTaskPlanner
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

let tiktoken;
try {
  tiktoken = require('tiktoken');
} catch (error) {
  // Silent fail - don't log anything here
}

// Cache encoding to avoid recreating it on each call
let encoding;
let logger;

/**
 * Initialize the tokenizer
 * 
 * @param {string} model Model name for tokenization (default is cl100k, used by many models)
 * @returns {void}
 */
function initTokenizer(model = 'cl100k_base') {
  try {
    // Only import logger after we're initialized to avoid circular dependencies
    if (!logger) {
      logger = require('../utils/logger').logger;
    }
    
    if (tiktoken) {
      encoding = tiktoken.getEncoding(model);
      logger.debug(`Tokenizer initialized with model: ${model}`);
    } else {
      // Fallback when tiktoken is not available - don't log warning
      logger.debug('Using fallback tokenization method - tiktoken not available');
      encoding = null;
    }
  } catch (error) {
    // Log error to file but don't throw - silently use fallback
    if (logger) {
      logger.debug('Failed to initialize tokenizer, using fallback', { error });
    }
    
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
    
    // Fallback tokenization (approximate) - don't log warning
    // This is a simplistic approximation - in practice, real tokenization is more complex
    return Math.ceil(text.length / 4);
  } catch (error) {
    if (logger) {
      logger.debug('Error counting tokens, using fallback', { error });
    }
    
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
    
    // Fallback truncation (approximate) - no warning log
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
    if (logger) {
      logger.debug('Error truncating text to tokens, using simple fallback', { error });
    }
    
    // Return a conservatively truncated text
    const approxCharPerToken = 4;
    const charLimit = maxTokens * approxCharPerToken;
    return text.substring(0, charLimit) + '...';
  }
}

// Lazy initialization - initialize when first used

module.exports = {
  countTokens,
  truncateToTokens,
  initTokenizer
};
```

# src/agent/tools/fileOperations.js

```js
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
const { logger } = require('../utils/logger');
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
    try {
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
    } catch (error) {
      logger.error('Failed to initialize API client connection', { error });
      isConnected = false;
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
      
      // Handle error from API client
      if (response.error) {
        // Add a special response for network errors
        conversationHistory.push({ role: 'assistant', content: response.text });
        
        return {
          text: response.text,
          error: true,
          tokenUsage: tokenMonitor.getCurrentUsage()
        };
      }
      
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
      
      // Add error to conversation history
      conversationHistory.push({ 
        role: 'assistant', 
        content: `Error processing message: ${error.message}` 
      });
      
      return {
        text: `Error processing message: ${error.message}`,
        error: true,
        tokenUsage: tokenMonitor.getCurrentUsage()
      };
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
      // Improved error handling for better TUI display
      logger.error('Failed to generate response from Ollama', { error });
      
      // Return a user-friendly message instead of throwing
      return {
        text: `Error: Connection to Ollama failed. Try running with --offline flag or make sure Ollama is running with 'ollama serve'\n\nTechnical details: ${error.message}`,
        tokens: 0,
        error: true
      };
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

// Export the client creation function
module.exports = {
  createClient
};
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
const { createOllamaClient } = require('./ollama-api-integration');

module.exports = {
  createClient,
  createQueue,
  createWebSocketClient,
  createOllamaClient
};
```

# src/api/ollama-api-integration.js

```js
/**
 * Ollama API Integration for FrankCode
 * 
 * This module provides an API client for connecting to Ollama LLM service.
 * It's designed to be a drop-in replacement for the distributed LLM client.
 */

const fetch = require('node-fetch');
const { logger } = require('../utils/logger');

/**
 * Create an Ollama API client
 * 
 * @param {Object} options Configuration options
 * @param {string} options.host Host for the Ollama service
 * @param {number} options.port Port for the Ollama service
 * @param {string} options.model Model to use
 * @param {number} options.temperature Temperature setting (0-1)
 * @returns {Object} The Ollama API client interface
 */
function createOllamaClient(options) {
  const {
    host = 'localhost',
    port = 11434,
    model = 'llama2',
    temperature = 0.7
  } = options;
  
  const baseUrl = `http://${host}:${port}`;
  let isConnected = false;
  
  /**
   * Initialize the client and test connection
   * 
   * @returns {Promise<boolean>} Connection success
   */
  async function initialize() {
    try {
      logger.debug('Testing Ollama connection...');
      
      // Check if Ollama is running
      const response = await fetch(`${baseUrl}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      const models = data.models || [];
      
      // Log available models
      logger.info(`Connected to Ollama. Available models: ${models.map(m => m.name).join(', ') || 'none'}`);
      
      // Check if our model is available
      const modelAvailable = models.some(m => m.name === model);
      if (!modelAvailable && models.length > 0) {
        // Automatically use the first available model
        model = models[0].name;
        logger.info(`Model switched to available model: ${model}`);
      } else if (!modelAvailable) {
        logger.warn(`Model '${model}' not found in Ollama. Available models: ${models.map(m => m.name).join(', ')}`);
      }
      
      await testOllamaEndpoint();
      
      isConnected = true;
      return true;
    } catch (error) {
      logger.error(`Failed to connect to Ollama at ${baseUrl}: ${error.message}`);
      isConnected = false;
      return false;
    }
  }
  // Add this function near the top of the ollama-api-integration.js file, just inside the createOllamaClient function
async function testOllamaEndpoint() {
    try {
      console.log(`Testing Ollama API at ${baseUrl}...`);
      
      // Test /api/tags endpoint
      console.log("Testing /api/tags endpoint...");
      const tagsResponse = await fetch(`${baseUrl}/api/tags`);
      console.log(`/api/tags status: ${tagsResponse.status}`);
      
      if (tagsResponse.ok) {
        const tagsData = await tagsResponse.json();
        console.log(`Available models: ${JSON.stringify(tagsData)}`);
      }
      
      // Test minimal generate request
      console.log("Testing /api/generate endpoint with minimal request...");
      const minimalRequest = { model, prompt: "Hello" };
      console.log(`Request body: ${JSON.stringify(minimalRequest)}`);
      
      const genResponse = await fetch(`${baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(minimalRequest)
      });
      
      console.log(`/api/generate status: ${genResponse.status}`);
      console.log(`/api/generate status text: ${genResponse.statusText}`);
      
      // Log full request and response details
      console.log(`Full request URL: ${baseUrl}/api/generate`);
      console.log(`Host: ${host}, Port: ${port}`);
      
      // Try different endpoint variations
      console.log("Testing alternative endpoints...");
      const endpoints = [
        "/api/generate",
        "/api/completion",
        "/api/completions",
        "/api/chat/completions"
      ];
      
      for (const endpoint of endpoints) {
        try {
          const testResp = await fetch(`${baseUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(minimalRequest)
          });
          console.log(`${endpoint} status: ${testResp.status}`);
        } catch (err) {
          console.log(`${endpoint} error: ${err.message}`);
        }
      }
    } catch (error) {
      console.error(`Test failed: ${error.message}`);
    }
  }
  
  // Then add this line at the end of the initialize function:
  
  /**
   * Generate a response from Ollama
   * 
   * @param {string} prompt The prompt to send
   * @param {Object} options Additional options
   * @returns {Promise<Object>} The response
   */
  async function generateResponse(prompt, options = {}) {
    try {
      logger.debug(`Sending request to Ollama (${model})`);
      
      // Use a minimal request body matching the successful curl command
      const requestBody = {
        model: model,
        prompt: prompt
      };
      
      const response = await fetch(`${baseUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      // Read response as text first
      const responseText = await response.text();
      
      // Handle different response formats
      try {
        // Try parsing as single JSON object
        const data = JSON.parse(responseText);
        return {
          text: data.response || "No response text",
          tokens: data.eval_count || 0,
          model: data.model || model
        };
      } catch (parseError) {
        // If not single JSON, try parsing as multiple JSON lines
        try {
          const lines = responseText.split('\n').filter(line => line.trim());
          // Get the last complete JSON object
          const lastLine = lines[lines.length - 1];
          const lastData = JSON.parse(lastLine);
          
          // Combine all responses
          const fullText = lines.map(line => {
            try {
              const obj = JSON.parse(line);
              return obj.response || '';
            } catch (e) {
              return '';
            }
          }).join('');
          
          return {
            text: fullText,
            tokens: lastData.eval_count || lines.length,
            model: model
          };
        } catch (e) {
          // If all parsing fails, return raw text
          return {
            text: responseText,
            tokens: responseText.length / 4, // rough estimate
            model: model
          };
        }
      }
    } catch (error) {
      logger.error(`Error generating response from Ollama: ${error.message}`);
      return {
        text: `Error: Failed to generate response from Ollama. ${error.message}`,
        tokens: 0,
        error: true
      };
    }
  }
  
  /**
   * Stream a response from Ollama
   * 
   * @param {string} prompt The prompt to send
   * @param {Function} onToken Callback for each token
   * @param {Function} onComplete Callback when complete
   * @param {Object} options Additional options
   */
  async function streamResponse(prompt, onToken, onComplete, options = {}) {
    try {
      const requestOptions = {
        temperature: options.temperature || temperature,
        max_tokens: options.maxTokens || 2048,
        stop: options.stop || ['\n\nUSER:', '\n\nASSISTANT:']
      };
      
      logger.debug(`Streaming request to Ollama (${model})`);
      
      const response = await fetch(`${baseUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model,
          prompt,
          temperature: requestOptions.temperature,
          max_tokens: requestOptions.max_tokens,
          stop: requestOptions.stop,
          stream: true
        })
      });
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
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
            logger.error(`Error parsing Ollama stream: ${error.message}`);
          }
        }
      }
      
      // Final completion if we haven't already called it
      onComplete({
        model,
        tokens: totalTokens
      });
    } catch (error) {
      logger.error(`Error streaming response from Ollama: ${error.message}`);
      onToken(`Error: Failed to stream response from Ollama. ${error.message}`);
      onComplete({ tokens: 0, error: true });
    }
  }
  
  /**
   * Get model information
   * 
   * @returns {Promise<Object>} Model information
   */
  async function getModelInfo() {
    try {
      const response = await fetch(`${baseUrl}/api/show`, {
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
        family: data.modelfile?.parameter || 'unknown'
      };
    } catch (error) {
      logger.error(`Error getting model info: ${error.message}`);
      return {
        name: model,
        parameters: {},
        size: 'unknown',
        family: 'unknown'
      };
    }
  }
  
  /**
   * Get connection status
   * 
   * @returns {boolean} Connection status
   */
  function getConnectionStatus() {
    return isConnected;
  }
  
  // Initialize on creation
  initialize().catch(error => {
    logger.error(`Failed to initialize Ollama client: ${error.message}`);
  });
  
  // Return the client interface
  return {
    generateResponse,
    streamResponse,
    getModelInfo,
    getConnectionStatus,
    reconnect: initialize
  };
}

module.exports = {
  createOllamaClient
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
    // Setup logging based on configuration - Specify TUI mode to disable console logging
    setupLogging(config.logging, true); // Pass 'true' to indicate we're in TUI mode
    
    logger.info('Starting SynthBot...');
    logger.debug('Configuration loaded', { config });
    
    // Initial project scan
    logger.info('Scanning project files...');
    const projectFiles = await scanProjectFiles(config.projectRoot, {
      exclude: config.context.excludeDirs.concat(config.context.excludeFiles),
      prioritize: config.context.priorityFiles
    });
    
    logger.info(`Found ${projectFiles.length} files in project`);
    
    // Create API client
    let apiClient;
    
    // Check if offline mode is enabled
    // Check if offline mode is enabled
    if (config.offline) {
      logger.info('Running in offline mode - no LLM connection will be attempted');
      
      // Create a dummy client for offline mode
      apiClient = {
        generateResponse: async (prompt) => ({ 
          text: "Running in offline mode. LLM services are not available.\n\nYour prompt was:\n" + prompt,
          tokens: 0 
        }),
        streamResponse: async (prompt, onToken, onComplete) => {
          onToken("Running in offline mode. LLM services are not available.");
          onComplete({ tokens: 0 });
        },
        getConnectionStatus: () => false,
        getModelInfo: async () => ({ name: 'offline', parameters: {} })
      };
    } else {
      // Normal mode - try to connect to LLM service
      // Check if we should use Ollama
      const useOllama = config.llm.api === 'ollama';
      
      logger.info(`Connecting to ${useOllama ? 'Ollama' : 'distributed LLM'} service...`);
      
      try {
        if (useOllama) {
          // Import the Ollama client
          const { createOllamaClient } = require('./api');
          
          apiClient = createOllamaClient({
            host: '127.0.0.1',
            port: config.llm.coordinatorPort,
            model: 'deepseek-r1:1.5b',  // Use the model you have installed
            temperature: config.llm.temperature
          });
          
          logger.info(`Attempting to connect to Ollama at ${config.llm.coordinatorHost}:${config.llm.coordinatorPort}`);
        } else {
          apiClient = createClient({
            host: config.llm.coordinatorHost,
            port: config.llm.coordinatorPort,
            model: config.llm.model,
            temperature: config.llm.temperature,
            api: config.llm.api
          });
          
          logger.info(`Attempting to connect to ${config.llm.api} at ${config.llm.coordinatorHost}:${config.llm.coordinatorPort}`);
        }
      } catch (error) {
        logger.error('Failed to create API client', { error });
        
        // Create a dummy client that returns error messages
        apiClient = {
          generateResponse: async () => ({ 
            text: "ERROR: Could not connect to LLM service. Please check if Ollama is running with 'ollama serve'\n\nTry running with --offline flag if you want to explore the UI without LLM connection.",
            tokens: 0 
          }),
          streamResponse: async (prompt, onToken, onComplete) => {
            onToken("ERROR: Could not connect to LLM service. Please check if Ollama is running with 'ollama serve'");
            onComplete({ tokens: 0 });
          },
          getConnectionStatus: () => false,
          getModelInfo: async () => ({ name: 'disconnected', parameters: {} })
        };
      }
    }
    
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
    
    // Start the TUI
    app.start();

    // Save statusBar reference for other components to use
    app.screen.statusBar = app.statusBar;
    
    // Handle application shutdown
    function shutdown() {
      logger.info('Shutting down FrankCode...');
      app.destroy();
      process.exit(0);
    }
    
    // Handle signals
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
    
    logger.info('SynthBot started successfully');
    
  } catch (error) {
    logger.error('Failed to start FrankCode', { error });
    console.error('Failed to start FrankCode:', error.message);
    process.exit(1);
  }
}

// IMPORTANT: Export the startApp function
module.exports = {
  startApp
};
```

# src/server/server-setup.js

```js
/**
 * FrankCode Server Setup
 * 
 * This script sets up an API server for FrankCode and creates a public URL using ngrok.
 * It allows remote access to the LLM service and provides a proxy for Ollama.
 */

const express = require('express');
const cors = require('cors');
const ngrok = require('ngrok');
const fetch = require('node-fetch');
const fs = require('fs').promises;
const path = require('path');
const dotenv = require('dotenv');
const { logger } = require('../utils/logger');

// Load environment variables from parent directories
function loadEnvFromParentDirs() {
  let currentDir = process.cwd();
  let rootDir = path.parse(currentDir).root;
  
  // Try to find .env in current and parent directories
  while (currentDir !== rootDir) {
    const envPath = path.join(currentDir, '.env');
    try {
      if (fs.existsSync(envPath)) {
        logger.info(`Found .env file at ${envPath}`);
        dotenv.config({ path: envPath });
        return true;
      }
    } catch (error) {
      // Ignore errors and continue searching
    }
    
    // Move up one directory
    currentDir = path.dirname(currentDir);
  }
  
  // Try root directory as a last resort
  const rootEnvPath = path.join(rootDir, '.env');
  try {
    if (fs.existsSync(rootEnvPath)) {
      logger.info(`Found .env file at ${rootEnvPath}`);
      dotenv.config({ path: rootEnvPath });
      return true;
    }
  } catch (error) {
    // Ignore errors
  }
  
  logger.warn('No .env file found in any parent directory');
  return false;
}

// Load environment variables
loadEnvFromParentDirs();

// Ollama API configuration
const ollamaHost = process.env.OLLAMA_HOST || 'localhost';
const ollamaPort = process.env.OLLAMA_PORT || 11434;
const ollamaUrl = `http://${ollamaHost}:${ollamaPort}`;

/**
 * Start the FrankCode server
 * 
 * @param {Object} options Configuration options
 * @param {number} options.port Port for the server
 * @param {boolean} options.enableNgrok Whether to enable ngrok
 * @returns {Promise<Object>} Server info
 */
async function startServer(options = {}) {
  const {
    port = 3000,
    enableNgrok = true
  } = options;
  
  // Create Express app
  const app = express();
  
  // Configure middleware
  app.use(cors());
  app.use(express.json());
  
  // Health check endpoint
  app.get('/health', (req, res) => {
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString()
    });
  });
  
  // Ollama proxy endpoint
  app.post('/api/generate', async (req, res) => {
    try {
      const response = await fetch(`${ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(req.body)
      });
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      logger.error(`Error proxying to Ollama: ${error.message}`);
      res.status(500).json({
        error: 'Failed to proxy request to Ollama',
        message: error.message
      });
    }
  });
  
  // Ollama streaming endpoint
  app.post('/api/generate/stream', async (req, res) => {
    try {
      // Set up streaming response
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      // Forward request to Ollama
      const ollamaReq = await fetch(`${ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...req.body,
          stream: true
        })
      });
      
      if (!ollamaReq.ok) {
        throw new Error(`Ollama API returned status ${ollamaReq.status}`);
      }
      
      // Stream response from Ollama to client
      const reader = ollamaReq.body.getReader();
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        // Forward chunk to client
        res.write(value);
        res.flush();
      }
      
      res.end();
    } catch (error) {
      logger.error(`Error streaming from Ollama: ${error.message}`);
      res.status(500).json({
        error: 'Failed to stream from Ollama',
        message: error.message
      });
    }
  });
  
  // Ollama model list endpoint
  app.get('/api/models', async (req, res) => {
    try {
      const response = await fetch(`${ollamaUrl}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      logger.error(`Error getting models from Ollama: ${error.message}`);
      res.status(500).json({
        error: 'Failed to get models from Ollama',
        message: error.message
      });
    }
  });
  
  // Start the server
  return new Promise((resolve, reject) => {
    try {
      const server = app.listen(port, async () => {
        logger.info(`FrankCode server running on http://localhost:${port}`);
        
        let ngrokUrl = null;
        
        // Set up ngrok if enabled
        if (enableNgrok) {
          try {
            // Check for ngrok auth token
            const ngrokToken = process.env.NGROK_AUTH_TOKEN;
            
            if (!ngrokToken) {
              logger.warn('NGROK_AUTH_TOKEN not found in environment variables');
              logger.warn('Ngrok may not work correctly without authentication');
              logger.warn('Try specifying the token directly when starting the server:');
              logger.warn('  NGROK_AUTH_TOKEN=your_token node bin/start-server.js');
            } else {
              // Configure ngrok with auth token
              logger.info('Found NGROK_AUTH_TOKEN, authenticating...');
              await ngrok.authtoken(ngrokToken);
              logger.info('Ngrok authenticated successfully');
            }
            
            // Start ngrok tunnel
            ngrokUrl = await ngrok.connect({
              addr: port,
              region: 'us'
            });
            
            logger.info(`Ngrok URL: ${ngrokUrl}`);
            
            // Save ngrok URL to file for future reference
            const ngrokFilePath = path.join(process.cwd(), 'ngrok_url.txt');
            await fs.writeFile(ngrokFilePath, ngrokUrl);
            logger.info(`Ngrok URL saved to ${ngrokFilePath}`);
          } catch (ngrokError) {
            logger.error(`Failed to start ngrok: ${ngrokError.message}`);
            logger.info('Continuing without ngrok...');
          }
        }
        
        resolve({
          server,
          localUrl: `http://localhost:${port}`,
          ngrokUrl
        });
      });
      
      // Handle server errors
      server.on('error', (error) => {
        logger.error(`Server error: ${error.message}`);
        reject(error);
      });
    } catch (error) {
      logger.error(`Failed to start server: ${error.message}`);
      reject(error);
    }
  });
}

/**
 * Stop the server and clean up resources
 * 
 * @param {Object} serverInfo Server info object
 */
async function stopServer(serverInfo) {
  try {
    // Close the server
    if (serverInfo && serverInfo.server) {
      await new Promise((resolve) => {
        serverInfo.server.close(resolve);
      });
      logger.info('Server closed');
    }
    
    // Disconnect ngrok
    if (serverInfo && serverInfo.ngrokUrl) {
      await ngrok.disconnect();
      await ngrok.kill();
      logger.info('Ngrok disconnected');
    }
  } catch (error) {
    logger.error(`Error stopping server: ${error.message}`);
  }
}

module.exports = {
  startServer,
  stopServer
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
const { createAgentCommandProcessor } = require('../agent/agentUtils');

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
  // Create screen object
  const screen = blessed.screen({
    smartCSR: true,
    title: 'FrankCode',
    fullUnicode: true,
    dockBorders: true,
    autoPadding: true,
    sendFocus: false,
    mouseEnabled: true,
    useMouse: true
  });

  // Store config and project root in screen for access by components
  screen.config = config;
  screen.cwd = projectRoot;
  
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
    // Add these settings for smoother scrolling
    mouse: true,
    scrollable: true,
    alwaysScroll: true,
    scrollbar: true,
    keys: true,
    vi: false,  // Turn off vi mode to enable normal selection
    inputOnFocus: false,
    // Increase scroll amount for smoother experience
    scrollAmount: 3,
    // Lower scroll time for smoother animation
    scrollSpeed: 10,
    // Allow mousewheel scrolling
    wheelBehavior: 'scroll',
    // Improved border style
    border: {
      type: 'line',
      fg: 'blue'
    },
    // Better padding for content
    padding: {
      left: 1,
      right: 1
    }
  });
  
  // Add additional key bindings for scrolling
  conversationPanel.key(['pageup'], () => {
    conversationPanel.setScroll(conversationPanel.getScroll() - conversationPanel.height + 2);
    screen.render();
  });
  
  conversationPanel.key(['pagedown'], () => {
    conversationPanel.setScroll(conversationPanel.getScroll() + conversationPanel.height - 2);
    screen.render();
  });
  
  // Allow using up/down arrow keys to scroll when focus is on conversation panel
  conversationPanel.key(['up'], () => {
    conversationPanel.scroll(-3); // Scroll up 3 lines
    screen.render();
  });
  
  conversationPanel.key(['down'], () => {
    conversationPanel.scroll(3); // Scroll down 3 lines
    screen.render();
  });
  
  // Add a key binding to toggle focus to the conversation panel for scrolling
  screen.key(['C-f'], () => {
    conversationPanel.focus();
    statusBarController.update('Scroll mode active. Press Tab to return to input.');
    screen.render();
  });
  
  // Update the tab key handler to cycle through all elements
  screen.key(['tab'], () => {
    if (screen.focused === inputBox) {
      fileTreePanel.focus();
    } else if (screen.focused === fileTreePanel) {
      conversationPanel.focus();
    } else {
      inputBox.focus();
    }
    screen.render();
  });
  
  // Add a mouse handler for better wheel behavior
  conversationPanel.on('wheeldown', () => {
    conversationPanel.scroll(3); // Scroll down 3 lines
    screen.render();
  });
  
  conversationPanel.on('wheelup', () => {
    conversationPanel.scroll(-3); // Scroll up 3 lines
    screen.render();
  });
  
  const inputBox = grid.set(8, 2, 2, 10, blessed.textarea, {
    label: 'Command',
    inputOnFocus: true,
    padding: {
      top: 1,
      left: 2
    },
    style: {
      fg: 'white',
      bg: 'black',
      focus: {
        fg: 'white',
        bg: 'black',
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

  try {
    const agentCommandProcessor = createAgentCommandProcessor({
      agent,
      llm: apiClient,
      screen,
      outputRenderer
    });
  // Add to input handler
  inputHandler.agentCommandProcessor = agentCommandProcessor;
    
  // Add help for agent commands
  if (agentCommandProcessor) {
    screen.key(['F1'], () => {
      const exampleCommands = agentCommandProcessor.getExampleCommands();
      
      outputRenderer.addSystemMessage('\n📚 Agent Command Examples:');
      exampleCommands.forEach(example => {
        outputRenderer.addSystemMessage(`• ${example}`);
      });
      
      outputRenderer.addSystemMessage('\nTry using these command patterns or press F1 again for more examples.');
      screen.render();
    });
  }
} catch (error) {
  logger.error('Failed to initialize agent command processor:', error);
  // Continue without agent capabilities
}
  const renderInterval = setInterval(() => {
    screen.render();
  }, 100);
  
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
// Return the application object
return {
  screen,
  statusBar: statusBarController,
  start: () => {
    // Initial rendering
    screen.render();
    
    // Welcome message
    outputRenderer.addSystemMessage('Welcome to FrankCode! Type your question or command below.');
    statusBarController.update('Ready');
    
    // Other initialization...
  },
  destroy: () => {
    // Cleanup code...
    screen.destroy();
  }
};
}

module.exports = {
createApp
};
```

# src/tui/confirmation.js

```js
/**
 * Confirmation UI Component for FrankCode Agent
 * 
 * This module provides UI components for confirming file operations
 * with visual diffs and interactive prompts.
 */

const blessed = require('blessed');
const chalk = require('chalk');
const { logger } = require('../utils/logger');

/**
 * Create a confirmation dialog for file operations
 * 
 * @param {Object} screen The blessed screen object
 * @param {Object} operation The operation result (from fileOperations.js)
 * @returns {Promise<string>} User's choice ('yes', 'no', 'yes-to-all', 'custom')
 */
function createConfirmationDialog(screen, operation) {
  return new Promise((resolve) => {
    // Create a box for the confirmation dialog
    const box = blessed.box({
      top: 'center',
      left: 'center',
      width: '80%',
      height: '80%',
      border: {
        type: 'line',
        fg: 'blue'
      },
      tags: true,
      keys: true,
      vi: true,
      mouse: true,
      scrollable: true,
      alwaysScroll: true,
      scrollbar: {
        ch: ' ',
        bg: 'blue'
      }
    });
    
    // Create title based on operation type
    let title;
    switch (operation.operation) {
      case 'update':
        title = `{bold}Confirm Update: ${operation.filePath}{/bold}`;
        break;
      case 'create':
        title = `{bold}Confirm Creation: ${operation.filePath}{/bold}`;
        break;
      case 'delete':
        title = `{bold}Confirm Deletion: ${operation.filePath}{/bold}`;
        break;
      default:
        title = `{bold}Confirm Operation: ${operation.operation}{/bold}`;
    }
    
    // Add title
    box.setLine(0, title);
    box.setLine(1, '');
    
    // Add operation details
    let lineIndex = 2;
    
    if (operation.operation === 'update') {
      box.setLine(lineIndex++, `{yellow-fg}Changes: ${operation.additions} additions, ${operation.removals} removals{/yellow-fg}`);
      box.setLine(lineIndex++, '');
      
      // Add diff display
      if (operation.diffDisplay) {
        // Convert chalk colors to blessed tags for the terminal UI
        const diffDisplay = operation.diffDisplay
          .replace(/\u001b\[32m/g, '{green-fg}') // green
          .replace(/\u001b\[31m/g, '{red-fg}')   // red
          .replace(/\u001b\[90m/g, '{grey-fg}')  // grey
          .replace(/\u001b\[39m/g, '{/}');       // reset
          
        diffDisplay.split('\n').forEach((line) => {
          box.setLine(lineIndex++, line);
        });
      } else {
        // Simple diff
        const oldLines = operation.originalContent.split('\n');
        const newLines = operation.newContent.split('\n');
        
        for (let i = 0; i < Math.max(oldLines.length, newLines.length); i++) {
          const oldLine = oldLines[i] !== undefined ? oldLines[i] : '';
          const newLine = newLines[i] !== undefined ? newLines[i] : '';
          
          if (oldLine !== newLine) {
            if (oldLine) {
              box.setLine(lineIndex++, `{red-fg}- ${i + 1} ${oldLine}{/red-fg}`);
            }
            if (newLine) {
              box.setLine(lineIndex++, `{green-fg}+ ${i + 1} ${newLine}{/green-fg}`);
            }
          } else {
            box.setLine(lineIndex++, `  ${i + 1} ${oldLine}`);
          }
        }
      }
    } else if (operation.operation === 'create') {
      box.setLine(lineIndex++, `{green-fg}New file: ${operation.filePath}{/green-fg}`);
      box.setLine(lineIndex++, '');
      
      // Show content
      const contentLines = operation.content.split('\n');
      contentLines.forEach((line, index) => {
        box.setLine(lineIndex++, `{green-fg}+ ${index + 1} ${line}{/green-fg}`);
      });
    } else if (operation.operation === 'delete') {
      box.setLine(lineIndex++, `{red-fg}Delete file: ${operation.filePath}{/red-fg}`);
      box.setLine(lineIndex++, '');
      
      // Show content being deleted
      const contentLines = operation.content.split('\n');
      contentLines.forEach((line, index) => {
        box.setLine(lineIndex++, `{red-fg}- ${index + 1} ${line}{/red-fg}`);
      });
    }
    
    // Add empty line
    box.setLine(lineIndex++, '');
    
    // Add buttons
    const buttonsLine = lineIndex++;
    box.setLine(buttonsLine, '{center}{bold}[Y]{/bold}es | {bold}[N]{/bold}o | {bold}[A]{/bold}ll | {bold}[C]{/bold}ustom | {bold}[Esc]{/bold} Cancel{/center}');
    
    // Add the box to the screen
    screen.append(box);
    
    // Focus the box
    box.focus();
    
    // Handle key events
    box.key(['y', 'Y'], () => {
      cleanup();
      resolve('yes');
    });
    
    box.key(['n', 'N'], () => {
      cleanup();
      resolve('no');
    });
    
    box.key(['a', 'A'], () => {
      cleanup();
      resolve('yes-to-all');
    });
    
    box.key(['c', 'C'], () => {
      cleanup();
      resolve('custom');
    });
    
    box.key(['escape'], () => {
      cleanup();
      resolve('cancel');
    });
    
    // Render the screen
    screen.render();
    
    // Function to clean up the dialog
    function cleanup() {
      screen.remove(box);
      screen.render();
    }
  });
}

/**
 * Create a custom editor for modifying file content
 * 
 * @param {Object} screen The blessed screen object
 * @param {Object} operation The operation result (from fileOperations.js)
 * @returns {Promise<string>} Modified content or null if cancelled
 */
function createCustomEditor(screen, operation) {
  return new Promise((resolve) => {
    // Create a box for the editor
    const box = blessed.textarea({
      top: 'center',
      left: 'center',
      width: '90%',
      height: '90%',
      border: {
        type: 'line',
        fg: 'blue'
      },
      inputOnFocus: true,
      keys: true,
      vi: true,
      mouse: true,
      scrollable: true,
      alwaysScroll: true,
      scrollbar: {
        ch: ' ',
        bg: 'blue'
      }
    });
    
    // Set initial content
    let initialContent;
    if (operation.operation === 'update') {
      initialContent = operation.newContent;
    } else if (operation.operation === 'create') {
      initialContent = operation.content;
    } else {
      initialContent = '';
    }
    
    box.setValue(initialContent);
    
    // Create help text
    const helpText = blessed.box({
      bottom: 0,
      left: 'center',
      width: '100%',
      height: 1,
      content: '{bold}Ctrl-S{/bold} Save | {bold}Esc{/bold} Cancel',
      tags: true
    });
    
    // Create title
    const title = blessed.box({
      top: 0,
      left: 'center',
      width: '100%',
      height: 1,
      content: `{bold}Editing: ${operation.filePath}{/bold}`,
      tags: true
    });
    
    // Add the components to the screen
    screen.append(box);
    screen.append(helpText);
    screen.append(title);
    
    // Focus the box
    box.focus();
    
    // Handle key events
    box.key(['C-s'], () => {
      const content = box.getValue();
      cleanup();
      resolve(content);
    });
    
    box.key(['escape'], () => {
      cleanup();
      resolve(null);
    });
    
    // Render the screen
    screen.render();
    
    // Function to clean up the editor
    function cleanup() {
      screen.remove(box);
      screen.remove(helpText);
      screen.remove(title);
      screen.render();
    }
  });
}

/**
 * Show a confirmation dialog for a file operation
 * 
 * @param {Object} screen The blessed screen object
 * @param {Object} operation The operation result
 * @returns {Promise<Object>} Confirmation result with user's choice
 */
async function confirmOperation(screen, operation) {
  try {
    // Log the confirmation request
    logger.debug(`Requesting confirmation for operation: ${operation.operation}`, { filePath: operation.filePath });
    
    // Check if operation is valid
    if (operation.status === 'error') {
      return {
        choice: 'error',
        operation,
        error: operation.error
      };
    }
    
    // Show confirmation dialog
    const choice = await createConfirmationDialog(screen, operation);
    
    // Handle custom editing
    let customContent = null;
    if (choice === 'custom') {
      customContent = await createCustomEditor(screen, operation);
      
      // If cancelled, return cancel
      if (customContent === null) {
        return {
          choice: 'cancel',
          operation
        };
      }
      
      // Update the operation with custom content
      if (operation.operation === 'update' || operation.operation === 'create') {
        operation.newContent = customContent;
        operation.content = customContent;
      }
    }
    
    // Return the result
    return {
      choice,
      operation,
      customContent
    };
  } catch (error) {
    logger.error(`Failed to confirm operation: ${operation.operation}`, { error });
    return {
      choice: 'error',
      operation,
      error: error.message
    };
  }
}

module.exports = {
  confirmOperation,
  createConfirmationDialog,
  createCustomEditor
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

const { logger } = require('../utils/logger');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);
const path = require('path');
const { createConversationManager } = require('../agent/conversationManager');

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
  
  // Agent command processor - will be set after initialization
  let agentCommandProcessor = null;
  
  // Initialize
  function init() {
    // Add event listener for keypress to ensure UI updates
    widget.on('keypress', function() {
      // Force a screen render after each key press
      screen.render();
    });

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
      widget.screen.render(); // Ensure UI updates
    });
  }
  /**
   * Clear the conversation history
   */
  async function clearConversation() {
    try {
      // Reset the agent's conversation history
      agent.reset();
      
      // Clear the output display
      outputRenderer.clear();
      
      // Display confirmation message
      outputRenderer.addSystemMessage('Conversation history cleared.');
      
      logger.info('Conversation history cleared by user');
    } catch (error) {
      logger.error('Failed to clear conversation', { error });
      outputRenderer.addErrorMessage(`Error clearing conversation: ${error.message}`);
    }
  }

  /**
   * Generate a summary of the conversation history
   */
  async function generateSummary() {
    try {
      // Get conversation history
      const history = agent.getConversationHistory();
      
      if (!history || history.length === 0) {
        outputRenderer.addSystemMessage('No conversation history to summarize.');
        return;
      }
      
      // Format the conversation
      const formattedHistory = history.map(msg => {
        // Clean up the content (remove thinking tags)
        const cleanContent = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
        return `${msg.role.toUpperCase()}: ${cleanContent.substring(0, 300)}${cleanContent.length > 300 ? '...' : ''}`;
      }).join('\n\n');
      
      // Create prompt for the LLM
      const prompt = `Create a concise summary of this conversation between a user and FrankCode (an AI coding assistant). Format it with clear sections for:

  1. Main topics and tasks discussed
  2. Problems solved or features implemented
  3. Files modified or discussed
  4. Next steps based on the conversation
  5. The most important achievements

  Use bold headers (**Header:**) and bullet points. Here's the conversation:

  ${formattedHistory}`;
      
      // Generate summary using the LLM
      outputRenderer.addSystemMessage('Generating conversation summary...');
      
      const response = await apiClient.generateResponse(prompt, {
        temperature: 0.3,
        maxTokens: 1024
      });
      
      return response.text;
    } catch (error) {
      logger.error('Failed to generate summary', { error });
      outputRenderer.addErrorMessage(`Error generating summary: ${error.message}`);
      return 'Failed to generate summary.';
    }
  }

  /**
   * Compact the conversation with a summary
   */
  async function compactConversation() {
    try {
      // Generate summary
      const summary = await generateSummary();
      
      if (!summary) {
        return;
      }
      
      // Reset the agent state
      agent.reset();
      
      // Add summary as context
      agent.addSystemContext(summary);
      
      // Clear the output display
      outputRenderer.clear();
      
      // Display the summary
      outputRenderer.addSystemMessage('***Session Summary***');
      
      // Format and display the summary
      const summaryLines = summary.split('\n');
      for (const line of summaryLines) {
        outputRenderer.addSystemMessage(line);
      }
      
      outputRenderer.addSystemMessage('\nConversation history has been compacted. The summary above has been retained in context.');
      
      logger.info('Conversation compacted with summary');
    } catch (error) {
      logger.error('Failed to compact conversation', { error });
      outputRenderer.addErrorMessage(`Error compacting conversation: ${error.message}`);
    }
  }

  /**
   * Export the conversation to a file
   */
  async function exportConversation(filePath) {
    try {
      // Create default filename if none provided
      if (!filePath) {
        filePath = `conversation_${new Date().toISOString().replace(/[:.]/g, '-')}.md`;
      }
      
      // Get absolute path
      const fullPath = path.isAbsolute(filePath) ? filePath : path.join(screen.cwd, filePath);
      
      // Get conversation history
      const history = agent.getConversationHistory();
      
      if (!history || history.length === 0) {
        outputRenderer.addSystemMessage('No conversation history to export.');
        return;
      }
      
      // Generate summary
      const summary = await generateSummary();
      
      // Format the conversation (remove thinking tags)
      const formattedHistory = history.map(msg => {
        const cleanContent = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
        return `${msg.role.toUpperCase()}:\n${cleanContent}\n\n---\n\n`;
      }).join('');
      
      
       // Try to generate a summary, but handle errors gracefully
       let summaryText = "Failed to generate summary.";    try {
      if (apiClient && history.length > 1) {
        const formattedHistoryForSummary = history.map(msg => {
          const cleanContent = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
          return `${msg.role.toUpperCase()}: ${cleanContent.substring(0, 300)}${cleanContent.length > 300 ? '...' : ''}`;
        }).join('\n\n');
        
        const summaryPrompt = `Create a concise summary of this conversation between a user and FrankCode (an AI coding assistant). Format it with clear sections for:

1. Main topics and tasks discussed
2. Problems solved or features implemented
3. Files modified or discussed
4. Next steps based on the conversation
5. The most important achievements

Use bold headers (**Header:**) and bullet points. Here's the conversation:

${formattedHistoryForSummary}`;
        
        outputRenderer.addSystemMessage('Generating conversation summary...');
        
        const response = await apiClient.generateResponse(summaryPrompt, {
          temperature: 0.3,
          maxTokens: 1024
        });
        
        if (response && response.text) {
          summary = response.text;
        }
      }
    } catch (error) {
      logger.error('Failed to generate summary', { error });
      outputRenderer.addErrorMessage(`Error generating summary: ${error.message}`);
      // Continue with the export even if summary generation fails
    }
    
    // Combine summary and history
    const content = `# Conversation Summary\n\n${summary}\n\n# Full Conversation\n\n${formattedHistory}`;

      
      // Ensure directory exists
      const dirPath = path.dirname(fullPath);
    try {
      // Use fs directly without promises (for compatibility)
      const fs = require('fs');
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
      }
      
      // Write to file using synchronous method (more reliable)
      fs.writeFileSync(fullPath, content, 'utf8');
      
      outputRenderer.addSystemMessage(`Conversation exported to ${fullPath}`);
      logger.info(`Conversation exported to ${fullPath}`);
      return true;
    } catch (dirError) {
      logger.error(`Failed to create directory: ${dirError.message}`);
      
      // Try writing to the current directory as fallback
      try {
        const fallbackPath = path.join('.', path.basename(filePath));
        fs.writeFileSync(fallbackPath, content, 'utf8');
        outputRenderer.addSystemMessage(`Conversation exported to ${fallbackPath} (fallback location)`);
        logger.info(`Conversation exported to ${fallbackPath} (fallback)`);
        return true;
      } catch (fallbackError) {
        throw new Error(`Failed to write to fallback location: ${fallbackError.message}`);
      }
    }
  } catch (error) {
    logger.error('Failed to export conversation', { error });
    outputRenderer.addErrorMessage(`Error exporting conversation: ${error.message}`);
    return false;
  }
}
  
  /**
   * Process user input
   * 
   * @param {string} input User input
   */
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
      
      // Check if this might be an agent task
      if (agentCommandProcessor && agentCommandProcessor.isPotentialAgentTask(input)) {
        const wasHandled = await agentCommandProcessor.processCommand(input);
        if (wasHandled) {
          return;
        }
      }
      
      try {
        // Send to agent for processing
        const response = await agent.processMessage(input);
        
        // Display assistant response
        if (response && response.text) {
          outputRenderer.addAssistantMessage(response.text);
          
          // Check for file modifications
          if (response.fileModifications && response.fileModifications.length > 0) {
            await handleFileModifications(response.fileModifications);
          }
          
          // Check if this looks like a potential agent task that wasn't recognized
          const suggestion = agentCommandProcessor ? agentCommandProcessor.suggestAgentCapability(input) : null;
          if (suggestion) {
            outputRenderer.addSystemMessage(`💡 ${suggestion}`);
            
            // Show examples occasionally (20% chance)
            if (Math.random() < 0.2) {
              const examples = agentCommandProcessor.getExampleCommands();
              const randomExample = examples[Math.floor(Math.random() * examples.length)];
              outputRenderer.addSystemMessage(`Example: "${randomExample}"`);
            }
          }
        } else {
          outputRenderer.addErrorMessage('No response received from the agent.');
        }
      } catch (processingError) {
        // Handle specific agent processing errors in a cleaner way
        logger.error('Agent processing error', { processingError });
        
        let errorMessage = processingError.message || 'Unknown error';
        
        // Check for common connection errors
        if (errorMessage.includes('ECONNREFUSED')) {
          outputRenderer.addErrorMessage(`Connection to Ollama failed. Try running with --offline flag or make sure Ollama is running with 'ollama serve'`);
        } else {
          outputRenderer.addErrorMessage(`Error: ${errorMessage}`);
        }
        
        // Force render to recover the UI
        widget.screen.render();
      }
    } catch (error) {
      logger.error('Failed to process input', { error });
      outputRenderer.addErrorMessage(`Error processing input: ${error.message}`);
      
      // Force render to recover the UI
      widget.screen.render();
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
          await clearConversation();
          break;
        
        case 'compact':
          await compactConversation();
          break;
        
        case 'export':
          await exportConversation(args[0]);
          break;
        case 'exit':
          process.exit(0);
          break;
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
          
        case 'ls':
          await executeShellCommand('ls -la');
          break;
          
        case 'pwd':
          await executeShellCommand('pwd');
          break;
          
        case 'models':
          await listAndSelectModels();
          break;
          
        case 'selectmodel':
          if (!args[0]) {
            outputRenderer.addErrorMessage('Please specify a model name or number');
            return;
          }
          await selectModel(args[0]);
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
          
        case 'offline':
          setOfflineMode(true);
          break;
          
        case 'online':
          setOfflineMode(false);
          break;
          case 'plan':

        case 'execute':
          if (agentEnhancer) {
            if (!args[0]) {
              outputRenderer.addErrorMessage('Please specify a task to execute');
              return;
            }
            const task = args.join(' ');
            await agentEnhancer.executeTask(task);
          } else {
            outputRenderer.addErrorMessage('Task execution not available');
          }
          break;

        case 'code':
          if (agentEnhancer) {
            if (args.length < 2) {
              outputRenderer.addErrorMessage('Usage: /code language filePath');
              return;
            }
            const [language, filePath] = args;
            const content = await agent.generateCode(language);
            if (content) {
              await agentEnhancer.createFile(filePath, content);
            }
          } else {
            outputRenderer.addErrorMessage('Code generation not available');
          }
          break;

        case 'run':
          if (agentEnhancer) {
            if (!args[0]) {
              outputRenderer.addErrorMessage('Please specify a code file to run');
              return;
            }
            const filePath = args[0];
            const language = path.extname(filePath).substring(1); // Get extension without dot
            const content = await readFile(filePath);
            if (content) {
              await agentEnhancer.executeCode(content, language);
            }
          } else {
            outputRenderer.addErrorMessage('Code execution not available');
          }
          break;

          case 'agent':
            if (args.length > 0) {
              const task = args.join(' ');
              if (agentCommandProcessor) {
                await agentCommandProcessor.executeTask(task);
              } else {
                outputRenderer.addErrorMessage('Agent command processor not initialized');
              }
            } else {
              outputRenderer.addSystemMessage('Please provide a task for the agent');
            }
            break;
        
        default:
          outputRenderer.addSystemMessage(`Unknown command: ${cmd}. Type /help for available commands.`);
      }
      
      // Force a render after processing command
      widget.screen.render();
    } catch (error) {
      logger.error('Failed to process command', { error });
      outputRenderer.addErrorMessage(`Error processing command: ${error.message}`);
      widget.screen.render();
    }
  }
  
  /**
   * Select a model to use
   * 
   * @param {string} modelIdentifier Model name or number
   */
  async function selectModel(modelIdentifier) {
    try {
      // Get models from Ollama
      const fetch = require('node-fetch');
      const host = 'localhost';
      const port = 11434;
      
      const response = await fetch(`http://${host}:${port}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      const models = data.models || [];
      
      if (models.length === 0) {
        outputRenderer.addSystemMessage('No models found');
        return;
      }
      
      // Find the model by name or index
      let selectedModel;
      
      if (!isNaN(parseInt(modelIdentifier))) {
        // Select by index
        const index = parseInt(modelIdentifier) - 1;
        if (index >= 0 && index < models.length) {
          selectedModel = models[index];
        }
      } else {
        // Select by name
        selectedModel = models.find(m => m.name === modelIdentifier);
      }
      
      if (!selectedModel) {
        outputRenderer.addErrorMessage(`Model "${modelIdentifier}" not found`);
        return;
      }
      
      // Update the apiClient
      const model = selectedModel.name;
      
      // Try to connect with new model
      outputRenderer.addSystemMessage(`Switching to model: ${model}`);
      
      try {
        // Create new client with selected model
        const { createClient } = require('../api');
        const newClient = createClient({
          host,
          port,
          model,
          temperature: 0.7,
          api: 'ollama'
        });
        
        // Replace the old client in the agent
        agent.setApiClient(newClient);
        
        outputRenderer.addSystemMessage(`Successfully switched to model: ${model}`);
        
        // Update status bar without using screen reference
        if (widget && widget.screen && widget.screen.statusBar) {
          widget.screen.statusBar.update(`Model: ${model}`);
        }
      } catch (error) {
        outputRenderer.addErrorMessage(`Failed to switch model: ${error.message}`);
      }
    } catch (error) {
      outputRenderer.addErrorMessage(`Failed to fetch models: ${error.message}`);
    }
  }
  
  /**
   * List and select models from Ollama
   */
  async function listAndSelectModels() {
    try {
      outputRenderer.addSystemMessage('Fetching available models from Ollama...');
      
      // Get models from Ollama
      const fetch = require('node-fetch');
      const host = 'localhost';
      const port = 11434;
      
      const response = await fetch(`http://${host}:${port}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      const models = data.models || [];
      
      if (models.length === 0) {
        outputRenderer.addSystemMessage('No models found. Make sure Ollama is running with `ollama serve`');
        return;
      }
      
      // Display available models
      outputRenderer.addSystemMessage(`Found ${models.length} models:`);
      models.forEach((model, index) => {
        const size = model.size ? `(${Math.round(model.size / (1024 * 1024))} MB)` : '(unknown size)';
        outputRenderer.addSystemMessage(`${index + 1}. ${model.name} ${size}`);
      });
      
      // Prompt to select a model
      outputRenderer.addSystemMessage('Type /selectmodel <number> or /selectmodel <name> to select a model');
    } catch (error) {
      outputRenderer.addErrorMessage(`Failed to fetch models: ${error.message}`);
      outputRenderer.addSystemMessage('Make sure Ollama is running with `ollama serve`');
    }
  }
  
  /**
   * Show help text
   */
  function showHelp() {
    const helpText = `
  Available commands:
    /help             - Show this help text
    /clear            - Clear the conversation history
    /compact          - Compact conversation history into a summary
    /export [file]    - Export conversation with summary to a file    /exit, /quit      - Exit the application
    /refresh          - Refresh the file tree
    /plan task        - Break down a task into smaller steps
    /execute task     - Execute a task with step-by-step guidance
    /code lang file   - Generate code in specified language and save to file
    /run file         - Run a code file and show output
    /exec <command>   - Execute a shell command
    /shell <command>  - Same as /exec
    /ls               - List files (shortcut for /exec ls)
    /pwd              - Show current directory (shortcut for /exec pwd)
    /models           - List available Ollama models
    /selectmodel <n>  - Select a model by number or name
    /offline          - Switch to offline mode (no LLM)
    /online           - Switch to online mode (try connecting to LLM)
    /load <file>      - Load a file into context
    /save <file>      - Save the conversation to a file
    /reset            - Reset agent context and conversation
    /yes, /y          - Approve the current file modification
    /no, /n           - Reject the current file modification
    /yesall, /ya      - Approve all pending file modifications
    /noall, /na       - Reject all pending file modifications
  
  Keyboard shortcuts:
    Ctrl+C            - Exit application
    Ctrl+R            - Refresh file tree
    Ctrl+L            - Clear conversation
    Ctrl+S            - Save conversation
    Ctrl+F            - Focus conversation panel (scroll mode)
    Tab               - Cycle focus between panels
    PageUp/PageDown   - Scroll conversation when focused
    Up/Down arrows    - Scroll conversation when focused
      `;
      
    outputRenderer.addSystemMessage(helpText);
  }

  /**
   * Set offline mode
   * 
   * @param {boolean} offline Whether to enable offline mode
   */
  async function setOfflineMode(offline) {
    try {
      const { createClient } = require('../api');
      
      if (offline) {
        // Create a dummy client for offline mode
        const offlineClient = {
          generateResponse: async (prompt) => ({ 
            text: "Running in offline mode. LLM services are not available.\n\nYour prompt was:\n" + prompt,
            tokens: 0 
          }),
          streamResponse: async (prompt, onToken, onComplete) => {
            onToken("Running in offline mode. LLM services are not available.");
            onComplete({ tokens: 0 });
          },
          getConnectionStatus: () => false,
          getModelInfo: async () => ({ name: 'offline', parameters: {} })
        };
        
        // Set in the agent
        agent.setApiClient(offlineClient);
        outputRenderer.addSystemMessage("Switched to offline mode. LLM services will not be used.");
      } else {
        // Try to create an online client
        try {
          const host = widget.screen.config?.llm?.coordinatorHost || 'localhost';
          const port = widget.screen.config?.llm?.coordinatorPort || 11434;
          const model = widget.screen.config?.llm?.model || 'llama2';
          
          const onlineClient = createClient({
            host,
            port,
            model,
            temperature: 0.7,
            api: 'ollama'
          });
          
          // Set in the agent
          agent.setApiClient(onlineClient);
          outputRenderer.addSystemMessage("Attempting to switch to online mode.");
        } catch (error) {
          outputRenderer.addErrorMessage(`Failed to go online: ${error.message}`);
        }
      }
    } catch (error) {
      outputRenderer.addErrorMessage(`Failed to change mode: ${error.message}`);
    }
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
      // Get conversation history
      const history = agent.getConversationHistory();
      
      // Format conversation
      const formatted = history.map(msg => {
        return `${msg.role.toUpperCase()}:\n${msg.content}\n\n`;
      }).join('---\n\n');
      
      // Resolve path
      const fullPath = path.resolve(path.join(screen.cwd, filePath));
      
      // Write to file
      const fs = require('fs').promises;
      await fs.writeFile(fullPath, formatted, 'utf8');
      
      outputRenderer.addSystemMessage(`Conversation saved to: ${filePath}`);
    } catch (error) {
      logger.error(`Failed to save conversation: ${filePath}`, { error });
      outputRenderer.addErrorMessage(`Error saving conversation: ${error.message}`);
    }
  }
  
  init();

  // Create conversation manager
  let conversationManager = null;
  try {
    conversationManager = createConversationManager({
      agent,
      outputRenderer,
      llm: apiClient
    });
    logger.debug('Conversation manager initialized');
  } catch (error) {
    logger.error('Failed to initialize conversation manager', { error });
  }
  let agentEnhancer = null;
  try {
    const { createAgentEnhancer } = require('../agent/agentEnhancer');
    agentEnhancer = createAgentEnhancer({
      agent,
      outputRenderer,
      apiClient
    });
    logger.debug('Agent enhancer initialized');
  } catch (error) {
    logger.error('Failed to initialize agent enhancer', { error });
  }
  

  // Return the input handler interface
  return {
    processInput,
    processCommand,
    handleFileModifications,
    agentCommandProcessor, // Allow setting from outside
    conversationManager    // Add this to make it accessible from outside if needed
  };
}

// Export the function
module.exports = {
  createInputHandler
};
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
    
    // Auto-scroll to bottom after adding new content
    widget.setScrollPerc(100);
    
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
    
    // Auto-scroll to bottom after adding new content
    widget.setScrollPerc(100);
    
    // Render the screen
    widget.screen.render();
  }
  
  /**
   * Add a system message to the conversation
   * 
   * @param {string} message Message text
   */
  function addSystemMessage(message) {
    // Format and add the message with nicer formatting
    widget.log(chalk.bold.yellow('┌─ SYSTEM ────────────────────────────────────'));
    
    // Split multi-line messages
    const lines = message.split('\n');
    lines.forEach(line => {
      widget.log(chalk.yellow('│ ') + line);
    });
    
    widget.log(chalk.bold.yellow('└────────────────────────────────────────────'));
    widget.log(''); // Empty line for spacing
    
    // Auto-scroll to bottom after adding new content
    widget.setScrollPerc(100);
    
    // Render the screen
    widget.screen.render();
  }
  
  /**
   * Add an error message to the conversation
   * 
   * @param {string} message Error message
   */
  function addErrorMessage(message) {
    // Format and add the message with nicer formatting
    widget.log(chalk.bold.red('┌─ ERROR ─────────────────────────────────────'));
    
    // Clean up any raw error text for better presentation
    const cleanedMessage = message
      .replace(/\{.*?\}/g, '') // Remove JSON objects
      .replace(/ECONNREFUSED/g, 'Connection Refused')
      .replace(/http:\/\/localhost:[0-9]+\/api\/generate/g, 'Ollama API')
      .replace(/connect ECONNREFUSED ::1:[0-9]+/g, 'Could not connect to Ollama API')
      .trim();
    
    // Split multi-line error messages
    const lines = cleanedMessage.split('\n');
    lines.forEach(line => {
      widget.log(chalk.red('│ ') + line);
    });
    
    widget.log(chalk.bold.red('└────────────────────────────────────────────'));
    widget.log(''); // Empty line for spacing
    
    // Auto-scroll to bottom after adding new content
    widget.setScrollPerc(100);
    
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
    // Create a nicer code block header
    widget.log(chalk.bold.cyan(`┌─ ${language || 'Code'} ${'─'.repeat(Math.max(0, 40 - language.length))}`));
    
    // Add the code with line numbers
    const lines = code.split('\n');
    lines.forEach((line, index) => {
      // Add line numbers for longer code blocks
      if (lines.length > 5) {
        const lineNum = String(index + 1).padStart(3, ' ');
        widget.log(chalk.dim(`│ ${lineNum} │`) + ' ' + line);
      } else {
        widget.log(chalk.dim('│') + ' ' + line);
      }
    });
    
    // Add a nice footer
    widget.log(chalk.bold.cyan(`└${'─'.repeat(44)}`));
    widget.log(''); // Empty line for spacing
    
    // Auto-scroll to bottom after adding new content
    widget.setScrollPerc(100);
    
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

# src/utils/embeddings.py

```py
#!/usr/bin/env python3
"""
Embeddings Generator using Jina Embeddings v3 or other Hugging Face models.

This script generates embeddings for text using Hugging Face models.
It's designed to be called from Node.js as a subprocess.
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Union

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import numpy as np
except ImportError:
    print("Required packages not found. Please install them with:")
    print("pip install sentence-transformers torch numpy")
    sys.exit(1)


def get_embedding(
    text: str,
    model_name: str = "jinaai/jina-embeddings-v3"
) -> List[float]:
    """
    Generate embeddings for text using the specified model.
    
    Args:
        text: The text to embed
        model_name: The Hugging Face model name to use
        
    Returns:
        A list of floats representing the embedding vector
    """
    try:
        # Load model
        model = SentenceTransformer(model_name)
        
        # Generate embedding
        embedding = model.encode(text, normalize_embeddings=True)
        
        # Convert to Python list
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().detach().numpy()
        
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}", file=sys.stderr)
        raise


def main():
    """Main function to handle CLI arguments and generate embeddings."""
    parser = argparse.ArgumentParser(description="Generate embeddings for text")
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--model", 
        default="jinaai/jina-embeddings-v3", 
        help="Hugging Face model name to use"
    )
    args = parser.parse_args()
    
    try:
        # Read input text
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Generate embedding
        embedding = get_embedding(text, args.model)
        
        # Save to output file
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"embedding": embedding}, f)
        
        print(f"Embedding saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
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

// Note: Order of imports is important to avoid circular dependencies
const { logger, setupLogging } = require('./logger');
const { formatTimestamp, formatFileSize, formatDuration, formatPercentage, truncate } = require('./formatters');
const { loadConfig, saveConfig } = require('./config');
const { scanProjectFiles, ensureDir, pathExists, getStats, getFileType } = require('./fileSystem');
const { createGitUtils } = require('./git');
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
// src/utils/logger.js - Complete rewrite to disable ALL console output

/**
 * Logger module
 * 
 * Provides logging capabilities for the application
 */

const winston = require('winston');
const fs = require('fs');
const path = require('path');

// Create logs directory if it doesn't exist
const logsDir = path.join(process.cwd(), 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir);
}

// Custom log levels
const levels = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3
};

// Create winston logger - NO CONSOLE TRANSPORT BY DEFAULT
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

// Store original console methods
const originalConsole = {
  log: console.log,
  warn: console.warn,
  error: console.error,
  info: console.info,
  debug: console.debug
};

/**
 * Set up logging based on configuration
 * 
 * @param {Object} config Logging configuration
 * @param {boolean} isTUI Whether running in TUI mode
 */
function setupLogging(config = {}, isTUI = false) {
  // Default configuration
  const {
    level = 'info',
    console = !isTUI, // Disable console in TUI mode
    file = true,
    filePath,
    maxSize = '10m',
    maxFiles = 5
  } = config;
  
  // Set log level
  logger.level = level;
  
  // Configure file transports
  if (file) {
    // Custom log path or default
    const logPath = filePath || logsDir;
    
    // Ensure log directory exists
    if (!fs.existsSync(logPath)) {
      fs.mkdirSync(logPath, { recursive: true });
    }
  }
  
  // Override console methods in TUI mode
  if (isTUI) {
    // Redirect ALL console output to files in TUI mode
    console.log = (...args) => {
      logger.info(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
    };
    
    console.warn = (...args) => {
      logger.warn(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
    };
    
    console.error = (...args) => {
      logger.error(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
    };
    
    console.info = (...args) => {
      logger.info(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
    };
    
    console.debug = (...args) => {
      logger.debug(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
    };
    
    // Remove all console transports from Winston
    logger.transports = logger.transports.filter(
      t => t.name !== 'console'
    );
  } else {
    // Restore console methods if not in TUI mode
    console.log = originalConsole.log;
    console.warn = originalConsole.warn;
    console.error = originalConsole.error;
    console.info = originalConsole.info;
    console.debug = originalConsole.debug;
    
    // Add console transport to Winston if enabled
    if (console) {
      // Remove existing console transports
      logger.transports = logger.transports.filter(
        t => t.name !== 'console'
      );
      
      // Add new console transport
      logger.add(new winston.transports.Console({
        format: winston.format.combine(
          winston.format.colorize(),
          winston.format.simple()
        )
      }));
    }
  }
  
  // Log initialization (to file only in TUI mode)
  logger.info('Logging initialized', { level, isTUI });
}

// Create a special startup function to immediately disable console output
function quietStartup() {
  // Immediately redirect console to file
  console.log = (...args) => {
    logger.info(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
  };
  
  console.warn = (...args) => {
    logger.warn(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
  };
  
  console.error = (...args) => {
    logger.error(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
  };
  
  console.info = (...args) => {
    logger.info(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
  };
  
  console.debug = (...args) => {
    logger.debug(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' '));
  };
  
  // Ensure no console transports
  logger.transports = logger.transports.filter(
    t => t.name !== 'console'
  );
  
  logger.info('Console output redirected to file for TUI mode');
}

module.exports = {
  logger,
  setupLogging,
  quietStartup
};
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

