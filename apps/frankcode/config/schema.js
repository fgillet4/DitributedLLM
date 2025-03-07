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