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