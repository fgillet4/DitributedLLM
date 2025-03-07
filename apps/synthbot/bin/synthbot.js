#!/usr/bin/env node

const { program } = require('commander');
const path = require('path');
const { startApp } = require('../src/index');
const { version } = require('../package.json');
const { loadConfig } = require('../src/utils/config');

program
  .name('synthbot')
  .description('Terminal-based coding agent powered by distributed LLMs')
  .version(version);

program
  .command('run')
  .description('Start SynthBot in the current directory')
  .option('-c, --config <path>', 'Path to configuration file')
  .option('--coordinator <host:port>', 'DistributedLLM coordinator address')
  .option('-v, --verbose', 'Enable verbose logging')
  .option('--model <model>', 'Specify LLM model to use')
  .option('--theme <theme>', 'UI theme (dark/light/system)')
  .action(async (options) => {
    // Load configuration
    const config = await loadConfig(options.config);
    
    // Override with command line options
    if (options.coordinator) {
      const [host, port] = options.coordinator.split(':');
      config.llm.coordinatorHost = host;
      config.llm.coordinatorPort = parseInt(port, 10);
    }
    
    if (options.model) {
      config.llm.model = options.model;
    }
    
    if (options.theme) {
      config.ui.theme = options.theme;
    }
    
    if (options.verbose) {
      config.logging.level = 'debug';
    }
    
    // Set project root to current directory
    config.projectRoot = process.cwd();
    
    // Start the application
    startApp(config);
  });

program
  .command('config')
  .description('Manage SynthBot configuration')
  .option('--init', 'Initialize default configuration')
  .option('--view', 'View current configuration')
  .option('--edit', 'Open configuration in editor')
  .action((options) => {
    if (options.init) {
      console.log('Initializing default configuration...');
      // Initialize config implementation
    } else if (options.view) {
      console.log('Current configuration:');
      // View config implementation
    } else if (options.edit) {
      console.log('Opening configuration in editor...');
      // Edit config implementation
    } else {
      console.log('Use one of the options: --init, --view, or --edit');
    }
  });

program.parse(process.argv);

// If no command is provided, show help
if (!process.argv.slice(2).length) {
  program.outputHelp();
}