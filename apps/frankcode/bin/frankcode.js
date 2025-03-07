#!/usr/bin/env node

/**
 * FrankCode - Command Line Entry Point
 * 
 * Parses command line arguments and starts the application
 */

const { program } = require('commander');
const path = require('path');
const { startApp } = require('../src');
const { loadConfig, logger } = require('../src/utils');
const packageJson = require('../package.json');

// Configure the command line interface
program
  .name('frankcode')
  .description('Terminal-based coding agent powered by local LLMs')
  .version(packageJson.version);

// Run command
program
  .command('run')
  .description('Start FrankCode in the current directory')
  .option('-c, --config <path>', 'Path to configuration file')
  .option('-h, --host <host>', 'LLM coordinator host')
  .option('-p, --port <port>', 'LLM coordinator port')
  .option('-m, --model <model>', 'LLM model to use')
  .option('-t, --temperature <temp>', 'Temperature setting (0-1)')
  .option('--api <api>', 'API to use: "ollama" or "distributed"')
  .option('--max-tokens <tokens>', 'Maximum tokens for context')
  .option('--theme <theme>', 'UI theme')
  .option('-v, --verbose', 'Enable verbose logging')
  .option('-q, --quiet', 'Quiet mode (errors only)')
  .option('--log-level <level>', 'Log level (error, warn, info, debug)')
  .action(async (options) => {
    try {
      // Load configuration
      const config = await loadConfig({
        configPath: options.config,
        projectRoot: process.cwd(),
        ...options
      });
      
      // Start the application
      await startApp(config);
    } catch (error) {
      console.error('Failed to start FrankCode:', error.message);
      process.exit(1);
    }
  });

// Init command
program
  .command('init')
  .description('Initialize a FrankCode configuration file')
  .option('-f, --force', 'Overwrite existing configuration')
  .action(async (options) => {
    try {
      const fs = require('fs').promises;
      const { saveConfig, defaultConfig } = require('../src/utils/config');
      
      // Check if configuration file exists
      const configPath = path.join(process.cwd(), '.frankcoderc.js');
      let exists = false;
      
      try {
        await fs.access(configPath);
        exists = true;
      } catch {
        // File doesn't exist
      }
      
      if (exists && !options.force) {
        console.error('Configuration file already exists. Use --force to overwrite.');
        process.exit(1);
      }
      
      // Save configuration
      await saveConfig(defaultConfig, configPath);
      
      console.log(`FrankCode configuration created at ${configPath}`);
    } catch (error) {
      console.error('Failed to initialize configuration:', error.message);
      process.exit(1);
    }
  });

// Models command
program
  .command('models')
  .description('List available Ollama models')
  .option('-h, --host <host>', 'Ollama host (default: localhost)')
  .option('-p, --port <port>', 'Ollama port (default: 11434)')
  .action(async (options) => {
    try {
      const fetch = require('node-fetch');
      
      const host = options.host || 'localhost';
      const port = options.port || 11434;
      
      console.log(`Connecting to Ollama at ${host}:${port}...`);
      
      const response = await fetch(`http://${host}:${port}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`Ollama API returned status ${response.status}`);
      }
      
      const data = await response.json();
      const models = data.models || [];
      
      if (models.length === 0) {
        console.log('No models found.');
        return;
      }
      
      console.log('Available models:');
      
      models.forEach(model => {
        console.log(`- ${model.name} (${model.size || 'unknown size'})`);
      });
    } catch (error) {
      console.error('Failed to list models:', error.message);
      process.exit(1);
    }
  });

// Parse command line arguments
program.parse(process.argv);

// If no arguments, show help
if (!process.argv.slice(2).length) {
  program.outputHelp();
}