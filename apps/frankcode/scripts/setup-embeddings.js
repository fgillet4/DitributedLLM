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