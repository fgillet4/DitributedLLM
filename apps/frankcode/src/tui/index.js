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