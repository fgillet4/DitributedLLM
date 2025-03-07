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