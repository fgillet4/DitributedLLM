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