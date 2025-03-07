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