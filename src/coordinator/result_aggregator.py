"""
Result Aggregator for the DistributedLLM system.

The ResultAggregator collects and combines model outputs from different workers,
handling the reassembly of sharded model outputs during distributed inference.
"""

import logging
import numpy as np
import threading
import time
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Collects and combines results from distributed computation tasks.
    Handles reassembly of model outputs that were sharded across workers.
    """
    
    def __init__(self):
        """Initialize the result aggregator."""
        self.results = {}
        self.task_groups = defaultdict(set)  # Group ID -> set of task IDs
        self.group_results = {}  # Group ID -> aggregated results
        self.group_status = {}  # Group ID -> status information
        self.lock = threading.RLock()  # Reentrant lock for thread safety
    
    def add_result(self, task):
        """
        Add a completed task result to the aggregator.
        
        Args:
            task: Completed task object
        """
        with self.lock:
            # Store individual task result
            self.results[task.id] = {
                'type': task.type,
                'result': task.result,
                'parameters': task.parameters,
                'completed_at': task.completed_at
            }
            
            # Check if this task belongs to a group
            group_id = task.parameters.get('group_id')
            if group_id:
                self.task_groups[group_id].add(task.id)
                
                # Check if all tasks in the group are complete
                if self._is_group_complete(group_id):
                    self._aggregate_group_results(group_id)
    
    def get_result(self, task_id):
        """
        Get the result for a specific task.
        
        Args:
            task_id: ID of the task
        
        Returns:
            Result data if available, None otherwise
        """
        with self.lock:
            if task_id in self.results:
                return self.results[task_id]['result']
            return None
    
    def get_group_result(self, group_id):
        """
        Get the aggregated result for a group of tasks.
        
        Args:
            group_id: ID of the task group
        
        Returns:
            Aggregated result if available, None otherwise
        """
        with self.lock:
            if group_id in self.group_results:
                return self.group_results[group_id]
            
            # If the group exists but results aren't aggregated yet,
            # check if we can aggregate now
            if group_id in self.task_groups:
                if self._is_group_complete(group_id):
                    self._aggregate_group_results(group_id)
                    return self.group_results.get(group_id)
            
            return None
    
    def get_group_status(self, group_id):
        """
        Get the status of a task group.
        
        Args:
            group_id: ID of the task group
        
        Returns:
            Status information including completion percentage
        """
        with self.lock:
            if group_id in self.group_status:
                return self.group_status[group_id]
            
            if group_id in self.task_groups:
                total_tasks = len(self.task_groups[group_id])
                completed_tasks = sum(1 for task_id in self.task_groups[group_id] if task_id in self.results)
                
                status = {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'percent_complete': (completed_tasks / max(total_tasks, 1)) * 100,
                    'is_complete': completed_tasks == total_tasks
                }
                
                self.group_status[group_id] = status
                return status
            
            return None
    
    def _is_group_complete(self, group_id):
        """
        Check if all tasks in a group are complete.
        
        Args:
            group_id: ID of the task group
        
        Returns:
            True if all tasks are complete, False otherwise
        """
        if group_id not in self.task_groups:
            return False
        
        return all(task_id in self.results for task_id in self.task_groups[group_id])
    
    def _aggregate_group_results(self, group_id):
        """
        Aggregate results for a completed task group.
        
        Args:
            group_id: ID of the task group
        """
        if not self._is_group_complete(group_id):
            return
        
        task_ids = self.task_groups[group_id]
        if not task_ids:
            return
        
        # Get the first task to determine the type
        first_task_id = next(iter(task_ids))
        task_type = self.results[first_task_id]['type']
        
        # Use different aggregation strategies based on task type
        if task_type == 'layer_computation':
            self._aggregate_layer_computation(group_id, task_ids)
        elif task_type == 'token_generation':
            self._aggregate_token_generation(group_id, task_ids)
        elif task_type == 'embedding':
            self._aggregate_embedding(group_id, task_ids)
        else:
            # Default aggregation: just collect all results in a list
            self.group_results[group_id] = {
                'type': task_type,
                'results': [self.results[task_id]['result'] for task_id in task_ids]
            }
        
        # Update group status
        self.group_status[group_id] = {
            'total_tasks': len(task_ids),
            'completed_tasks': len(task_ids),
            'percent_complete': 100.0,
            'is_complete': True,
            'aggregated': True
        }
        
        logger.info(f"Aggregated results for task group {group_id}")
    
    def _aggregate_layer_computation(self, group_id, task_ids):
        """
        Aggregate layer computation results.
        
        Args:
            group_id: ID of the task group
            task_ids: List of task IDs in the group
        """
        # Sort tasks by layer index
        tasks_by_layer = {}
        for task_id in task_ids:
            layer_index = self.results[task_id]['parameters'].get('layer_index', 0)
            tasks_by_layer[layer_index] = task_id
        
        # Collect outputs in order
        layer_outputs = []
        for layer_idx in sorted(tasks_by_layer.keys()):
            task_id = tasks_by_layer[layer_idx]
            layer_outputs.append({
                'layer_index': layer_idx,
                'output': self.results[task_id]['result']['output_data']
            })
        
        # Store aggregated result
        self.group_results[group_id] = {
            'type': 'layer_computation',
            'layer_outputs': layer_outputs,
            'aggregation_method': 'layer_sequence'
        }
    
    def _aggregate_token_generation(self, group_id, task_ids):
        """
        Aggregate token generation results.
        
        Args:
            group_id: ID of the task group
            task_ids: List of task IDs in the group
        """
        # For token generation, we need to concatenate in the right order
        # Get sequence indices
        tasks_by_sequence = {}
        for task_id in task_ids:
            seq_index = self.results[task_id]['parameters'].get('sequence_index', 0)
            tasks_by_sequence[seq_index] = task_id
        
        # Concatenate token sequences
        all_tokens = []
        for seq_idx in sorted(tasks_by_sequence.keys()):
            task_id = tasks_by_sequence[seq_idx]
            output_ids = self.results[task_id]['result']['output_ids']
            all_tokens.extend(output_ids)
        
        # Calculate total generation time
        total_time = sum(
            self.results[task_id]['result'].get('generation_time', 0)
            for task_id in task_ids
        )
        
        # Store aggregated result
        self.group_results[group_id] = {
            'type': 'token_generation',
            'output_ids': all_tokens,
            'total_generation_time': total_time,
            'aggregation_method': 'sequence_concatenation'
        }
    
    def _aggregate_embedding(self, group_id, task_ids):
        """
        Aggregate embedding results.
        
        Args:
            group_id: ID of the task group
            task_ids: List of task IDs in the group
        """
        # For embeddings, we typically average or concatenate
        aggregation_method = self.results[next(iter(task_ids))]['parameters'].get('aggregation_method', 'average')
        
        all_embeddings = [
            self.results[task_id]['result']['embeddings']
            for task_id in task_ids
        ]
        
        if aggregation_method == 'average':
            # Convert to numpy arrays and average
            try:
                np_embeddings = [np.array(emb) for emb in all_embeddings]
                aggregated_embedding = np.mean(np_embeddings, axis=0).tolist()
            except:
                # Fallback if numpy conversion fails
                aggregated_embedding = all_embeddings[0]
                logger.warning(f"Failed to average embeddings for group {group_id}")
        
        elif aggregation_method == 'concatenate':
            # Flatten and concatenate all embeddings
            aggregated_embedding = []
            for emb in all_embeddings:
                if isinstance(emb, list):
                    aggregated_embedding.extend(emb)
                else:
                    aggregated_embedding.append(emb)
        
        else:
            # Default to returning all embeddings
            aggregated_embedding = all_embeddings
        
        # Store aggregated result
        self.group_results[group_id] = {
            'type': 'embedding',
            'embeddings': aggregated_embedding,
            'aggregation_method': aggregation_method
        }
    
    def clean_old_results(self, max_age_seconds=3600):
        """
        Clean up old results to prevent memory leaks.
        
        Args:
            max_age_seconds: Maximum age of results to keep
        """
        with self.lock:
            current_time = time.time()
            
            # Clean individual task results
            task_ids_to_remove = []
            for task_id, result in self.results.items():
                if current_time - result.get('completed_at', 0) > max_age_seconds:
                    task_ids_to_remove.append(task_id)
            
            for task_id in task_ids_to_remove:
                del self.results[task_id]
            
            # Clean group results and status
            group_ids_to_remove = []
            for group_id in self.group_results:
                if not any(group_id in task_group for task_group in self.task_groups.values()):
                    group_ids_to_remove.append(group_id)
            
            for group_id in group_ids_to_remove:
                if group_id in self.group_results:
                    del self.group_results[group_id]
                if group_id in self.group_status:
                    del self.group_status[group_id]
            
            # Clean empty task groups
            group_ids_to_remove = []
            for group_id, task_ids in self.task_groups.items():
                if not task_ids:
                    group_ids_to_remove.append(group_id)
            
            for group_id in group_ids_to_remove:
                del self.task_groups[group_id]
            
            logger.info(f"Cleaned {len(task_ids_to_remove)} old results")