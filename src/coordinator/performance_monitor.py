"""
Performance Monitor for the DistributedLLM system.

Tracks worker performance, resource utilization, and system health metrics
to enable intelligent workload balancing and optimization.
"""

import logging
import time
import threading
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors performance metrics across workers in the distributed system.
    Used for making workload balancing decisions and identifying bottlenecks.
    """
    
    def __init__(self, metrics_history_size: int = 100):
        """
        Initialize the performance monitor.
        
        Args:
            metrics_history_size: Number of metrics points to keep in history
        """
        self.metrics_history_size = metrics_history_size
        self.worker_metrics = defaultdict(lambda: defaultdict(deque))
        self.task_metrics = defaultdict(lambda: defaultdict(deque))
        self.system_metrics = defaultdict(deque)
        self.lock = threading.RLock()
        
        # Track task completions by worker
        self.task_completions = defaultdict(list)
        
        # Performance profiles for different task types
        self.task_type_profiles = defaultdict(dict)
        
        # Record when monitoring started
        self.start_time = time.time()
    
    def update_worker_metrics(self, worker_id: str, metrics: Dict[str, Any]):
        """
        Update metrics for a specific worker.
        
        Args:
            worker_id: ID of the worker
            metrics: Dictionary of metrics
        """
        with self.lock:
            timestamp = time.time()
            
            for metric_name, metric_value in metrics.items():
                # Only store numeric metrics
                if isinstance(metric_value, (int, float)):
                    self.worker_metrics[worker_id][metric_name].append((timestamp, metric_value))
                    
                    # Trim history if needed
                    if len(self.worker_metrics[worker_id][metric_name]) > self.metrics_history_size:
                        self.worker_metrics[worker_id][metric_name].popleft()
    
    def record_task_completion(self, worker_id: str, task_id: str, metrics: Dict[str, Any]):
        """
        Record metrics for a completed task.
        
        Args:
            worker_id: ID of the worker that completed the task
            task_id: ID of the completed task
            metrics: Dictionary of metrics for the task
        """
        with self.lock:
            timestamp = time.time()
            task_type = metrics.get('task_type', 'unknown')
            
            # Store task completion record
            self.task_completions[worker_id].append({
                'task_id': task_id,
                'task_type': task_type,
                'timestamp': timestamp,
                'metrics': metrics
            })
            
            # Trim history if needed
            if len(self.task_completions[worker_id]) > self.metrics_history_size:
                self.task_completions[worker_id].pop(0)
            
            # Update task type metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.task_metrics[task_type][metric_name].append((timestamp, metric_value))
                    
                    # Trim history if needed
                    if len(self.task_metrics[task_type][metric_name]) > self.metrics_history_size:
                        self.task_metrics[task_type][metric_name].popleft()
            
            # Update task type performance profile
            self._update_task_profile(task_type, worker_id, metrics)
    
    def _update_task_profile(self, task_type: str, worker_id: str, metrics: Dict[str, Any]):
        """
        Update performance profile for a specific task type.
        
        Args:
            task_type: Type of the task
            worker_id: ID of the worker that completed the task
            metrics: Metrics from the task completion
        """
        if 'elapsed_time' not in metrics and 'execution_time' not in metrics:
            return
        
        execution_time = metrics.get('execution_time', metrics.get('elapsed_time'))
        
        # Initialize profile for this task type and worker if needed
        if worker_id not in self.task_type_profiles[task_type]:
            self.task_type_profiles[task_type][worker_id] = {
                'count': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'recent_times': deque(maxlen=10)  # Keep last 10 times
            }
        
        profile = self.task_type_profiles[task_type][worker_id]
        
        # Update profile
        profile['count'] += 1
        profile['total_time'] += execution_time
        profile['min_time'] = min(profile['min_time'], execution_time)
        profile['max_time'] = max(profile['max_time'], execution_time)
        profile['recent_times'].append(execution_time)
        
        # Calculate rolling average
        profile['avg_time'] = profile['total_time'] / profile['count']
        profile['recent_avg_time'] = sum(profile['recent_times']) / len(profile['recent_times'])
    
    def get_worker_performance(self, worker_id: str, task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a specific worker.
        
        Args:
            worker_id: ID of the worker
            task_type: Optional task type to filter by
        
        Returns:
            Dictionary of performance metrics
        """
        with self.lock:
            performance = {
                'task_count': 0,
                'avg_execution_time': 0,
                'recent_execution_time': 0,
                'task_types': {}
            }
            
            # Get task completions for this worker
            completions = self.task_completions.get(worker_id, [])
            
            if not completions:
                return performance
            
            # Filter by task type if specified
            if task_type:
                completions = [c for c in completions if c['task_type'] == task_type]
            
            if not completions:
                return performance
            
            # Calculate overall metrics
            performance['task_count'] = len(completions)
            
            # Calculate execution times
            execution_times = []
            for completion in completions:
                metrics = completion['metrics']
                if 'execution_time' in metrics:
                    execution_times.append(metrics['execution_time'])
                elif 'elapsed_time' in metrics:
                    execution_times.append(metrics['elapsed_time'])
            
            if execution_times:
                performance['avg_execution_time'] = sum(execution_times) / len(execution_times)
                performance['recent_execution_time'] = sum(execution_times[-5:]) / min(len(execution_times), 5)
            
            # Calculate task type breakdown
            task_type_counts = defaultdict(int)
            for completion in completions:
                task_type_counts[completion['task_type']] += 1
            
            for t_type, count in task_type_counts.items():
                performance['task_types'][t_type] = {
                    'count': count,
                    'percentage': (count / performance['task_count']) * 100
                }
                
                # Add detailed profile if available
                if t_type in self.task_type_profiles and worker_id in self.task_type_profiles[t_type]:
                    performance['task_types'][t_type].update(self.task_type_profiles[t_type][worker_id])
            
            return performance
    
    def get_task_type_performance(self, task_type: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific task type across all workers.
        
        Args:
            task_type: Type of task to get metrics for
        
        Returns:
            Dictionary of performance metrics for the task type
        """
        with self.lock:
            performance = {
                'worker_count': 0,
                'global_avg_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'workers': {}
            }
            
            if task_type not in self.task_type_profiles:
                return performance
            
            profiles = self.task_type_profiles[task_type]
            performance['worker_count'] = len(profiles)
            
            # Calculate global stats
            total_count = sum(profile['count'] for profile in profiles.values())
            total_time = sum(profile['total_time'] for profile in profiles.values())
            
            if total_count > 0:
                performance['global_avg_time'] = total_time / total_count
            
            # Find min and max times
            min_times = [profile['min_time'] for profile in profiles.values() if profile['min_time'] < float('inf')]
            max_times = [profile['max_time'] for profile in profiles.values()]
            
            if min_times:
                performance['min_time'] = min(min_times)
            if max_times:
                performance['max_time'] = max(max_times)
            
            # Add per-worker performance
            for worker_id, profile in profiles.items():
                performance['workers'][worker_id] = {
                    'count': profile['count'],
                    'avg_time': profile['avg_time'],
                    'recent_avg_time': profile.get('recent_avg_time', profile['avg_time']),
                    'min_time': profile['min_time'],
                    'max_time': profile['max_time'],
                    'efficiency': performance['global_avg_time'] / profile['avg_time'] if profile['avg_time'] > 0 else 0
                }
            
            return performance
    
    def get_worker_utilization(self, worker_id: str) -> Dict[str, Any]:
        """
        Get resource utilization metrics for a specific worker.
        
        Args:
            worker_id: ID of the worker
        
        Returns:
            Dictionary of utilization metrics
        """
        with self.lock:
            utilization = {
                'cpu_percent': None,
                'memory_percent': None,
                'gpu_percent': None,
                'idle_percent': None,
                'trends': {}
            }
            
            if worker_id not in self.worker_metrics:
                return utilization
            
            metrics = self.worker_metrics[worker_id]
            
            # Get latest values for key metrics
            for metric_name in ['cpu_percent', 'memory_percent', 'gpu_percent']:
                if metric_name in metrics and metrics[metric_name]:
                    utilization[metric_name] = metrics[metric_name][-1][1]
            
            # Calculate idle time percentage
            busy_periods = self._calculate_busy_periods(worker_id)
            if busy_periods:
                total_time = time.time() - self.start_time
                busy_time = sum(end - start for start, end in busy_periods)
                utilization['idle_percent'] = 100 - (busy_time / total_time) * 100
            
            # Calculate trends for key metrics
            for metric_name in ['cpu_percent', 'memory_percent', 'gpu_percent']:
                if metric_name in metrics and len(metrics[metric_name]) >= 2:
                    history = metrics[metric_name]
                    
                    # Calculate short-term trend (last 10 points)
                    short_term = history[-min(10, len(history)):]
                    if len(short_term) >= 2:
                        values = [v for _, v in short_term]
                        utilization['trends'][f'{metric_name}_short_term'] = (values[-1] - values[0]) / len(values)
                    
                    # Calculate long-term trend (all points)
                    if len(history) >= 5:
                        values = [v for _, v in history]
                        utilization['trends'][f'{metric_name}_long_term'] = (values[-1] - values[0]) / len(values)
            
            return utilization
    
    def _calculate_busy_periods(self, worker_id: str) -> List[Tuple[float, float]]:
        """
        Calculate periods when a worker was busy.
        
        Args:
            worker_id: ID of the worker
        
        Returns:
            List of (start_time, end_time) tuples representing busy periods
        """
        completions = self.task_completions.get(worker_id, [])
        if not completions:
            return []
        
        # Sort completions by timestamp
        sorted_completions = sorted(completions, key=lambda c: c['timestamp'])
        
        # Extract start and end times from task metrics
        periods = []
        for completion in sorted_completions:
            metrics = completion['metrics']
            timestamp = completion['timestamp']
            
            if 'execution_time' in metrics:
                start_time = timestamp - metrics['execution_time']
                end_time = timestamp
                periods.append((start_time, end_time))
        
        # Merge overlapping periods
        if not periods:
            return []
        
        periods.sort()
        merged = [periods[0]]
        
        for current_start, current_end in periods[1:]:
            prev_start, prev_end = merged[-1]
            
            if current_start <= prev_end:
                # Periods overlap, merge them
                merged[-1] = (prev_start, max(prev_end, current_end))
            else:
                # No overlap, add as new period
                merged.append((current_start, current_end))
        
        return merged
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics for the entire system.
        
        Returns:
            Dictionary of system-wide metrics
        """
        with self.lock:
            metrics = {
                'worker_metrics': {},
                'task_type_metrics': {},
                'system_metrics': {
                    'total_tasks_completed': 0,
                    'tasks_per_second': 0,
                    'worker_count': len(self.worker_metrics)
                }
            }
            
            # Collect worker metrics
            for worker_id in self.worker_metrics:
                metrics['worker_metrics'][worker_id] = {
                    'utilization': self.get_worker_utilization(worker_id),
                    'performance': self.get_worker_performance(worker_id)
                }
            
            # Collect task type metrics
            for task_type in self.task_type_profiles:
                metrics['task_type_metrics'][task_type] = self.get_task_type_performance(task_type)
            
            # Calculate system-wide metrics
            total_tasks = sum(len(completions) for completions in self.task_completions.values())
            metrics['system_metrics']['total_tasks_completed'] = total_tasks
            
            # Calculate tasks per second
            elapsed_time = max(time.time() - self.start_time, 1)
            metrics['system_metrics']['tasks_per_second'] = total_tasks / elapsed_time
            
            # Calculate load balance metrics
            if self.worker_metrics:
                cpu_utilizations = []
                for worker_id in self.worker_metrics:
                    util = self.get_worker_utilization(worker_id)
                    if util['cpu_percent'] is not None:
                        cpu_utilizations.append(util['cpu_percent'])
                
                if cpu_utilizations:
                    metrics['system_metrics']['avg_cpu_utilization'] = sum(cpu_utilizations) / len(cpu_utilizations)
                    metrics['system_metrics']['cpu_utilization_stddev'] = float(np.std(cpu_utilizations)) if len(cpu_utilizations) > 1 else 0
                    metrics['system_metrics']['load_balance_index'] = metrics['system_metrics']['cpu_utilization_stddev'] / max(metrics['system_metrics']['avg_cpu_utilization'], 1) if metrics['system_metrics']['avg_cpu_utilization'] > 0 else 0
            
            return metrics
    
    def should_rebalance(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if workload rebalancing is needed.
        
        Returns:
            Tuple of (should_rebalance, reason_dict)
        """
        metrics = self.get_current_metrics()
        
        # Default to not rebalancing
        should_rebalance = False
        reason = {'reason': 'no_rebalance_needed', 'details': {}}
        
        # Check CPU utilization imbalance
        if 'avg_cpu_utilization' in metrics['system_metrics'] and 'load_balance_index' in metrics['system_metrics']:
            lbi = metrics['system_metrics']['load_balance_index']
            avg_util = metrics['system_metrics']['avg_cpu_utilization']
            
            # If LBI is high (> 0.25) and average utilization is significant (> 30%)
            if lbi > 0.25 and avg_util > 30:
                should_rebalance = True
                reason = {
                    'reason': 'cpu_utilization_imbalance',
                    'details': {
                        'load_balance_index': lbi,
                        'avg_cpu_utilization': avg_util
                    }
                }
        
        # Check task execution time imbalance
        if not should_rebalance and len(self.task_type_profiles) > 0:
            for task_type, profiles in self.task_type_profiles.items():
                if len(profiles) < 2:
                    continue
                
                perf = self.get_task_type_performance(task_type)
                
                # Check for significant performance difference between workers
                worker_times = [w['avg_time'] for w in perf['workers'].values() if w['count'] > 2]
                if len(worker_times) < 2:
                    continue
                
                max_time = max(worker_times)
                min_time = min(worker_times)
                
                # If fastest worker is at least 1.5x faster than slowest
                if min_time > 0 and max_time / min_time > 1.5:
                    should_rebalance = True
                    reason = {
                        'reason': 'task_execution_time_imbalance',
                        'details': {
                            'task_type': task_type,
                            'max_time': max_time,
                            'min_time': min_time,
                            'ratio': max_time / min_time
                        }
                    }
                    break
        
        return should_rebalance, reason
    
    def suggest_workload_distribution(self) -> Dict[str, float]:
        """
        Suggest workload distribution among workers based on performance.
        
        Returns:
            Dictionary mapping worker IDs to workload fractions (0-1)
        """
        with self.lock:
            distribution = {}
            
            # If we have performance profiles, use them
            if self.task_type_profiles:
                # Average across all task types, weighted by count
                worker_speeds = defaultdict(float)
                total_tasks = 0
                
                for task_type, profiles in self.task_type_profiles.items():
                    type_total_tasks = sum(profile['count'] for profile in profiles.values())
                    total_tasks += type_total_tasks
                    
                    for worker_id, profile in profiles.items():
                        if profile['avg_time'] > 0:
                            # Speed is inverse of time (faster workers have higher speed)
                            speed = 1.0 / profile['avg_time']
                            worker_speeds[worker_id] += speed * profile['count']
                
                # Normalize to fractions
                total_speed = sum(worker_speeds.values())
                if total_speed > 0:
                    for worker_id, speed in worker_speeds.items():
                        distribution[worker_id] = speed / total_speed
                
                # If we have workers with no tasks yet, give them a small share
                for worker_id in self.worker_metrics:
                    if worker_id not in distribution:
                        # Give them a small fraction (10% of an equal share)
                        equal_share = 1.0 / len(self.worker_metrics)
                        distribution[worker_id] = equal_share * 0.1
                
                # Renormalize
                total = sum(distribution.values())
                if total > 0:
                    for worker_id in distribution:
                        distribution[worker_id] /= total
            
            # If no performance data, distribute equally
            if not distribution:
                worker_count = len(self.worker_metrics)
                if worker_count > 0:
                    equal_share = 1.0 / worker_count
                    for worker_id in self.worker_metrics:
                        distribution[worker_id] = equal_share
            
            return distribution