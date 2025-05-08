"""
Base task definitions for the mining dispatch system.

Defines the abstract task interface and common task behavior.
All specific task types should inherit from these base classes.
"""

from abc import ABC, abstractmethod
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from datetime import datetime
import math

from utils.geo.coordinates import Point2D
from utils.io.serialization import Serializable
from utils.logger import get_logger

# Initialize logger
logger = get_logger("task")


class TaskStatus:
    """Status constants for tasks."""
    PENDING = "pending"       # Task is waiting to be assigned
    ASSIGNED = "assigned"     # Task is assigned but not started
    IN_PROGRESS = "in_progress"  # Task is in progress
    COMPLETED = "completed"   # Task is completed successfully
    FAILED = "failed"         # Task failed
    CANCELED = "canceled"     # Task was canceled


class TaskError(Exception):
    """Base exception for task-related errors."""
    pass


class TaskStateError(TaskError):
    """Exception raised for invalid task state transitions."""
    pass


class TaskAssignmentError(TaskError):
    """Exception raised for errors during task assignment."""
    pass


class Task(ABC, Serializable):
    """
    Abstract base class for all task types.
    
    Defines the common interface and behavior that all tasks must implement.
    """
    
    # Valid state transitions
    STATE_TRANSITIONS = {
        TaskStatus.PENDING: {TaskStatus.ASSIGNED, TaskStatus.CANCELED},
        TaskStatus.ASSIGNED: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELED},
        TaskStatus.IN_PROGRESS: {TaskStatus.COMPLETED, TaskStatus.FAILED},
        TaskStatus.COMPLETED: set(),  # Terminal state
        TaskStatus.FAILED: {TaskStatus.PENDING},  # Can retry
        TaskStatus.CANCELED: {TaskStatus.PENDING}  # Can requeue
    }
    
    def __init__(self, task_id: Optional[str] = None, 
                priority: int = 1,
                deadline: Optional[datetime] = None):
        """
        Initialize a task.
        
        Args:
            task_id: Unique identifier for the task (generated if None)
            priority: Task priority (higher numbers = higher priority)
            deadline: Task deadline
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.priority = priority
        self.creation_time = datetime.now()
        self.deadline = deadline
        
        # State tracking
        self._status = TaskStatus.PENDING
        self._status_history = [(TaskStatus.PENDING, self.creation_time)]
        
        # Assignment tracking
        self.assigned_to = None
        self.assigned_time = None
        
        # Execution tracking
        self.start_time = None
        self.end_time = None
        self.progress = 0.0  # 0-1 progress indicator
        self.retry_count = 0
    
    @property
    def status(self) -> str:
        """Get the current status of the task."""
        return self._status
    
    @status.setter
    def status(self, new_status: str) -> None:
        """
        Set the task status.
        
        Args:
            new_status: New status
            
        Raises:
            TaskStateError: If transition is invalid
        """
        self._set_status(new_status)
    
    def _set_status(self, new_status: str, force: bool = False) -> None:
        """
        Internal method to set status with transition validation.
        
        Args:
            new_status: New status
            force: If True, skip transition validation
            
        Raises:
            TaskStateError: If transition is invalid and force=False
        """
        # Check if transition is valid
        if (not force and new_status != self._status and 
            new_status not in self.STATE_TRANSITIONS.get(self._status, set())):
            raise TaskStateError(
                f"Invalid task state transition: {self._status} -> {new_status}"
            )
        
        # Update status
        old_status = self._status
        self._status = new_status
        
        # Record status change
        self._status_history.append((new_status, datetime.now()))
        
        # Additional processing for specific transitions
        if new_status == TaskStatus.IN_PROGRESS and self.start_time is None:
            self.start_time = datetime.now()
        elif new_status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED):
            self.end_time = datetime.now()
            
        logger.debug(f"Task {self.task_id} status changed: {old_status} -> {new_status}")
    
    @property
    def is_active(self) -> bool:
        """Check if the task is active (assigned or in progress)."""
        return self.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
    
    @property
    def is_completed(self) -> bool:
        """Check if the task is completed."""
        return self.status == TaskStatus.COMPLETED
    
    @property
    def is_pending(self) -> bool:
        """Check if the task is pending assignment."""
        return self.status == TaskStatus.PENDING
    
    @property
    def execution_time(self) -> Optional[float]:
        """
        Get the task execution time in seconds.
        
        Returns:
            Optional[float]: Execution time or None if not started
        """
        if self.start_time is None:
            return None
            
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def wait_time(self) -> float:
        """
        Get the task wait time in seconds.
        
        Returns:
            float: Wait time
        """
        start = self.start_time or datetime.now()
        return (start - self.creation_time).total_seconds()
    
    @property
    def is_overdue(self) -> bool:
        """Check if the task is past its deadline."""
        if self.deadline is None:
            return False
            
        return datetime.now() > self.deadline
    
    @property
    def time_to_deadline(self) -> Optional[float]:
        """
        Get time remaining until deadline in seconds.
        
        Returns:
            Optional[float]: Time to deadline or None if no deadline
        """
        if self.deadline is None:
            return None
            
        return (self.deadline - datetime.now()).total_seconds()
    
    @property
    def urgency(self) -> float:
        """
        Calculate task urgency (0-1).
        
        Considers priority and deadline.
        
        Returns:
            float: Urgency score
        """
        # Base urgency from priority (normalize to 0-0.5)
        # Assuming priorities typically range from 1-10
        priority_urgency = min(0.5, self.priority / 20.0)
        
        # Deadline urgency (0-0.5)
        deadline_urgency = 0.0
        if self.deadline is not None:
            time_left = self.time_to_deadline
            if time_left is not None:
                if time_left <= 0:
                    # Past deadline
                    deadline_urgency = 0.5
                else:
                    # Approaching deadline (assumes 1 hour = urgent)
                    hours_left = time_left / 3600.0
                    deadline_urgency = 0.5 * max(0.0, 1.0 - min(1.0, hours_left / 1.0))
        
        return priority_urgency + deadline_urgency
    
    def assign(self, assignee_id: str) -> None:
        """
        Assign the task to an entity.
        
        Args:
            assignee_id: ID of the entity to assign the task to
            
        Raises:
            TaskStateError: If task is not in a state that can be assigned
        """
        if not self.is_pending:
            raise TaskStateError(f"Cannot assign task in state: {self.status}")
            
        self.assigned_to = assignee_id
        self.assigned_time = datetime.now()
        self.status = TaskStatus.ASSIGNED
        
        logger.info(f"Task {self.task_id} assigned to {assignee_id}")
    
    def start(self) -> None:
        """
        Start the task execution.
        
        Raises:
            TaskStateError: If task is not in a state that can be started
        """
        if self.status != TaskStatus.ASSIGNED:
            raise TaskStateError(f"Cannot start task in state: {self.status}")
            
        self.status = TaskStatus.IN_PROGRESS
        self.start_time = datetime.now()
        
        logger.info(f"Task {self.task_id} started")
    
    def complete(self) -> None:
        """
        Mark the task as completed.
        
        Raises:
            TaskStateError: If task is not in a state that can be completed
        """
        if self.status != TaskStatus.IN_PROGRESS:
            raise TaskStateError(f"Cannot complete task in state: {self.status}")
            
        self.status = TaskStatus.COMPLETED
        self.end_time = datetime.now()
        self.progress = 1.0
        
        logger.info(f"Task {self.task_id} completed")
    
    def fail(self, reason: str = "") -> None:
        """
        Mark the task as failed.
        
        Args:
            reason: Reason for failure
            
        Raises:
            TaskStateError: If task is not in a state that can be failed
        """
        if self.status != TaskStatus.IN_PROGRESS:
            raise TaskStateError(f"Cannot fail task in state: {self.status}")
            
        self.status = TaskStatus.FAILED
        self.end_time = datetime.now()
        
        logger.warning(f"Task {self.task_id} failed: {reason}")
    
    def cancel(self) -> None:
        """
        Cancel the task.
        
        Raises:
            TaskStateError: If task is in a terminal state
        """
        if self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            raise TaskStateError(f"Cannot cancel task in terminal state: {self.status}")
            
        self.status = TaskStatus.CANCELED
        self.end_time = datetime.now()
        
        logger.info(f"Task {self.task_id} canceled")
    
    def reset(self) -> None:
        """
        Reset the task to pending state.
        
        Useful for retrying failed tasks.
        """
        if self.status not in (TaskStatus.FAILED, TaskStatus.CANCELED):
            raise TaskStateError(f"Cannot reset task in state: {self.status}")
            
        self.status = TaskStatus.PENDING
        self.assigned_to = None
        self.assigned_time = None
        self.start_time = None
        self.end_time = None
        self.progress = 0.0
        
        if self.status == TaskStatus.FAILED:
            self.retry_count += 1
            
        logger.info(f"Task {self.task_id} reset to pending state")
    
    def update_progress(self, progress: float) -> None:
        """
        Update the task progress.
        
        Args:
            progress: Progress value (0-1)
        """
        if not self.is_active:
            logger.warning(f"Updating progress for inactive task {self.task_id}")
            
        self.progress = max(0.0, min(1.0, progress))
        
        # Auto-complete when progress reaches 1.0
        if math.isclose(self.progress, 1.0) and self.status == TaskStatus.IN_PROGRESS:
            self.complete()
    
    def get_duration_estimate(self) -> float:
        """
        Get an estimate of the task duration in seconds.
        
        Returns:
            float: Estimated duration
        """
        # Default implementation - subclasses should override
        if self.execution_time is not None:
            return self.execution_time
            
        # Return a default estimate based on priority
        # Higher priority typically means more urgent/shorter
        return 600.0 / max(1, self.priority)  # 10 minutes for priority 1
    
    def get_status_history(self) -> List[Tuple[str, datetime]]:
        """
        Get the history of status changes.
        
        Returns:
            List[Tuple[str, datetime]]: List of (status, timestamp) tuples
        """
        return self._status_history.copy()
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'task_id': self.task_id,
            'priority': self.priority,
            'status': self.status,
            'creation_time': self.creation_time.isoformat(),
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'assigned_to': self.assigned_to,
            'assigned_time': self.assigned_time.isoformat() if self.assigned_time else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'progress': self.progress,
            'retry_count': self.retry_count
        }
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """
        Create a task from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Task: New task instance
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the task."""
        return f"{self.__class__.__name__}(id={self.task_id}, status={self.status}, priority={self.priority})"


class TaskAssignment:
    """
    Represents the assignment of a task to an entity.
    
    Used to track task assignments and their performance metrics.
    """
    
    def __init__(self, task: Task, assignee_id: str, 
                estimated_duration: Optional[float] = None):
        """
        Initialize a task assignment.
        
        Args:
            task: The task being assigned
            assignee_id: ID of the entity the task is assigned to
            estimated_duration: Estimated duration in seconds
        """
        self.task = task
        self.assignee_id = assignee_id
        self.assignment_time = datetime.now()
        self.estimated_duration = estimated_duration or task.get_duration_estimate()
        self.actual_duration = None
        self.completion_status = None
        
        # Assign the task
        self.task.assign(assignee_id)
    
    def start(self) -> None:
        """Start the assigned task."""
        self.task.start()
    
    def complete(self) -> None:
        """Mark the assigned task as completed."""
        self.task.complete()
        self.actual_duration = self.task.execution_time
        self.completion_status = TaskStatus.COMPLETED
    
    def fail(self, reason: str = "") -> None:
        """Mark the assigned task as failed."""
        self.task.fail(reason)
        self.actual_duration = self.task.execution_time
        self.completion_status = TaskStatus.FAILED
    
    def cancel(self) -> None:
        """Cancel the assigned task."""
        self.task.cancel()
        self.actual_duration = self.task.execution_time
        self.completion_status = TaskStatus.CANCELED
    
    @property
    def is_active(self) -> bool:
        """Check if the assignment is active."""
        return self.task.is_active
    
    @property
    def is_completed(self) -> bool:
        """Check if the assignment is completed."""
        return self.task.is_completed
    
    @property
    def duration_performance(self) -> Optional[float]:
        """
        Calculate the duration performance ratio.
        
        Returns ratio of actual to estimated duration (< 1 means faster than estimated).
        
        Returns:
            Optional[float]: Performance ratio or None if not completed
        """
        if self.actual_duration is None or self.estimated_duration is None:
            return None
            
        if self.estimated_duration == 0:
            return float('inf')
            
        return self.actual_duration / self.estimated_duration
    
    def update_progress(self, progress: float) -> None:
        """Update the progress of the assigned task."""
        self.task.update_progress(progress)
        
        # If task completes, update completion metrics
        if self.task.is_completed and self.completion_status is None:
            self.actual_duration = self.task.execution_time
            self.completion_status = TaskStatus.COMPLETED
    
    def __repr__(self) -> str:
        """String representation of the assignment."""
        status = self.task.status
        completion = f", performance={self.duration_performance:.2f}" if self.duration_performance else ""
        return f"TaskAssignment(task={self.task.task_id}, assignee={self.assignee_id}, status={status}{completion})"