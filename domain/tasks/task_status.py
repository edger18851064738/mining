"""
Task status definitions for the mining dispatch system.

Defines the possible statuses and transitions for tasks in the system.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from enum import Enum, auto
from typing import Dict, List, Set, Optional


class TaskStatus(Enum):
    """
    Enumeration of possible task statuses.
    
    Each status represents a discrete state in the task lifecycle.
    """
    
    PENDING = auto()      # Task is waiting to be assigned
    ASSIGNED = auto()     # Task is assigned but not started
    IN_PROGRESS = auto()  # Task is in progress
    COMPLETED = auto()    # Task is completed successfully
    FAILED = auto()       # Task failed
    CANCELED = auto()     # Task was canceled
    BLOCKED = auto()      # Task is blocked by a dependency or constraint
    PAUSED = auto()       # Task is paused

    @classmethod
    def get_valid_transitions(cls, current_status: 'TaskStatus') -> Set['TaskStatus']:
        """
        Get the set of valid status transitions from a given status.
        
        Args:
            current_status: The current task status
            
        Returns:
            Set[TaskStatus]: Set of valid next statuses
        """
        # Define valid status transitions
        transitions = {
            cls.PENDING: {cls.ASSIGNED, cls.CANCELED, cls.BLOCKED},
            cls.ASSIGNED: {cls.IN_PROGRESS, cls.PENDING, cls.CANCELED, cls.BLOCKED},
            cls.IN_PROGRESS: {cls.COMPLETED, cls.FAILED, cls.PAUSED, cls.CANCELED, cls.BLOCKED},
            cls.COMPLETED: set(),  # Terminal state
            cls.FAILED: {cls.PENDING},  # Can retry
            cls.CANCELED: {cls.PENDING},  # Can requeue
            cls.BLOCKED: {cls.PENDING, cls.ASSIGNED, cls.CANCELED},
            cls.PAUSED: {cls.IN_PROGRESS, cls.CANCELED, cls.BLOCKED}
        }
        
        return transitions.get(current_status, set())
    
    @classmethod
    def can_transition(cls, current_status: 'TaskStatus', 
                      next_status: 'TaskStatus') -> bool:
        """
        Check if a transition from current_status to next_status is valid.
        
        Args:
            current_status: The current task status
            next_status: The target task status
            
        Returns:
            bool: True if transition is valid, False otherwise
        """
        valid_transitions = cls.get_valid_transitions(current_status)
        return next_status in valid_transitions

    @classmethod
    def is_terminal(cls, status: 'TaskStatus') -> bool:
        """
        Check if a status is terminal (no outgoing transitions).
        
        Args:
            status: The task status to check
            
        Returns:
            bool: True if status is terminal, False otherwise
        """
        return not bool(cls.get_valid_transitions(status))
    
    @classmethod
    def is_active(cls, status: 'TaskStatus') -> bool:
        """
        Check if a status represents an active task.
        
        Args:
            status: The task status to check
            
        Returns:
            bool: True if status is active, False otherwise
        """
        return status in {cls.ASSIGNED, cls.IN_PROGRESS, cls.PAUSED}


class TaskPriority(Enum):
    """
    Enumeration of task priority levels.
    
    Higher values indicate higher priority.
    """
    
    LOW = 1
    NORMAL = 3
    HIGH = 5
    URGENT = 8
    CRITICAL = 10


class TaskType(Enum):
    """
    Enumeration of task types in a mining system.
    
    Each type represents a different kind of mining operation.
    """
    
    TRANSPORT = auto()    # Moving material from one location to another
    LOADING = auto()      # Loading material onto vehicles
    UNLOADING = auto()    # Unloading material from vehicles
    EXCAVATION = auto()   # Excavating material from the ground
    DRILLING = auto()     # Drilling for blasting or sampling
    MAINTENANCE = auto()  # Vehicle or equipment maintenance
    REFUELING = auto()    # Refueling vehicles
    INSPECTION = auto()   # Inspecting equipment or areas
    SURVEYING = auto()    # Surveying an area
    CLEANUP = auto()      # Cleaning up an area


class TaskStatusTransitionError(Exception):
    """Exception raised for invalid task status transitions."""
    
    def __init__(self, current_status: TaskStatus, target_status: TaskStatus):
        """
        Initialize a status transition error.
        
        Args:
            current_status: Current task status
            target_status: Target task status
        """
        self.current_status = current_status
        self.target_status = target_status
        super().__init__(
            f"Invalid task status transition: {current_status.name} -> {target_status.name}"
        )


class TaskStatusManager:
    """
    Manages the status and status transitions of a task.
    
    Ensures that status transitions are valid and maintains history of statuses.
    """
    
    def __init__(self, initial_status: TaskStatus = TaskStatus.PENDING):
        """
        Initialize the status manager.
        
        Args:
            initial_status: Initial task status
        """
        self._current_status = initial_status
        self._status_history = [(initial_status, None)]  # (status, timestamp)
    
    @property
    def current_status(self) -> TaskStatus:
        """Get the current task status."""
        return self._current_status
    
    def transition_to(self, target_status: TaskStatus, 
                     force: bool = False) -> bool:
        """
        Transition to a new status.
        
        Args:
            target_status: Target status
            force: If True, allow invalid transitions
            
        Returns:
            bool: True if transition was successful
            
        Raises:
            TaskStatusTransitionError: If transition is invalid and force=False
        """
        # Check if transition is valid
        if not force and not TaskStatus.can_transition(self._current_status, target_status):
            raise TaskStatusTransitionError(self._current_status, target_status)
        
        # Update current status
        self._current_status = target_status
        
        # Update history with timestamp
        import time
        self._status_history.append((target_status, time.time()))
        
        return True
    
    def get_status_history(self) -> List[tuple]:
        """
        Get the history of status transitions.
        
        Returns:
            List[tuple]: List of (status, timestamp) tuples
        """
        return self._status_history.copy()
    
    def get_time_in_current_status(self) -> float:
        """
        Get the time spent in the current status (seconds).
        
        Returns:
            float: Time in seconds
        """
        if len(self._status_history) < 1:
            return 0.0
            
        import time
        last_transition_time = self._status_history[-1][1]
        
        if last_transition_time is None:
            return 0.0
            
        return time.time() - last_transition_time
    
    def reset(self, status: TaskStatus = TaskStatus.PENDING) -> None:
        """
        Reset the status manager to a specified status.
        
        Args:
            status: Status to reset to
        """
        import time
        self._current_status = status
        self._status_history = [(status, time.time())]
    
    def __str__(self) -> str:
        """String representation of the current status."""
        return f"TaskStatus: {self._current_status.name}"