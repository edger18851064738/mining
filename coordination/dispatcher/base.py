"""
Base dispatcher interfaces and abstract classes.

Defines the core interfaces and abstract classes for the dispatch system.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import abc
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable
from datetime import datetime

from utils.logger import get_logger
from utils.io.serialization import Serializable
from utils.geo.coordinates import Point2D

# Import from dispatch_events
from coordination.dispatcher.dispatch_events import EventType, DispatchEvent, EventListener, EventDispatcher


class DispatchStatus(Enum):
    """Status of a dispatch operation."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()


class DispatchStrategy(Enum):
    """Strategies for dispatching vehicles to tasks."""
    FIFO = auto()              # First In, First Out
    PRIORITY = auto()          # Based on task priority
    NEAREST = auto()           # Nearest vehicle to task
    BALANCED = auto()          # Balance load across vehicles
    OPTIMIZED = auto()         # Globally optimized solution


class DispatchError(Exception):
    """Base class for dispatch system errors."""
    pass


class VehicleNotFoundError(DispatchError):
    """Error raised when a vehicle is not found."""
    pass


class TaskNotFoundError(DispatchError):
    """Error raised when a task is not found."""
    pass


class DispatcherConfig:
    """Configuration for the dispatcher."""
    
    def __init__(self, 
                 dispatch_interval: float = 1.0,
                 dispatch_strategy: DispatchStrategy = DispatchStrategy.OPTIMIZED,
                 max_tasks_per_vehicle: int = 5,
                 replan_on_change: bool = True,
                 conflict_resolution_enabled: bool = True,
                 **kwargs):
        """
        Initialize dispatcher configuration.
        
        Args:
            dispatch_interval: Time between dispatch cycles in seconds
            dispatch_strategy: Strategy to use for dispatching
            max_tasks_per_vehicle: Maximum number of tasks per vehicle
            replan_on_change: Whether to replan when environment changes
            conflict_resolution_enabled: Whether to enable conflict resolution
            **kwargs: Additional configuration parameters
        """
        self.dispatch_interval = dispatch_interval
        self.dispatch_strategy = dispatch_strategy
        self.max_tasks_per_vehicle = max_tasks_per_vehicle
        self.replan_on_change = replan_on_change
        self.conflict_resolution_enabled = conflict_resolution_enabled
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)


class Dispatcher(abc.ABC, Serializable):
    """
    Abstract base class for dispatchers.
    
    A dispatcher manages the assignment of tasks to vehicles
    and coordinates their movement and actions.
    """
    
    def __init__(self, config: Optional[DispatcherConfig] = None):
        """
        Initialize the dispatcher.
        
        Args:
            config: Dispatcher configuration
        """
        self.config = config or DispatcherConfig()
        self.logger = get_logger(self.__class__.__name__)
        self.event_dispatcher = EventDispatcher()
        
        # State
        self._status = DispatchStatus.IDLE
        self._last_dispatch_time = None
    
    @abc.abstractmethod
    def add_vehicle(self, vehicle) -> None:
        """
        Add a vehicle to the dispatch system.
        
        Args:
            vehicle: Vehicle to add
        """
        pass
    
    @abc.abstractmethod
    def remove_vehicle(self, vehicle_id: str) -> bool:
        """
        Remove a vehicle from the dispatch system.
        
        Args:
            vehicle_id: ID of the vehicle to remove
            
        Returns:
            bool: True if vehicle was removed, False if not found
        """
        pass
    
    @abc.abstractmethod
    def get_vehicle(self, vehicle_id: str):
        """
        Get a vehicle by ID.
        
        Args:
            vehicle_id: ID of the vehicle
            
        Returns:
            Vehicle: The vehicle
            
        Raises:
            VehicleNotFoundError: If vehicle not found
        """
        pass
    
    @abc.abstractmethod
    def get_all_vehicles(self) -> dict:
        """
        Get all vehicles.
        
        Returns:
            Dict[str, Vehicle]: Dictionary of vehicle_id to vehicle
        """
        pass
    
    @abc.abstractmethod
    def add_task(self, task) -> None:
        """
        Add a task to the dispatch system.
        
        Args:
            task: Task to add
        """
        pass
    
    @abc.abstractmethod
    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the dispatch system.
        
        Args:
            task_id: ID of the task to remove
            
        Returns:
            bool: True if task was removed, False if not found
        """
        pass
    
    @abc.abstractmethod
    def get_task(self, task_id: str):
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task: The task
            
        Raises:
            TaskNotFoundError: If task not found
        """
        pass
    
    @abc.abstractmethod
    def get_all_tasks(self) -> dict:
        """
        Get all tasks.
        
        Returns:
            Dict[str, Task]: Dictionary of task_id to task
        """
        pass
    
    @abc.abstractmethod
    def assign_task(self, task_id: str, vehicle_id: str) -> bool:
        """
        Manually assign a task to a vehicle.
        
        Args:
            task_id: ID of the task
            vehicle_id: ID of the vehicle
            
        Returns:
            bool: True if assignment was successful
            
        Raises:
            TaskNotFoundError: If task not found
            VehicleNotFoundError: If vehicle not found
        """
        pass
    
    @abc.abstractmethod
    def get_assignments(self) -> Dict[str, List[str]]:
        """
        Get current task assignments.
        
        Returns:
            Dict[str, List[str]]: Dictionary of vehicle_id to list of task_ids
        """
        pass
    
    @abc.abstractmethod
    def dispatch_cycle(self) -> None:
        """
        Execute a dispatch cycle.
        
        This is the main method that performs task assignments and
        coordinates vehicle movement.
        """
        pass
    
    @abc.abstractmethod
    def start(self) -> None:
        """
        Start the dispatcher.
        
        Begins continuous dispatching according to dispatch interval.
        """
        pass
    
    @abc.abstractmethod
    def stop(self) -> None:
        """
        Stop the dispatcher.
        
        Stops continuous dispatching.
        """
        pass
    
    @abc.abstractmethod
    def pause(self) -> None:
        """
        Pause the dispatcher.
        
        Temporarily pauses continuous dispatching.
        """
        pass
    
    @abc.abstractmethod
    def resume(self) -> None:
        """
        Resume the dispatcher.
        
        Resumes continuous dispatching after a pause.
        """
        pass
    
    @property
    def status(self) -> DispatchStatus:
        """Get the current status of the dispatcher."""
        return self._status
    
    def add_event_listener(self, event_type: EventType, listener: EventListener) -> None:
        """
        Add an event listener.
        
        Args:
            event_type: Type of event to listen for
            listener: Listener to add
        """
        self.event_dispatcher.add_listener(event_type, listener)
    
    def remove_event_listener(self, event_type: EventType, listener: EventListener) -> None:
        """
        Remove an event listener.
        
        Args:
            event_type: Type of event
            listener: Listener to remove
        """
        self.event_dispatcher.remove_listener(event_type, listener)
    
    def dispatch_event(self, event: DispatchEvent) -> None:
        """
        Dispatch an event.
        
        Args:
            event: Event to dispatch
        """
        self.event_dispatcher.dispatch_event(event)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dispatcher state to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "status": self._status.name,
            "last_dispatch_time": self._last_dispatch_time.isoformat() if self._last_dispatch_time else None,
            "config": {
                "dispatch_interval": self.config.dispatch_interval,
                "dispatch_strategy": self.config.dispatch_strategy.name,
                "max_tasks_per_vehicle": self.config.max_tasks_per_vehicle,
                "replan_on_change": self.config.replan_on_change,
                "conflict_resolution_enabled": self.config.conflict_resolution_enabled,
            }
        }