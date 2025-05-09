"""
Event definitions for the dispatch system.

Defines events that can occur during the dispatch process,
including vehicle state changes, task status changes, and dispatching decisions.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import needed domain models
from utils.geo.coordinates import Point2D


class EventType(Enum):
    """Types of events in the dispatch system."""
    # General event type for listeners that want all events
    ALL = auto()
    
    # Vehicle events
    VEHICLE_ADDED = auto()
    VEHICLE_REMOVED = auto()
    VEHICLE_STATE_CHANGED = auto()
    VEHICLE_POSITION_CHANGED = auto()
    
    # Task events
    TASK_ADDED = auto()
    TASK_ASSIGNED = auto()
    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    TASK_CANCELED = auto()
    
    # Dispatch events
    DISPATCH_CYCLE_STARTED = auto()
    DISPATCH_CYCLE_COMPLETED = auto()
    DISPATCH_DECISION_MADE = auto()
    
    # Path events
    PATH_PLANNED = auto()
    PATH_REPLANNED = auto()
    
    # Conflict events
    CONFLICT_DETECTED = auto()
    CONFLICT_RESOLVED = auto()
    
    # System events
    SYSTEM_ERROR = auto()
    SYSTEM_WARNING = auto()
    SYSTEM_INFO = auto()


@dataclass
class DispatchEvent:
    """Base class for all dispatch events."""
    event_type: EventType
    timestamp: datetime = None
    
    def __post_init__(self):
        """Initialize with current timestamp if none provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VehicleEvent(DispatchEvent):
    """Event related to a vehicle."""
    vehicle_id: str = None 
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        result = super().to_dict()
        result.update({
            "vehicle_id": self.vehicle_id,
        })
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class TaskEvent(DispatchEvent):
    """Event related to a task."""
    task_id: str = None 
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        result = super().to_dict()
        result.update({
            "task_id": self.task_id,
        })
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class AssignmentEvent(DispatchEvent):
    """Event related to task assignment."""
    task_id: str = None 
    vehicle_id: str = None 
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        result = super().to_dict()
        result.update({
            "task_id": self.task_id,
            "vehicle_id": self.vehicle_id,
        })
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class PathEvent(DispatchEvent):
    """Event related to path planning."""
    vehicle_id: str = None 
    path_points: List[Point2D] = None 
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        result = super().to_dict()
        result.update({
            "vehicle_id": self.vehicle_id,
            "path_points": [(p.x, p.y) for p in self.path_points],
        })
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class ConflictEvent(DispatchEvent):
    """Event related to conflict detection and resolution."""
    vehicle_ids: List[str] = None 
    conflict_location: Optional[Point2D] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        result = super().to_dict()
        result.update({
            "vehicle_ids": self.vehicle_ids,
        })
        if self.conflict_location:
            result["conflict_location"] = (self.conflict_location.x, self.conflict_location.y)
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class SystemEvent(DispatchEvent):
    """Event related to system operations."""
    message: str = None 
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        result = super().to_dict()
        result.update({
            "message": self.message,
        })
        if self.details:
            result["details"] = self.details
        return result


class EventListener:
    """Interface for event listeners."""
    
    def on_event(self, event: DispatchEvent) -> None:
        """
        Handle a dispatch event.
        
        Args:
            event: The event to handle
        """
        pass


class EventDispatcher:
    """
    Dispatches events to registered listeners.
    
    Provides a centralized event handling system for the dispatcher.
    """
    
    def __init__(self):
        """Initialize the event dispatcher."""
        self._listeners = {}
    
    def add_listener(self, event_type: EventType, listener: EventListener) -> None:
        """
        Add a listener for a specific event type.
        
        Args:
            event_type: Type of event to listen for
            listener: Listener to notify
        """
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        
        if listener not in self._listeners[event_type]:
            self._listeners[event_type].append(listener)
    
    def remove_listener(self, event_type: EventType, listener: EventListener) -> None:
        """
        Remove a listener for a specific event type.
        
        Args:
            event_type: Type of event
            listener: Listener to remove
        """
        if event_type in self._listeners and listener in self._listeners[event_type]:
            self._listeners[event_type].remove(listener)
    
    def dispatch_event(self, event: DispatchEvent) -> None:
        """
        Dispatch an event to all registered listeners.
        
        Args:
            event: Event to dispatch
        """
        # Notify listeners for this specific event type
        if event.event_type in self._listeners:
            for listener in self._listeners[event.event_type]:
                listener.on_event(event)
        
        # Also notify listeners for all events
        if EventType.ALL in self._listeners:
            for listener in self._listeners[EventType.ALL]:
                listener.on_event(event)