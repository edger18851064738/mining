"""
Transport task implementation for the mining dispatch system.

Provides task types for transporting materials between locations:
- Loading materials from source
- Transporting materials to destination
- Unloading materials at destination
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from datetime import datetime, timedelta
import uuid
import math

from utils.geo.coordinates import Point2D
from utils.logger import get_logger
from utils.io.serialization import Serializable

from domain.tasks.base import Task, TaskError
from domain.tasks.task_status import TaskStatus, TaskType, TaskPriority

# Initialize logger
logger = get_logger("transport_task")


class TransportTaskError(TaskError):
    """Exception raised for transport task specific errors."""
    pass


class TransportTask(Task):
    """
    Task for transporting material from one location to another.
    
    Represents a complete transport operation, including loading,
    transportation, and unloading phases.
    """
    
    def __init__(self, task_id: Optional[str] = None,
                start_point: Union[Point2D, Tuple[float, float]],
                end_point: Union[Point2D, Tuple[float, float]],
                task_type: str = "transport",
                material_type: str = "ore",
                amount: float = 50000.0,
                priority: Union[int, TaskPriority] = TaskPriority.NORMAL,
                deadline: Optional[datetime] = None):
        """
        Initialize a transport task.
        
        Args:
            task_id: Unique identifier for the task
            start_point: Starting location
            end_point: Destination location
            task_type: Type of transport task
            material_type: Type of material to transport
            amount: Amount of material in kg
            priority: Task priority
            deadline: Task deadline
        """
        # Convert priority enum to int if needed
        if isinstance(priority, TaskPriority):
            priority = priority.value
        
        # Initialize base class
        super().__init__(task_id, priority, deadline)
        
        # Convert points to Point2D
        self.start_point = self._to_point2d(start_point)
        self.end_point = self._to_point2d(end_point)
        self.waypoints = []
        
        # Transport specific attributes
        self.task_type = task_type
        self.material_type = material_type
        self.total_amount = amount
        self.remaining_amount = amount
        self.distance = self._calculate_distance()
        
        # Path planning
        self.path = []
        
        # Operational attributes
        self.load_time_estimate = self._estimate_load_time()
        self.transport_time_estimate = self._estimate_transport_time()
        self.unload_time_estimate = self._estimate_unload_time()
        
        # Current state tracking
        self.current_location = self.start_point
        self.current_phase = "preparation"  # preparation, loading, transport, unloading
        self.phase_progress = 0.0
    
    def _to_point2d(self, point: Union[Point2D, Tuple[float, float]]) -> Point2D:
        """Convert a point to Point2D."""
        if isinstance(point, Point2D):
            return point
        elif isinstance(point, tuple) and len(point) >= 2:
            return Point2D(point[0], point[1])
        else:
            raise ValueError(f"Invalid point format: {point}")
    
    def _calculate_distance(self) -> float:
        """Calculate the direct distance between start and end points."""
        return self.start_point.distance_to(self.end_point)
    
    def _estimate_load_time(self) -> float:
        """
        Estimate the time needed for loading.
        
        Returns:
            float: Estimated time in seconds
        """
        # Simplified model: 10 kg/s loading rate
        loading_rate = 10000.0  # kg/s
        return self.total_amount / loading_rate
    
    def _estimate_transport_time(self) -> float:
        """
        Estimate the time needed for transport.
        
        Returns:
            float: Estimated time in seconds
        """
        # Simplified model: 5 m/s average speed
        avg_speed = 5.0  # m/s
        
        # Calculate path distance (default to direct distance)
        path_distance = self.distance
        if self.path:
            path_distance = 0.0
            for i in range(len(self.path) - 1):
                path_distance += self.path[i].distance_to(self.path[i+1])
        
        return path_distance / avg_speed
    
    def _estimate_unload_time(self) -> float:
        """
        Estimate the time needed for unloading.
        
        Returns:
            float: Estimated time in seconds
        """
        # Simplified model: 15 kg/s unloading rate
        unloading_rate = 15000.0  # kg/s
        return self.total_amount / unloading_rate
    
    def get_duration_estimate(self) -> float:
        """
        Get an estimate of the total task duration.
        
        Returns:
            float: Estimated duration in seconds
        """
        return (self.load_time_estimate + 
                self.transport_time_estimate + 
                self.unload_time_estimate)
    
    def set_path(self, path: List[Union[Point2D, Tuple[float, float]]]) -> None:
        """
        Set the transport path.
        
        Args:
            path: List of path points
        """
        # Convert all points to Point2D
        self.path = [self._to_point2d(p) for p in path]
        
        # Recalculate transport time based on path
        self.transport_time_estimate = self._estimate_transport_time()
        
        logger.debug(f"Path set for task {self.task_id} with {len(path)} points")
    
    def add_waypoint(self, waypoint: Union[Point2D, Tuple[float, float]]) -> None:
        """
        Add a waypoint to the task.
        
        Args:
            waypoint: Waypoint location
        """
        self.waypoints.append(self._to_point2d(waypoint))
    
    def update_location(self, location: Union[Point2D, Tuple[float, float]]) -> None:
        """
        Update the current location during execution.
        
        Args:
            location: Current location
        """
        self.current_location = self._to_point2d(location)
        
        # Update progress based on phase and location
        if self.current_phase == "transport":
            self._update_transport_progress()
    
    def _update_transport_progress(self) -> None:
        """Update progress based on current location and path."""
        if not self.path:
            # Use direct distance if no path
            total_distance = self.distance
            traveled_distance = self.start_point.distance_to(self.current_location)
            self.phase_progress = min(1.0, traveled_distance / total_distance)
            return
        
        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0
        
        for i, point in enumerate(self.path):
            dist = point.distance_to(self.current_location)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Calculate distance traveled along path
        distance_traveled = 0.0
        for i in range(closest_idx):
            distance_traveled += self.path[i].distance_to(self.path[i+1])
        
        # Calculate total path length
        total_path_length = 0.0
        for i in range(len(self.path) - 1):
            total_path_length += self.path[i].distance_to(self.path[i+1])
        
        if total_path_length > 0:
            self.phase_progress = min(1.0, distance_traveled / total_path_length)
        else:
            self.phase_progress = 0.0
    
    def start_phase(self, phase: str) -> None:
        """
        Start a specific phase of the transport task.
        
        Args:
            phase: Phase name (preparation, loading, transport, unloading)
            
        Raises:
            TransportTaskError: If phase transition is invalid
        """
        valid_phases = {
            "preparation": ["loading"],
            "loading": ["transport"],
            "transport": ["unloading"],
            "unloading": []
        }
        
        if phase not in valid_phases.get(self.current_phase, []):
            raise TransportTaskError(
                f"Invalid phase transition: {self.current_phase} -> {phase}"
            )
        
        self.current_phase = phase
        self.phase_progress = 0.0
        
        logger.info(f"Task {self.task_id} entered {phase} phase")
        
        # Update overall task progress
        self._update_task_progress()
    
    def update_phase_progress(self, progress: float) -> None:
        """
        Update the progress of the current phase.
        
        Args:
            progress: Progress value (0-1)
        """
        self.phase_progress = max(0.0, min(1.0, progress))
        
        # Update overall task progress
        self._update_task_progress()
        
        # Auto-transition to next phase if complete
        if math.isclose(self.phase_progress, 1.0):
            self._complete_current_phase()
    
    def _update_task_progress(self) -> None:
        """Update overall task progress based on phase progress."""
        # Assign weights to each phase
        phase_weights = {
            "preparation": 0.05,
            "loading": 0.25,
            "transport": 0.45,
            "unloading": 0.25
        }
        
        # Calculate completed phases
        completed_weight = 0.0
        
        if self.current_phase == "loading":
            completed_weight = phase_weights["preparation"]
        elif self.current_phase == "transport":
            completed_weight = phase_weights["preparation"] + phase_weights["loading"]
        elif self.current_phase == "unloading":
            completed_weight = phase_weights["preparation"] + phase_weights["loading"] + phase_weights["transport"]
        
        # Add current phase progress
        current_phase_contribution = self.phase_progress * phase_weights.get(self.current_phase, 0.0)
        
        total_progress = completed_weight + current_phase_contribution
        
        # Update base class progress
        super().update_progress(total_progress)
    
    def _complete_current_phase(self) -> None:
        """Handle completion of the current phase."""
        if self.current_phase == "preparation":
            self.start_phase("loading")
        elif self.current_phase == "loading":
            self.start_phase("transport")
        elif self.current_phase == "transport":
            self.start_phase("unloading")
        elif self.current_phase == "unloading":
            # All phases complete, complete task
            self.complete()
    
    def update_amount(self, remaining: float) -> None:
        """
        Update the remaining amount of material.
        
        Args:
            remaining: Remaining amount in kg
        """
        self.remaining_amount = max(0.0, min(remaining, self.total_amount))
        
        # Update phase progress based on amount
        if self.current_phase == "loading":
            loaded = self.total_amount - self.remaining_amount
            self.phase_progress = loaded / self.total_amount
        elif self.current_phase == "unloading":
            unloaded = self.total_amount - self.remaining_amount
            self.phase_progress = unloaded / self.total_amount
        
        # Update overall task progress
        self._update_task_progress()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert transport task to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        # Get base class dict
        base_dict = super().to_dict()
        
        # Add transport-specific fields
        transport_dict = {
            'start_point': {
                'x': self.start_point.x,
                'y': self.start_point.y
            },
            'end_point': {
                'x': self.end_point.x,
                'y': self.end_point.y
            },
            'task_type': self.task_type,
            'material_type': self.material_type,
            'total_amount': self.total_amount,
            'remaining_amount': self.remaining_amount,
            'distance': self.distance,
            'current_phase': self.current_phase,
            'phase_progress': self.phase_progress,
            'waypoints': [{'x': p.x, 'y': p.y} for p in self.waypoints],
            'path': [{'x': p.x, 'y': p.y} for p in self.path] if self.path else [],
            'current_location': {
                'x': self.current_location.x,
                'y': self.current_location.y
            }
        }
        
        # Merge dictionaries
        return {**base_dict, **transport_dict}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransportTask':
        """
        Create a transport task from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            TransportTask: New transport task instance
        """
        # Create a new task with required parameters
        start_point = None
        if 'start_point' in data:
            start_point = Point2D(data['start_point']['x'], data['start_point']['y'])
            
        end_point = None
        if 'end_point' in data:
            end_point = Point2D(data['end_point']['x'], data['end_point']['y'])
            
        if start_point is None or end_point is None:
            raise ValueError("Missing required start_point or end_point in data")
        
        # Parse deadline if present
        deadline = None
        if data.get('deadline'):
            try:
                deadline = datetime.fromisoformat(data['deadline'])
            except (ValueError, TypeError):
                pass
        
        # Create task instance
        task = cls(
            task_id=data.get('task_id'),
            start_point=start_point,
            end_point=end_point,
            task_type=data.get('task_type', 'transport'),
            material_type=data.get('material_type', 'ore'),
            amount=data.get('total_amount', 50000.0),
            priority=data.get('priority', 1),
            deadline=deadline
        )
        
        # Set additional fields
        task._status = data.get('status', TaskStatus.PENDING.name)
        task.remaining_amount = data.get('remaining_amount', task.total_amount)
        task.current_phase = data.get('current_phase', 'preparation')
        task.phase_progress = data.get('phase_progress', 0.0)
        
        # Set timestamps
        if data.get('creation_time'):
            try:
                task.creation_time = datetime.fromisoformat(data['creation_time'])
            except (ValueError, TypeError):
                pass
                
        if data.get('assigned_time'):
            try:
                task.assigned_time = datetime.fromisoformat(data['assigned_time'])
            except (ValueError, TypeError):
                pass
                
        if data.get('start_time'):
            try:
                task.start_time = datetime.fromisoformat(data['start_time'])
            except (ValueError, TypeError):
                pass
                
        if data.get('end_time'):
            try:
                task.end_time = datetime.fromisoformat(data['end_time'])
            except (ValueError, TypeError):
                pass
        
        # Set assignee
        task.assigned_to = data.get('assigned_to')
        
        # Set progress
        task.progress = data.get('progress', 0.0)
        
        # Set retry count
        task.retry_count = data.get('retry_count', 0)
        
        # Set waypoints
        if 'waypoints' in data:
            for wp in data['waypoints']:
                task.add_waypoint(Point2D(wp['x'], wp['y']))
        
        # Set path
        if 'path' in data and data['path']:
            path = [Point2D(p['x'], p['y']) for p in data['path']]
            task.set_path(path)
        
        # Set current location
        if 'current_location' in data:
            loc = data['current_location']
            task.current_location = Point2D(loc['x'], loc['y'])
        
        return task
    
    def __repr__(self) -> str:
        """String representation of the transport task."""
        return (f"TransportTask(id={self.task_id}, type={self.task_type}, "
                f"status={self.status}, progress={self.progress:.1f}, "
                f"phase={self.current_phase})")


class LoadingTask(TransportTask):
    """
    Specialized task for loading material at a source location.
    
    Focuses on the loading phase of a transport operation.
    """
    
    def __init__(self, task_id: Optional[str] = None,
                loading_point: Union[Point2D, Tuple[float, float]],
                material_type: str = "ore",
                amount: float = 50000.0,
                priority: Union[int, TaskPriority] = TaskPriority.NORMAL,
                deadline: Optional[datetime] = None):
        """
        Initialize a loading task.
        
        Args:
            task_id: Unique identifier for the task
            loading_point: Loading location
            material_type: Type of material to load
            amount: Amount of material in kg
            priority: Task priority
            deadline: Task deadline
        """
        # For loading task, both start and end point are the loading point
        super().__init__(
            task_id=task_id,
            start_point=loading_point,
            end_point=loading_point,
            task_type="loading",
            material_type=material_type,
            amount=amount,
            priority=priority,
            deadline=deadline
        )
        
        # Loading specific attributes
        self.loading_point = self._to_point2d(loading_point)
        self.loading_rate = 10000.0  # kg/s
        
        # Skip preparation and transport phases
        self.current_phase = "loading"
    
    def get_duration_estimate(self) -> float:
        """
        Get an estimate of the loading task duration.
        
        Returns:
            float: Estimated duration in seconds
        """
        return self.total_amount / self.loading_rate
    
    def update_loading_rate(self, rate: float) -> None:
        """
        Update the loading rate.
        
        Args:
            rate: New loading rate in kg/s
        """
        self.loading_rate = max(1.0, rate)
        self.load_time_estimate = self.total_amount / self.loading_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert loading task to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        base_dict = super().to_dict()
        
        # Add loading-specific fields
        loading_dict = {
            'loading_point': {
                'x': self.loading_point.x,
                'y': self.loading_point.y
            },
            'loading_rate': self.loading_rate
        }
        
        return {**base_dict, **loading_dict}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoadingTask':
        """
        Create a loading task from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            LoadingTask: New loading task instance
        """
        # Extract loading point
        loading_point = None
        if 'loading_point' in data:
            loading_point = Point2D(data['loading_point']['x'], data['loading_point']['y'])
        elif 'start_point' in data:
            loading_point = Point2D(data['start_point']['x'], data['start_point']['y'])
        
        if loading_point is None:
            raise ValueError("Missing required loading_point in data")
        
        # Parse deadline if present
        deadline = None
        if data.get('deadline'):
            try:
                deadline = datetime.fromisoformat(data['deadline'])
            except (ValueError, TypeError):
                pass
        
        # Create task instance
        task = cls(
            task_id=data.get('task_id'),
            loading_point=loading_point,
            material_type=data.get('material_type', 'ore'),
            amount=data.get('total_amount', 50000.0),
            priority=data.get('priority', 1),
            deadline=deadline
        )
        
        # Set additional fields
        task._status = data.get('status', TaskStatus.PENDING.name)
        task.remaining_amount = data.get('remaining_amount', task.total_amount)
        task.current_phase = data.get('current_phase', 'loading')
        task.phase_progress = data.get('phase_progress', 0.0)
        task.loading_rate = data.get('loading_rate', 10000.0)
        
        # Set timestamps and other fields from base class
        # (same as in TransportTask.from_dict)
        
        return task


class UnloadingTask(TransportTask):
    """
    Specialized task for unloading material at a destination location.
    
    Focuses on the unloading phase of a transport operation.
    """
    
    def __init__(self, task_id: Optional[str] = None,
                unloading_point: Union[Point2D, Tuple[float, float]],
                material_type: str = "ore",
                amount: float = 50000.0,
                priority: Union[int, TaskPriority] = TaskPriority.NORMAL,
                deadline: Optional[datetime] = None):
        """
        Initialize an unloading task.
        
        Args:
            task_id: Unique identifier for the task
            unloading_point: Unloading location
            material_type: Type of material to unload
            amount: Amount of material in kg
            priority: Task priority
            deadline: Task deadline
        """
        # For unloading task, both start and end point are the unloading point
        super().__init__(
            task_id=task_id,
            start_point=unloading_point,
            end_point=unloading_point,
            task_type="unloading",
            material_type=material_type,
            amount=amount,
            priority=priority,
            deadline=deadline
        )
        
        # Unloading specific attributes
        self.unloading_point = self._to_point2d(unloading_point)
        self.unloading_rate = 15000.0  # kg/s
        
        # Skip preparation and transport phases
        self.current_phase = "unloading"
    
    def get_duration_estimate(self) -> float:
        """
        Get an estimate of the unloading task duration.
        
        Returns:
            float: Estimated duration in seconds
        """
        return self.total_amount / self.unloading_rate
    
    def update_unloading_rate(self, rate: float) -> None:
        """
        Update the unloading rate.
        
        Args:
            rate: New unloading rate in kg/s
        """
        self.unloading_rate = max(1.0, rate)
        self.unload_time_estimate = self.total_amount / self.unloading_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert unloading task to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        base_dict = super().to_dict()
        
        # Add unloading-specific fields
        unloading_dict = {
            'unloading_point': {
                'x': self.unloading_point.x,
                'y': self.unloading_point.y
            },
            'unloading_rate': self.unloading_rate
        }
        
        return {**base_dict, **unloading_dict}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnloadingTask':
        """
        Create an unloading task from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            UnloadingTask: New unloading task instance
        """
        # Extract unloading point
        unloading_point = None
        if 'unloading_point' in data:
            unloading_point = Point2D(data['unloading_point']['x'], data['unloading_point']['y'])
        elif 'end_point' in data:
            unloading_point = Point2D(data['end_point']['x'], data['end_point']['y'])
        
        if unloading_point is None:
            raise ValueError("Missing required unloading_point in data")
        
        # Parse deadline if present
        deadline = None
        if data.get('deadline'):
            try:
                deadline = datetime.fromisoformat(data['deadline'])
            except (ValueError, TypeError):
                pass
        
        # Create task instance
        task = cls(
            task_id=data.get('task_id'),
            unloading_point=unloading_point,
            material_type=data.get('material_type', 'ore'),
            amount=data.get('total_amount', 50000.0),
            priority=data.get('priority', 1),
            deadline=deadline
        )
        
        # Set additional fields
        task._status = data.get('status', TaskStatus.PENDING.name)
        task.remaining_amount = data.get('remaining_amount', task.total_amount)
        task.current_phase = data.get('current_phase', 'unloading')
        task.phase_progress = data.get('phase_progress', 0.0)
        task.unloading_rate = data.get('unloading_rate', 15000.0)
        
        # Set timestamps and other fields from base class
        # (same as in TransportTask.from_dict)
        
        return task