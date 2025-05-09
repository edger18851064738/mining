"""
Mining dispatcher implementation.

Provides a concrete implementation of the Dispatcher interface
for coordinating vehicles in a mining environment.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import threading
import time
import heapq
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable

from utils.logger import get_logger, timed
from utils.geo.coordinates import Point2D
from utils.math.trajectories import Path

from coordination.dispatcher.base import (
    Dispatcher, DispatcherConfig, DispatchStatus, DispatchStrategy,
    VehicleNotFoundError, TaskNotFoundError
)
from coordination.dispatcher.dispatch_events import (
    EventType, DispatchEvent, VehicleEvent, TaskEvent, 
    AssignmentEvent, PathEvent, ConflictEvent, SystemEvent
)

# These will be properly implemented when we have the corresponding modules
# For now we'll use placeholders that will be replaced later
try:
    from domain.vehicles.vehicle_state import VehicleState
    from domain.tasks.task_status import TaskStatus, TaskPriority
except ImportError:
    # Create placeholder enums if real ones aren't available yet
    from enum import Enum, auto
    
    class VehicleState(Enum):
        IDLE = auto()
        MOVING = auto()
        LOADING = auto()
        UNLOADING = auto()
        WAITING = auto()
    
    class TaskStatus(Enum):
        PENDING = auto()
        ASSIGNED = auto()
        IN_PROGRESS = auto()
        COMPLETED = auto()
        FAILED = auto()
        CANCELED = auto()
    
    class TaskPriority(Enum):
        LOW = 1
        NORMAL = 3
        HIGH = 5
        URGENT = 8
        CRITICAL = 10


class MiningDispatcher(Dispatcher):
    """
    Dispatcher for mining operations.
    
    Coordinates mining vehicles for tasks like ore transport,
    loading, and unloading.
    """
    
    def __init__(self, environment=None, config: Optional[DispatcherConfig] = None):
        """
        Initialize the mining dispatcher.
        
        Args:
            environment: Mining environment
            config: Dispatcher configuration
        """
        super().__init__(config or DispatcherConfig())
        self.logger = get_logger("MiningDispatcher")
        
        # Store environment
        self.environment = environment
        
        # Initialize collections
        self._vehicles = {}  # vehicle_id -> vehicle
        self._tasks = {}     # task_id -> task
        self._assignments = {}  # vehicle_id -> [task_id, ...]
        self._pending_tasks = []  # Priority queue of pending tasks
        
        # Initialize dispatch thread
        self._dispatch_thread = None
        self._stop_event = threading.Event()
        
        # Initialize task allocator based on strategy
        self._init_allocator()
        
        # Initialize conflict resolver if enabled
        self._init_conflict_resolver()
        
        # Log initialization
        self.logger.info("Mining dispatcher initialized")
    
    def _init_allocator(self) -> None:
        """Initialize the task allocator based on configuration."""
        strategy = self.config.dispatch_strategy
        
        # Create allocator based on strategy
        # This will be replaced with actual allocator implementations when they're available
        if strategy == DispatchStrategy.PRIORITY:
            # self.allocator = PriorityAllocator()
            self.logger.info("Using priority-based task allocator")
            self.allocator = None  # Placeholder until implementation
        elif strategy == DispatchStrategy.OPTIMIZED:
            # self.allocator = MIQPAllocator()
            self.logger.info("Using optimized (MIQP) task allocator")
            self.allocator = None  # Placeholder until implementation
        else:
            # Default to simple allocator for now
            self.logger.info(f"Using default allocator for strategy: {strategy.name}")
            self.allocator = None  # Placeholder for simple allocator
    
    def _init_conflict_resolver(self) -> None:
        """Initialize the conflict resolver if enabled."""
        if self.config.conflict_resolution_enabled:
            # self.conflict_resolver = CBSResolver()
            self.logger.info("Conflict resolution enabled (CBS)")
            self.conflict_resolver = None  # Placeholder until implementation
        else:
            self.logger.info("Conflict resolution disabled")
            self.conflict_resolver = None
    
    def add_vehicle(self, vehicle) -> None:
        """
        Add a vehicle to the dispatch system.
        
        Args:
            vehicle: Vehicle to add
        """
        vehicle_id = vehicle.vehicle_id
        
        # Check if vehicle already exists
        if vehicle_id in self._vehicles:
            self.logger.warning(f"Vehicle {vehicle_id} already exists, updating")
        
        # Add vehicle
        self._vehicles[vehicle_id] = vehicle
        self._assignments[vehicle_id] = []
        
        # Dispatch event
        self.dispatch_event(VehicleEvent(
            event_type=EventType.VEHICLE_ADDED,
            vehicle_id=vehicle_id,
            details={"position": vehicle.current_location.as_tuple() if hasattr(vehicle, "current_location") else None}
        ))
        
        self.logger.info(f"Added vehicle {vehicle_id}")
    
    def remove_vehicle(self, vehicle_id: str) -> bool:
        """
        Remove a vehicle from the dispatch system.
        
        Args:
            vehicle_id: ID of the vehicle to remove
            
        Returns:
            bool: True if vehicle was removed, False if not found
        """
        if vehicle_id not in self._vehicles:
            return False
        
        # Get assigned tasks
        assigned_tasks = self._assignments.get(vehicle_id, [])
        
        # Unassign tasks and move back to pending
        for task_id in assigned_tasks:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if task.status == TaskStatus.ASSIGNED:
                    task._set_status(TaskStatus.PENDING)
                    # Add back to pending queue
                    heapq.heappush(self._pending_tasks, (task.priority.value, task.task_id))
        
        # Remove vehicle and assignments
        del self._vehicles[vehicle_id]
        if vehicle_id in self._assignments:
            del self._assignments[vehicle_id]
        
        # Dispatch event
        self.dispatch_event(VehicleEvent(
            event_type=EventType.VEHICLE_REMOVED,
            vehicle_id=vehicle_id
        ))
        
        self.logger.info(f"Removed vehicle {vehicle_id}")
        return True
    
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
        if vehicle_id not in self._vehicles:
            raise VehicleNotFoundError(f"Vehicle {vehicle_id} not found")
        
        return self._vehicles[vehicle_id]
    
    def get_all_vehicles(self) -> dict:
        """
        Get all vehicles.
        
        Returns:
            Dict[str, Vehicle]: Dictionary of vehicle_id to vehicle
        """
        return self._vehicles.copy()
    
    def add_task(self, task) -> None:
        """
        Add a task to the dispatch system.
        
        Args:
            task: Task to add
        """
        task_id = task.task_id
        
        # Check if task already exists
        if task_id in self._tasks:
            self.logger.warning(f"Task {task_id} already exists, updating")
        
        # Add task
        self._tasks[task_id] = task
        
        # If task is pending, add to pending queue
        if task.status == TaskStatus.PENDING:
            heapq.heappush(self._pending_tasks, (task.priority.value, task_id))
        
        # Dispatch event
        self.dispatch_event(TaskEvent(
            event_type=EventType.TASK_ADDED,
            task_id=task_id,
            details={"status": task.status, "priority": str(task.priority)}
        ))
        
        self.logger.info(f"Added task {task_id} with priority {task.priority}")
    
    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the dispatch system.
        
        Args:
            task_id: ID of the task to remove
            
        Returns:
            bool: True if task was removed, False if not found
        """
        if task_id not in self._tasks:
            return False
        
        # Get assigned vehicle, if any
        assigned_vehicle = None
        for vehicle_id, tasks in self._assignments.items():
            if task_id in tasks:
                assigned_vehicle = vehicle_id
                break
        
        # Remove task from assignments
        if assigned_vehicle:
            self._assignments[assigned_vehicle].remove(task_id)
        
        # Remove task
        del self._tasks[task_id]
        
        # Update pending tasks - rebuild if necessary
        self._rebuild_pending_tasks()
        
        # Dispatch event
        self.dispatch_event(TaskEvent(
            event_type=EventType.TASK_CANCELED,
            task_id=task_id,
            details={"vehicle_id": assigned_vehicle}
        ))
        
        self.logger.info(f"Removed task {task_id}")
        return True
    
    def _rebuild_pending_tasks(self) -> None:
        """Rebuild the pending tasks queue."""
        self._pending_tasks = []
        for task_id, task in self._tasks.items():
            if task.status == TaskStatus.PENDING:
                heapq.heappush(self._pending_tasks, (task.priority.value, task_id))
    
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
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task {task_id} not found")
        
        return self._tasks[task_id]
    
    def get_all_tasks(self) -> dict:
        """
        Get all tasks.
        
        Returns:
            Dict[str, Task]: Dictionary of task_id to task
        """
        return self._tasks.copy()
    
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
        # Check if task and vehicle exist
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task {task_id} not found")
        
        if vehicle_id not in self._vehicles:
            raise VehicleNotFoundError(f"Vehicle {vehicle_id} not found")
        
        # Get task and vehicle
        task = self._tasks[task_id]
        vehicle = self._vehicles[vehicle_id]
        
        # Check if task is already assigned
        for v_id, tasks in self._assignments.items():
            if task_id in tasks and v_id != vehicle_id:
                # Unassign from current vehicle
                self._assignments[v_id].remove(task_id)
                self.logger.info(f"Unassigned task {task_id} from vehicle {v_id}")
        
        # Check if vehicle has capacity
        if len(self._assignments[vehicle_id]) >= self.config.max_tasks_per_vehicle:
            self.logger.warning(f"Vehicle {vehicle_id} is at capacity, cannot assign task {task_id}")
            return False
        
        # Assign task to vehicle
        if task_id not in self._assignments[vehicle_id]:
            self._assignments[vehicle_id].append(task_id)
        
        # Update task status
        if task.status == TaskStatus.PENDING:
            task.assign(vehicle_id)
        
        # Remove from pending tasks
        self._rebuild_pending_tasks()
        
        # Dispatch event
        self.dispatch_event(AssignmentEvent(
            event_type=EventType.TASK_ASSIGNED,
            task_id=task_id,
            vehicle_id=vehicle_id,
            details={"task_type": task.__class__.__name__}
        ))
        
        self.logger.info(f"Assigned task {task_id} to vehicle {vehicle_id}")
        return True
    
    def get_assignments(self) -> Dict[str, List[str]]:
        """
        Get current task assignments.
        
        Returns:
            Dict[str, List[str]]: Dictionary of vehicle_id to list of task_ids
        """
        return {v_id: tasks.copy() for v_id, tasks in self._assignments.items()}
    
    @timed("dispatch_cycle")
    def dispatch_cycle(self) -> None:
        """
        Execute a dispatch cycle.
        
        This is the main method that performs task assignments and
        coordinates vehicle movement.
        """
        # Skip if not running
        if self._status != DispatchStatus.RUNNING:
            return
        
        # Record start time
        start_time = datetime.now()
        self._last_dispatch_time = start_time
        
        # Dispatch event
        self.dispatch_event(DispatchEvent(
            event_type=EventType.DISPATCH_CYCLE_STARTED,
            timestamp=start_time
        ))
        
        try:
            # Update vehicle states
            self._update_vehicles()
            
            # Update task states
            self._update_tasks()
            
            # Perform task allocation
            self._allocate_tasks()
            
            # Plan paths
            self._plan_paths()
            
            # Check for conflicts
            if self.config.conflict_resolution_enabled and self.conflict_resolver:
                self._resolve_conflicts()
            
        except Exception as e:
            self.logger.error(f"Error in dispatch cycle: {str(e)}", exc_info=True)
            self._status = DispatchStatus.ERROR
            
            # Dispatch error event
            self.dispatch_event(SystemEvent(
                event_type=EventType.SYSTEM_ERROR,
                message=f"Error in dispatch cycle: {str(e)}"
            ))
        
        # Record completion
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Dispatch completion event
        self.dispatch_event(DispatchEvent(
            event_type=EventType.DISPATCH_CYCLE_COMPLETED,
            timestamp=end_time
        ))
        
        self.logger.info(f"Dispatch cycle completed in {duration:.3f} seconds")
    
    def _update_vehicles(self) -> None:
        """Update vehicle states and positions."""
        for vehicle_id, vehicle in self._vehicles.items():
            # Save previous state and position for change detection
            prev_state = vehicle.state if hasattr(vehicle, 'state') else None
            prev_pos = vehicle.current_location if hasattr(vehicle, 'current_location') else None
            
            # Update vehicle position (if it has an update_position method)
            if hasattr(vehicle, 'update_position') and callable(vehicle.update_position):
                vehicle.update_position()
            
            # Check for state change
            if hasattr(vehicle, 'state') and prev_state != vehicle.state:
                self.dispatch_event(VehicleEvent(
                    event_type=EventType.VEHICLE_STATE_CHANGED,
                    vehicle_id=vehicle_id,
                    details={
                        "old_state": prev_state.name if prev_state else None,
                        "new_state": vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                    }
                ))
            
            # Check for position change
            if (hasattr(vehicle, 'current_location') and prev_pos and 
                prev_pos.distance_to(vehicle.current_location) > 0.1):
                self.dispatch_event(VehicleEvent(
                    event_type=EventType.VEHICLE_POSITION_CHANGED,
                    vehicle_id=vehicle_id,
                    details={
                        "old_position": prev_pos.as_tuple(),
                        "new_position": vehicle.current_location.as_tuple()
                    }
                ))
    
    def _update_tasks(self) -> None:
        """Update task states and check for completions."""
        for task_id, task in list(self._tasks.items()):  # Use list to allow dict modification
            # Skip tasks that are not active
            if task.status not in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
                continue
            
            # Check for task completion
            if task.status == TaskStatus.COMPLETED:
                # Task was completed during vehicle update
                self.dispatch_event(TaskEvent(
                    event_type=EventType.TASK_COMPLETED,
                    task_id=task_id
                ))
                
                # Find assigned vehicle and remove task
                for vehicle_id, tasks in self._assignments.items():
                    if task_id in tasks:
                        tasks.remove(task_id)
                        break
            
            # Check for task failure
            elif task.status == TaskStatus.FAILED:
                # Task failed during vehicle update
                self.dispatch_event(TaskEvent(
                    event_type=EventType.TASK_FAILED,
                    task_id=task_id,
                    details={"reason": getattr(task, "failure_reason", "Unknown reason")}
                ))
                
                # Find assigned vehicle and remove task
                for vehicle_id, tasks in self._assignments.items():
                    if task_id in tasks:
                        tasks.remove(task_id)
                        break
    
    def _allocate_tasks(self) -> None:
        """Allocate pending tasks to available vehicles."""
        # Check if we have pending tasks and available vehicles
        if not self._pending_tasks:
            return
        
        # Get available vehicles
        available_vehicles = {
            v_id: vehicle for v_id, vehicle in self._vehicles.items()
            if (hasattr(vehicle, 'is_available') and vehicle.is_available and 
                len(self._assignments[v_id]) < self.config.max_tasks_per_vehicle)
        }
        
        if not available_vehicles:
            return
        
        # Get pending tasks
        pending_tasks = {}
        pending_task_items = self._pending_tasks.copy()  
        for _, task_id in pending_task_items:
            if task_id in self._tasks and self._tasks[task_id].status == TaskStatus.PENDING:
                pending_tasks[task_id] = self._tasks[task_id]
        
        if not pending_tasks:
            return
        
        # Perform allocation (ideally using allocator)
        if self.allocator:
            # TODO: Implement with actual allocator
            allocation_result = {}  # result from allocator would go here
        else:
            # Simple allocation strategy (priority FIFO)
            allocation_result = self._simple_allocation(available_vehicles, pending_tasks)
        
        # Apply allocations
        for vehicle_id, task_ids in allocation_result.items():
            for task_id in task_ids:
                self.assign_task(task_id, vehicle_id)
    
    def _simple_allocation(self, available_vehicles, pending_tasks):
        """
        Simple allocation strategy (priority FIFO).
        
        This is a fallback when no allocator is available.
        """
        result = {v_id: [] for v_id in available_vehicles}
        
        # Sort vehicles by number of assigned tasks (ascending)
        vehicles_by_load = sorted(
            available_vehicles.keys(), 
            key=lambda v_id: len(self._assignments[v_id])
        )
        
        # Sort tasks by priority (descending) and creation time (ascending)
        sorted_tasks = sorted(
            pending_tasks.values(),
            key=lambda t: (-t.priority.value, getattr(t, 'creation_time', datetime.now()))
        )
        
        # Assign tasks to vehicles in order
        task_index = 0
        while task_index < len(sorted_tasks):
            # No more vehicles with capacity
            if not vehicles_by_load:
                break
                
            # Get next task and vehicle
            task = sorted_tasks[task_index]
            vehicle_id = vehicles_by_load[0]
            vehicle = available_vehicles[vehicle_id]
            
            # Check if vehicle can handle this task
            if not self._can_handle_task(vehicle, task):
                # Move to next task
                task_index += 1
                continue
            
            # Assign task to vehicle
            result[vehicle_id].append(task.task_id)
            
            # Update vehicle load and reorder
            vehicles_by_load.pop(0)
            
            # If vehicle still has capacity, reinsert it
            if len(result[vehicle_id]) + len(self._assignments[vehicle_id]) < self.config.max_tasks_per_vehicle:
                # Find insertion point (sort by current load)
                insert_pos = 0
                while (insert_pos < len(vehicles_by_load) and 
                       len(result[vehicles_by_load[insert_pos]]) + len(self._assignments[vehicles_by_load[insert_pos]]) <= 
                       len(result[vehicle_id]) + len(self._assignments[vehicle_id])):
                    insert_pos += 1
                
                vehicles_by_load.insert(insert_pos, vehicle_id)
            
            # Move to next task
            task_index += 1
        
        return result
    
    def _can_handle_task(self, vehicle, task):
        """
        Check if a vehicle can handle a task.
        
        Args:
            vehicle: The vehicle
            task: The task
            
        Returns:
            bool: True if vehicle can handle task
        """
        # Check vehicle type compatibility with task type
        if hasattr(vehicle, 'vehicle_type') and hasattr(task, 'required_vehicle_type'):
            if task.required_vehicle_type and vehicle.vehicle_type != task.required_vehicle_type:
                return False
        
        # Check capacity
        if hasattr(task, 'amount') and hasattr(vehicle, 'max_capacity'):
            if task.amount > vehicle.max_capacity:
                return False
        
        # Check terrain capability
        if (hasattr(task, 'location') and hasattr(vehicle, 'terrain_capability') and 
            self.environment and hasattr(self.environment, 'get_terrain_property')):
            task_location = task.location if hasattr(task, 'location') else None
            if task_location:
                hardness = self.environment.get_terrain_property(task_location, 'hardness')
                if hardness > vehicle.terrain_capability:
                    return False
        
        return True
    
    def _plan_paths(self) -> None:
        """Plan paths for vehicles with assigned tasks."""
        for vehicle_id, task_ids in self._assignments.items():
            vehicle = self._vehicles[vehicle_id]
            
            # Skip if no tasks or vehicle is not available
            if not task_ids or not getattr(vehicle, 'is_available', True):
                continue
            
            # Skip if vehicle already has a path
            if hasattr(vehicle, 'current_path') and vehicle.current_path:
                continue
            
            # Get next task
            next_task_id = task_ids[0]
            task = self._tasks[next_task_id]
            
            # Get destination based on task type
            destination = None
            if hasattr(task, 'location'):
                destination = task.location
            elif hasattr(task, 'start_point'):
                destination = task.start_point
            
            if not destination:
                continue
            
            # Skip if vehicle is already at destination
            if (hasattr(vehicle, 'current_location') and vehicle.current_location and
                vehicle.current_location.distance_to(destination) < 1.0):
                continue
            
            # Calculate path
            path = None
            if hasattr(vehicle, 'calculate_path_to') and callable(vehicle.calculate_path_to):
                path = vehicle.calculate_path_to(destination)
            elif self.environment and hasattr(self.environment, 'find_path'):
                path = self.environment.find_path(vehicle.current_location, destination, vehicle)
            
            if not path:
                self.logger.warning(f"Failed to plan path for vehicle {vehicle_id} to task {next_task_id}")
                continue
            
            # Assign path to vehicle
            if hasattr(vehicle, 'assign_path') and callable(vehicle.assign_path):
                vehicle.assign_path(path)
                
                # Dispatch event
                self.dispatch_event(PathEvent(
                    event_type=EventType.PATH_PLANNED,
                    vehicle_id=vehicle_id,
                    path_points=path if isinstance(path, list) else path.points,
                    details={"task_id": next_task_id}
                ))
    
    def _resolve_conflicts(self) -> None:
        """Resolve conflicts between vehicle paths."""
        if not self.conflict_resolver:
            return
        
        # Collect current paths for each vehicle
        paths = {}
        for vehicle_id, vehicle in self._vehicles.items():
            if hasattr(vehicle, 'current_path') and vehicle.current_path:
                path = vehicle.current_path
                if isinstance(path, Path):
                    paths[vehicle_id] = path
                elif isinstance(path, list):
                    # Convert list of points to Path object
                    from utils.math.trajectories import Path as PathClass
                    paths[vehicle_id] = PathClass(path)
        
        if len(paths) < 2:
            # Not enough vehicles with paths to have conflicts
            return
        
        # Find conflicts
        conflicts = self.conflict_resolver.find_conflicts(paths)
        
        if not conflicts:
            return
        
        # Dispatch conflict events
        for conflict in conflicts:
            self.dispatch_event(ConflictEvent(
                event_type=EventType.CONFLICT_DETECTED,
                vehicle_ids=conflict.vehicle_ids,
                conflict_location=conflict.location if hasattr(conflict, 'location') else None,
                details={"time": conflict.time if hasattr(conflict, 'time') else None}
            ))
        
        # Resolve conflicts
        resolved_paths = self.conflict_resolver.resolve_conflicts(paths)
        
        # Update vehicle paths
        for vehicle_id, new_path in resolved_paths.items():
            if vehicle_id in self._vehicles and new_path != paths.get(vehicle_id):
                vehicle = self._vehicles[vehicle_id]
                
                if hasattr(vehicle, 'assign_path') and callable(vehicle.assign_path):
                    vehicle.assign_path(new_path)
                    
                    # Dispatch event
                    self.dispatch_event(PathEvent(
                        event_type=EventType.PATH_REPLANNED,
                        vehicle_id=vehicle_id,
                        path_points=new_path.points if isinstance(new_path, Path) else new_path,
                        details={"reason": "conflict_resolution"}
                    ))
    
    def start(self) -> None:
        """
        Start the dispatcher.
        
        Begins continuous dispatching according to dispatch interval.
        """
        if self._status == DispatchStatus.RUNNING:
            return
        
        # Set status
        self._status = DispatchStatus.RUNNING
        
        # Clear stop event
        self._stop_event.clear()
        
        # Start dispatch thread
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop)
        self._dispatch_thread.daemon = True
        self._dispatch_thread.start()
        
        self.logger.info("Dispatcher started")
    
    def stop(self) -> None:
        """
        Stop the dispatcher.
        
        Stops continuous dispatching.
        """
        if self._status == DispatchStatus.IDLE:
            return
        
        # Set status
        self._status = DispatchStatus.IDLE
        
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._dispatch_thread and self._dispatch_thread.is_alive():
            self._dispatch_thread.join(timeout=2.0)
        
        self._dispatch_thread = None
        
        self.logger.info("Dispatcher stopped")
    
    def pause(self) -> None:
        """
        Pause the dispatcher.
        
        Temporarily pauses continuous dispatching.
        """
        if self._status != DispatchStatus.RUNNING:
            return
        
        # Set status
        self._status = DispatchStatus.PAUSED
        
        self.logger.info("Dispatcher paused")
    
    def resume(self) -> None:
        """
        Resume the dispatcher.
        
        Resumes continuous dispatching after a pause.
        """
        if self._status != DispatchStatus.PAUSED:
            return
        
        # Set status
        self._status = DispatchStatus.RUNNING
        
        self.logger.info("Dispatcher resumed")
    
    def _dispatch_loop(self) -> None:
        """Main dispatch loop for continuous operation."""
        while not self._stop_event.is_set():
            try:
                # Execute dispatch cycle
                if self._status == DispatchStatus.RUNNING:
                    self.dispatch_cycle()
                
                # Sleep until next cycle
                self._stop_event.wait(self.config.dispatch_interval)
                
            except Exception as e:
                self.logger.error(f"Error in dispatch loop: {str(e)}", exc_info=True)
                
                # Dispatch error event
                self.dispatch_event(SystemEvent(
                    event_type=EventType.SYSTEM_ERROR,
                    message=f"Error in dispatch loop: {str(e)}"
                ))
                
                # Set error status
                self._status = DispatchStatus.ERROR
                
                # Sleep before retrying
                time.sleep(1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dispatcher state to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        result = super().to_dict()
        
        # Add mining dispatcher specific state
        result.update({
            "vehicle_count": len(self._vehicles),
            "task_count": len(self._tasks),
            "pending_task_count": len(self._pending_tasks),
            "assignments": {v_id: tasks.copy() for v_id, tasks in self._assignments.items()}
        })
        
        return result