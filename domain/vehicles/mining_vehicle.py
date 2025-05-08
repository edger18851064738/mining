"""
Mining vehicle implementation for the mining dispatch system.

Provides specialized vehicle types for mining operations:
- Dump trucks for ore transport
- Excavators for loading
- Support vehicles
"""

from typing import Dict, List, Tuple, Optional, Any, Set, Union
from datetime import datetime
import math
import uuid

from utils.geo.coordinates import Point2D
from utils.math.vectors import Vector2D
from utils.logger import get_logger
from utils.io.serialization import Serializable

from domain.vehicles.base import ConstrainedVehicle
from domain.vehicles.vehicle_state import (
    VehicleState, TransportStage, VehicleStateManager, VehicleStateError
)

# Initialize logger
logger = get_logger("mining_vehicle")


class MiningVehicleError(Exception):
    """Base exception for mining vehicle errors."""
    pass


class TaskAssignmentError(MiningVehicleError):
    """Exception raised when a task cannot be assigned to a vehicle."""
    pass


class MiningVehicle(ConstrainedVehicle):
    """
    Mining vehicle specialized for mining operations.
    
    Extends ConstrainedVehicle with mining-specific attributes like:
    - Cargo capacity
    - Terrain capabilities
    - Specialized equipment
    """
    
    def __init__(self, vehicle_id: Optional[str] = None, 
                 max_speed: float = 5.0,
                 max_capacity: float = 50000.0,
                 terrain_capability: float = 0.7,
                 turning_radius: float = 10.0,
                 length: float = 5.0,
                 width: float = 2.0,
                 vehicle_type: str = "standard",
                 environment = None):
        """
        Initialize a mining vehicle.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            max_speed: Maximum speed in m/s
            max_capacity: Maximum load capacity in kg
            terrain_capability: Terrain traversal capability (0-1)
            turning_radius: Minimum turning radius in meters
            length: Vehicle length in meters
            width: Vehicle width in meters
            vehicle_type: Type designation (standard, heavy, etc.)
            environment: Reference to the operating environment
        """
        super().__init__(
            vehicle_id=vehicle_id,
            max_speed=max_speed,
            turning_radius=turning_radius,
            length=length,
            width=width
        )
        
        # Mining-specific attributes
        self.max_capacity = max_capacity
        self.current_load = 0.0
        self.terrain_capability = terrain_capability
        self.vehicle_type = vehicle_type
        self.environment = environment
        
        # Operational attributes
        self.base_location = Point2D(0, 0)
        self.state_manager = VehicleStateManager(VehicleState.IDLE)
        self.current_task = None
        self.task_history = []
        
        # Maintenance and fault tracking
        self.total_distance_traveled = 0.0
        self.operation_hours = 0.0
        self.maintenance_due = False
        self.fault_codes = set()
        self.last_maintenance_time = datetime.now()
        
        # Performance metrics
        self.metrics = {
            'tasks_completed': 0,
            'total_load_transported': 0.0,
            'waiting_time': 0.0,
            'travel_time': 0.0,
            'fuel_consumption': 0.0
        }
    
    @property
    def state(self) -> VehicleState:
        """Get the current state of the vehicle."""
        return self.state_manager.current_state
    
    @state.setter
    def state(self, new_state: VehicleState) -> None:
        """
        Set the vehicle state.
        
        Args:
            new_state: New vehicle state
            
        Raises:
            VehicleStateError: If state transition is invalid
        """
        try:
            self.state_manager.transition_to(new_state)
            logger.debug(f"Vehicle {self.vehicle_id} state changed to {new_state.name}")
        except VehicleStateError as e:
            logger.warning(f"Invalid state transition for vehicle {self.vehicle_id}: {str(e)}")
            raise
    
    @property
    def transport_stage(self) -> TransportStage:
        """Get the current transport stage."""
        return self.state_manager.transport_stage
    
    @transport_stage.setter
    def transport_stage(self, stage: TransportStage) -> None:
        """
        Set the transport stage.
        
        Args:
            stage: New transport stage
        """
        self.state_manager.transport_stage = stage
        logger.debug(f"Vehicle {self.vehicle_id} transport stage changed to {stage.name}")
    
    @property
    def load_ratio(self) -> float:
        """Get the current load ratio (0-1)."""
        if self.max_capacity <= 0:
            return 0.0
        return min(1.0, self.current_load / self.max_capacity)
    
    @property
    def is_loaded(self) -> bool:
        """Check if vehicle is carrying a load."""
        return self.current_load > 0.0
    
    @property
    def is_full(self) -> bool:
        """Check if vehicle is at full capacity."""
        return self.current_load >= self.max_capacity * 0.99  # Allow for small rounding errors
    
    @property
    def is_empty(self) -> bool:
        """Check if vehicle is empty."""
        return self.current_load <= 0.001  # Allow for small rounding errors
    
    @property
    def is_available(self) -> bool:
        """Check if vehicle is available for new tasks."""
        return (self.state == VehicleState.IDLE and 
                self.current_task is None and
                not self.fault_codes and
                not self.maintenance_due)
    
    @property
    def is_operational(self) -> bool:
        """Check if vehicle is operational (not faulted or out of service)."""
        return (self.state != VehicleState.OUT_OF_SERVICE and
                self.state != VehicleState.FAULT and
                self.state != VehicleState.MAINTENANCE)
    
    def update_position(self, dt: float = 1.0) -> None:
        """
        Update the vehicle's position based on its path and speed.
        
        Args:
            dt: Time step in seconds
        """
        if self.state != VehicleState.EN_ROUTE or not self.current_path:
            return
        
        # Get current and next waypoint
        if self.is_at_path_end:
            # Reached end of path, update state
            self._handle_path_completion()
            return
        
        current_pos = self.current_location
        next_wp = self.next_waypoint
        
        if next_wp is None:
            return
        
        # Calculate direction and distance to next waypoint
        direction = Vector2D(next_wp.x - current_pos.x, next_wp.y - current_pos.y)
        distance = direction.magnitude
        
        # Update speed based on target speed
        self.update_speed(dt)
        
        # Calculate movement distance for this time step
        movement_distance = self.current_speed * dt
        
        # Check if we reach or pass the next waypoint
        if movement_distance >= distance:
            # Move to the next waypoint
            self.move_to_next_waypoint()
            
            # Update heading
            if self.next_waypoint:
                next_direction = Vector2D(
                    self.next_waypoint.x - self.current_location.x,
                    self.next_waypoint.y - self.current_location.y
                )
                if next_direction.magnitude > 0:
                    self.heading = math.atan2(next_direction.y, next_direction.x)
            
            # Use remaining movement distance for next leg
            remaining_distance = movement_distance - distance
            if remaining_distance > 0 and not self.is_at_path_end:
                # Recursive call with remaining distance
                # Convert remaining distance back to time
                if self.current_speed > 0:
                    self.update_position(remaining_distance / self.current_speed)
        else:
            # Move along the path
            if direction.magnitude > 0:
                normalized_dir = direction.normalized
                new_x = current_pos.x + normalized_dir.x * movement_distance
                new_y = current_pos.y + normalized_dir.y * movement_distance
                self.current_location = Point2D(new_x, new_y)
                
                # Update heading
                self.heading = math.atan2(normalized_dir.y, normalized_dir.x)
        
        # Update metrics
        self.total_distance_traveled += movement_distance
        
        # Check for maintenance triggers
        self._check_maintenance_triggers()
    
    def _handle_path_completion(self) -> None:
        """Handle completion of the current path."""
        if not self.current_task:
            # No task, just stop
            self.state = VehicleState.IDLE
            return
        
        # Get task type
        task_type = getattr(self.current_task, 'task_type', '')
        
        # Determine what to do based on task type and transport stage
        if task_type == 'loading' and self.transport_stage == TransportStage.APPROACHING:
            # Arrived at loading point
            self.state = VehicleState.LOADING
            # Loading logic will be handled by the task system
        elif task_type == 'unloading' and self.transport_stage == TransportStage.TRANSPORTING:
            # Arrived at unloading point
            self.state = VehicleState.UNLOADING
            # Unloading logic will be handled by the task system
        elif self.transport_stage == TransportStage.RETURNING:
            # Returned to base location
            self.state = VehicleState.IDLE
            self._complete_current_task()
        else:
            # Default behavior
            self.state = VehicleState.IDLE
    
    def assign_task(self, task: Any) -> None:
        """
        Assign a task to the vehicle.
        
        Args:
            task: Task to assign
            
        Raises:
            TaskAssignmentError: If task can't be assigned
        """
        # Check if vehicle is available
        if not self.is_available:
            raise TaskAssignmentError(
                f"Vehicle {self.vehicle_id} is not available for new tasks"
            )
        
        # Check if task is compatible
        task_type = getattr(task, 'task_type', '')
        
        if task_type == 'loading' and self.is_full:
            raise TaskAssignmentError(
                f"Vehicle {self.vehicle_id} is already full, can't assign loading task"
            )
        
        if task_type == 'unloading' and self.is_empty:
            raise TaskAssignmentError(
                f"Vehicle {self.vehicle_id} is empty, can't assign unloading task"
            )
        
        # Assign the task
        self.current_task = task
        
        # Set initial state and transport stage
        self.state = VehicleState.PREPARING
        if task_type == 'loading':
            self.transport_stage = TransportStage.APPROACHING
        elif task_type == 'unloading':
            self.transport_stage = TransportStage.TRANSPORTING
        else:
            self.transport_stage = TransportStage.APPROACHING
        
        # If task has a path, assign it
        if hasattr(task, 'path') and task.path:
            self.assign_path(task.path)
        
        logger.info(f"Vehicle {self.vehicle_id} assigned task {task.task_id}")
    
    def calculate_path_to(self, destination: Point2D) -> List[Point2D]:
        """
        Calculate a path from current location to a destination.
        
        Args:
            destination: Target location
            
        Returns:
            List[Point2D]: Calculated path
        """
        # If environment has a path planner, use it
        if self.environment and hasattr(self.environment, 'find_path'):
            try:
                path = self.environment.find_path(self.current_location, destination, self)
                return path
            except Exception as e:
                logger.warning(f"Path planning failed: {str(e)}")
        
        # Fallback: simple direct path
        return [self.current_location, destination]
    
    def assign_path(self, path: List[Union[Point2D, Tuple[float, float]]]) -> None:
        """
        Assign a new path to the vehicle.
        
        Args:
            path: List of points forming the path
        """
        # Set the path
        super().assign_path(path)
        
        # Update state for movement
        if self.state == VehicleState.PREPARING or self.state == VehicleState.IDLE:
            self.state = VehicleState.EN_ROUTE
        
        # Calculate path distance for metrics
        path_distance = 0.0
        for i in range(len(self.current_path) - 1):
            path_distance += self.current_path[i].distance_to(self.current_path[i+1])
        
        logger.debug(f"Vehicle {self.vehicle_id} assigned path with {len(path)} points, "
                    f"distance: {path_distance:.1f}m")
    
    def load(self, amount: float) -> float:
        """
        Load material onto the vehicle.
        
        Args:
            amount: Amount to load in kg
            
        Returns:
            float: Amount actually loaded
        """
        # Check state
        if self.state != VehicleState.LOADING and self.state != VehicleState.IDLE:
            logger.warning(f"Vehicle {self.vehicle_id} tried to load while in state {self.state.name}")
            return 0.0
        
        # Calculate how much can be loaded
        available_capacity = self.max_capacity - self.current_load
        loadable_amount = min(amount, available_capacity)
        
        # Add to current load
        self.current_load += loadable_amount
        
        logger.info(f"Vehicle {self.vehicle_id} loaded {loadable_amount:.1f}kg, "
                   f"current load: {self.current_load:.1f}kg")
        
        return loadable_amount
    
    def unload(self, amount: Optional[float] = None) -> float:
        """
        Unload material from the vehicle.
        
        Args:
            amount: Amount to unload in kg, or None for full unload
            
        Returns:
            float: Amount actually unloaded
        """
        # Check state
        if self.state != VehicleState.UNLOADING and self.state != VehicleState.IDLE:
            logger.warning(f"Vehicle {self.vehicle_id} tried to unload while in state {self.state.name}")
            return 0.0
        
        # Determine amount to unload
        if amount is None:
            unload_amount = self.current_load
        else:
            unload_amount = min(amount, self.current_load)
        
        # Update current load
        self.current_load -= unload_amount
        
        # Update metrics
        self.metrics['total_load_transported'] += unload_amount
        
        logger.info(f"Vehicle {self.vehicle_id} unloaded {unload_amount:.1f}kg, "
                   f"current load: {self.current_load:.1f}kg")
        
        return unload_amount
    
    def _complete_current_task(self) -> None:
        """Complete the current task and update vehicle state."""
        if not self.current_task:
            return
        
        # Check if task has a completion handler
        if hasattr(self.current_task, 'complete') and callable(self.current_task.complete):
            try:
                self.current_task.complete()
            except Exception as e:
                logger.error(f"Error completing task {self.current_task.task_id}: {str(e)}")
        
        # Add to task history
        self.task_history.append(self.current_task)
        
        # Update metrics
        self.metrics['tasks_completed'] += 1
        
        # Clear current task
        task_id = getattr(self.current_task, 'task_id', 'unknown')
        logger.info(f"Vehicle {self.vehicle_id} completed task {task_id}")
        self.current_task = None
        
        # Reset state
        self.state = VehicleState.IDLE
        self.transport_stage = TransportStage.NONE
    
    def _check_maintenance_triggers(self) -> None:
        """Check if maintenance is required based on distance or hours."""
        # Example maintenance triggers
        MAINTENANCE_DISTANCE = 10000.0  # 10km
        MAINTENANCE_HOURS = 100.0       # 100 operating hours
        
        if (self.total_distance_traveled >= MAINTENANCE_DISTANCE or
            self.operation_hours >= MAINTENANCE_HOURS):
            self.maintenance_due = True
            
            if not self.fault_codes:
                self.fault_codes.add("SCHEDULED_MAINTENANCE")
    
    def perform_maintenance(self) -> None:
        """Perform maintenance and reset maintenance triggers."""
        # Check if already in maintenance
        if self.state != VehicleState.MAINTENANCE:
            if self.current_task:
                logger.warning(f"Vehicle {self.vehicle_id} entering maintenance with active task")
                # Task will be kept but vehicle becomes unavailable
            
            # Change state
            try:
                self.state = VehicleState.MAINTENANCE
            except VehicleStateError:
                # Force state change if needed
                self.state_manager.transition_to(VehicleState.MAINTENANCE, force=True)
        
        # Reset maintenance triggers
        self.maintenance_due = False
        self.fault_codes.discard("SCHEDULED_MAINTENANCE")
        
        # Record maintenance time
        self.last_maintenance_time = datetime.now()
        
        logger.info(f"Vehicle {self.vehicle_id} completed maintenance")
    
    def add_fault(self, fault_code: str, description: str = "") -> None:
        """
        Add a fault code to the vehicle.
        
        Args:
            fault_code: Fault code identifier
            description: Description of the fault
        """
        self.fault_codes.add(fault_code)
        
        # Update state if operational
        if self.is_operational:
            try:
                self.state = VehicleState.FAULT
            except VehicleStateError:
                # Force state change if needed
                self.state_manager.transition_to(VehicleState.FAULT, force=True)
        
        logger.warning(f"Vehicle {self.vehicle_id} reported fault: {fault_code} - {description}")
    
    def clear_fault(self, fault_code: str) -> bool:
        """
        Clear a specific fault code.
        
        Args:
            fault_code: Fault code to clear
            
        Returns:
            bool: True if fault was cleared, False if not found
        """
        if fault_code in self.fault_codes:
            self.fault_codes.remove(fault_code)
            
            # If no more faults and no maintenance due, restore to IDLE state
            if not self.fault_codes and not self.maintenance_due:
                if self.state == VehicleState.FAULT:
                    self.state = VehicleState.IDLE
                    
            logger.info(f"Fault {fault_code} cleared for vehicle {self.vehicle_id}")
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset the vehicle to initial state."""
        # Reset position and path
        self.reset_path()
        
        # Reset load
        self.current_load = 0.0
        
        # Reset state
        self.state_manager.reset(VehicleState.IDLE)
        
        # Clear task
        self.current_task = None
        
        # Clear faults
        self.fault_codes.clear()
        self.maintenance_due = False
        
        logger.info(f"Vehicle {self.vehicle_id} reset to initial state")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert vehicle to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'vehicle_id': self.vehicle_id,
            'vehicle_type': self.vehicle_type,
            'position': {
                'x': self.current_location.x,
                'y': self.current_location.y
            },
            'heading': self.heading,
            'state': self.state.name,
            'transport_stage': self.transport_stage.name,
            'current_load': self.current_load,
            'max_capacity': self.max_capacity,
            'max_speed': self.max_speed,
            'current_speed': self.current_speed,
            'task_id': getattr(self.current_task, 'task_id', None),
            'fault_codes': list(self.fault_codes),
            'maintenance_due': self.maintenance_due,
            'metrics': self.metrics.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MiningVehicle':
        """
        Create a vehicle from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            MiningVehicle: New vehicle instance
        """
        vehicle = cls(
            vehicle_id=data.get('vehicle_id'),
            max_speed=data.get('max_speed', 5.0),
            max_capacity=data.get('max_capacity', 50000.0),
            vehicle_type=data.get('vehicle_type', 'standard')
        )
        
        # Set position and heading
        if 'position' in data:
            pos = data['position']
            vehicle.current_location = Point2D(pos['x'], pos['y'])
        
        vehicle.heading = data.get('heading', 0.0)
        
        # Set load
        vehicle.current_load = data.get('current_load', 0.0)
        
        # Set state
        try:
            state_name = data.get('state', 'IDLE')
            vehicle.state = VehicleState[state_name]
        except (KeyError, ValueError):
            vehicle.state = VehicleState.IDLE
        
        try:
            stage_name = data.get('transport_stage', 'NONE')
            vehicle.transport_stage = TransportStage[stage_name]
        except (KeyError, ValueError):
            vehicle.transport_stage = TransportStage.NONE
        
        # Set speed
        vehicle.current_speed = data.get('current_speed', 0.0)
        
        # Set faults
        for fault in data.get('fault_codes', []):
            vehicle.fault_codes.add(fault)
        
        vehicle.maintenance_due = data.get('maintenance_due', False)
        
        # Set metrics
        if 'metrics' in data:
            vehicle.metrics.update(data['metrics'])
        
        return vehicle


class DumpTruck(MiningVehicle):
    """
    Specialized mining vehicle for transporting ore.
    
    Features:
    - High capacity
    - Medium speed
    - Medium terrain capability
    """
    
    def __init__(self, vehicle_id: Optional[str] = None, environment = None):
        """
        Initialize a dump truck.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            environment: Reference to the operating environment
        """
        super().__init__(
            vehicle_id=vehicle_id,
            max_speed=4.0,             # 4 m/s (~14.4 km/h)
            max_capacity=80000.0,      # 80 metric tons
            terrain_capability=0.6,    # Medium off-road capability
            turning_radius=12.0,       # 12m turning radius
            length=10.0,               # 10m length
            width=5.0,                 # 5m width
            vehicle_type="dump_truck",
            environment=environment
        )
        
        # Truck-specific attributes
        self.bed_raised = False
        self.dump_rate = 10000.0  # kg/s
        self.load_rate = 5000.0   # kg/s
    
    def raise_bed(self) -> bool:
        """
        Raise the truck bed for dumping.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.state != VehicleState.UNLOADING:
            logger.warning(f"Cannot raise bed when not in UNLOADING state")
            return False
            
        self.bed_raised = True
        return True
    
    def lower_bed(self) -> bool:
        """
        Lower the truck bed after dumping.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.bed_raised = False
        return True
    
    def update_position(self, dt: float = 1.0) -> None:
        """Override to adjust speed based on load."""
        # Adjust max speed based on load
        load_factor = 1.0 - (self.load_ratio * 0.3)  # Up to 30% slower when fully loaded
        original_max_speed = self.max_speed
        self.max_speed = original_max_speed * load_factor
        
        # Call parent method
        super().update_position(dt)
        
        # Restore original max speed
        self.max_speed = original_max_speed


class Excavator(MiningVehicle):
    """
    Specialized mining vehicle for excavating and loading.
    
    Features:
    - Limited mobility
    - High digging capability
    - Can load other vehicles
    """
    
    def __init__(self, vehicle_id: Optional[str] = None, environment = None):
        """
        Initialize an excavator.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            environment: Reference to the operating environment
        """
        super().__init__(
            vehicle_id=vehicle_id,
            max_speed=1.5,             # 1.5 m/s (~5.4 km/h)
            max_capacity=5000.0,       # 5 metric tons bucket capacity
            terrain_capability=0.8,    # High off-road capability
            turning_radius=8.0,        # 8m turning radius
            length=12.0,               # 12m length
            width=4.0,                 # 4m width
            vehicle_type="excavator",
            environment=environment
        )
        
        # Excavator-specific attributes
        self.arm_position = 0.0  # 0 = stowed, 1 = fully extended
        self.bucket_capacity = 5000.0  # kg
        self.dig_rate = 3000.0  # kg/s
        self.current_bucket_load = 0.0
    
    def extend_arm(self, position: float) -> None:
        """
        Extend excavator arm.
        
        Args:
            position: Arm position (0-1)
        """
        self.arm_position = max(0.0, min(1.0, position))
    
    def dig(self, amount: float) -> float:
        """
        Dig material into bucket.
        
        Args:
            amount: Amount to dig in kg
            
        Returns:
            float: Amount actually dug
        """
        if self.arm_position < 0.5:
            logger.warning("Arm not sufficiently extended for digging")
            return 0.0
            
        available_capacity = self.bucket_capacity - self.current_bucket_load
        dig_amount = min(amount, available_capacity)
        
        self.current_bucket_load += dig_amount
        return dig_amount
    
    def load_vehicle(self, vehicle: MiningVehicle) -> float:
        """
        Load material from bucket to another vehicle.
        
        Args:
            vehicle: Vehicle to load
            
        Returns:
            float: Amount loaded
        """
        if self.current_bucket_load <= 0:
            return 0.0
            
        # Check if vehicles are close enough
        if self.distance_to(vehicle.current_location) > 15.0:
            logger.warning("Vehicle too far to load")
            return 0.0
            
        # Attempt to load vehicle
        loaded_amount = vehicle.load(self.current_bucket_load)
        self.current_bucket_load -= loaded_amount
        
        return loaded_amount


class SupportVehicle(MiningVehicle):
    """
    Specialized mining vehicle for maintenance and support operations.
    
    Features:
    - High mobility
    - Specialized equipment for repairs
    - Fuel and parts transport
    """
    
    def __init__(self, vehicle_id: Optional[str] = None, environment = None):
        """
        Initialize a support vehicle.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            environment: Reference to the operating environment
        """
        super().__init__(
            vehicle_id=vehicle_id,
            max_speed=8.0,             # 8 m/s (~28.8 km/h)
            max_capacity=2000.0,       # 2 metric tons cargo capacity
            terrain_capability=0.9,    # Very high off-road capability
            turning_radius=5.0,        # 5m turning radius
            length=6.0,                # 6m length
            width=2.5,                 # 2.5m width
            vehicle_type="support",
            environment=environment
        )
        
        # Support vehicle-specific attributes
        self.repair_equipment = True
        self.fuel_capacity = 1000.0  # Liters
        self.current_fuel = 1000.0
        self.spare_parts = {}  # Dictionary of spare parts
    
    def repair_vehicle(self, vehicle: MiningVehicle) -> List[str]:
        """
        Perform repairs on another vehicle.
        
        Args:
            vehicle: Vehicle to repair
            
        Returns:
            List[str]: List of faults fixed
        """
        if not self.repair_equipment:
            logger.warning("No repair equipment available")
            return []
            
        # Check if vehicles are close enough
        if self.distance_to(vehicle.current_location) > 10.0:
            logger.warning("Vehicle too far to repair")
            return []
        
        # Fix faults
        fixed_faults = list(vehicle.fault_codes)
        for fault in fixed_faults:
            vehicle.clear_fault(fault)
        
        # Reset maintenance due flag
        vehicle.maintenance_due = False
        
        logger.info(f"Support vehicle {self.vehicle_id} fixed {len(fixed_faults)} "
                   f"faults on vehicle {vehicle.vehicle_id}")
        
        return fixed_faults
    
    def refuel_vehicle(self, vehicle: MiningVehicle, amount: float) -> float:
        """
        Refuel another vehicle.
        
        Args:
            vehicle: Vehicle to refuel
            amount: Amount of fuel to transfer
            
        Returns:
            float: Amount actually transferred
        """
        if not hasattr(vehicle, 'fuel_capacity') or not hasattr(vehicle, 'current_fuel'):
            logger.warning("Vehicle doesn't support refueling")
            return 0.0
        
        # Check if vehicles are close enough
        if self.distance_to(vehicle.current_location) > 10.0:
            logger.warning("Vehicle too far to refuel")
            return 0.0
        
        # Calculate refueling amount
        available_fuel = self.current_fuel
        vehicle_capacity = vehicle.fuel_capacity
        vehicle_current = vehicle.current_fuel
        needed = vehicle_capacity - vehicle_current
        
        transfer_amount = min(amount, available_fuel, needed)
        
        # Transfer fuel
        self.current_fuel -= transfer_amount
        vehicle.current_fuel += transfer_amount
        
        logger.info(f"Support vehicle {self.vehicle_id} transferred {transfer_amount:.1f}L "
                   f"of fuel to vehicle {vehicle.vehicle_id}")
        
        return transfer_amount