import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

"""
Priority-based task allocation for the mining dispatch system.

Implements a priority-based allocation strategy where tasks are assigned
to vehicles based on their priorities and costs, with various heuristics
to improve solution quality.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import time
import heapq
from collections import defaultdict

from utils.logger import get_logger, timed

from coordination.allocation.base import (
    TaskAllocator, AllocationConfig, AllocationResult, 
    AllocationStatus, AllocationStrategy, AllocationObjective
)

# Get logger
logger = get_logger("allocation.priority")


class PriorityAllocator(TaskAllocator):
    """
    Priority-based task allocator.
    
    Assigns tasks to vehicles based on task priorities and vehicle suitability,
    using a greedy approach to minimize total cost.
    """
    
    def __init__(self, config: Optional[AllocationConfig] = None):
        """
        Initialize the priority allocator.
        
        Args:
            config: Allocator configuration
        """
        super().__init__(config)
        
        # Update strategy in config if needed
        if self.config.strategy != AllocationStrategy.PRIORITY:
            self.config.strategy = AllocationStrategy.PRIORITY
            logger.info("Updated allocation strategy to PRIORITY")
    
    @timed("priority_allocation")
    def allocate(self, tasks: Dict[str, Any], vehicles: Dict[str, Any], 
                cost_matrix: Optional[Dict[Tuple[str, str], float]] = None) -> AllocationResult:
        """
        Allocate tasks to vehicles based on priority.
        
        Args:
            tasks: Dict of task IDs to task objects
            vehicles: Dict of vehicle IDs to vehicle objects
            cost_matrix: Optional matrix of assignment costs
            
        Returns:
            AllocationResult: Allocation result
        """
        start_time = time.time()
        
        # Validate inputs
        if not tasks:
            return AllocationResult(
                status=AllocationStatus.NO_TASKS,
                computation_time=time.time() - start_time
            )
        
        if not vehicles:
            return AllocationResult(
                status=AllocationStatus.NO_VEHICLES,
                computation_time=time.time() - start_time
            )
        
        # Create default cost matrix if not provided
        if cost_matrix is None:
            cost_matrix = self._create_default_cost_matrix(tasks, vehicles)
        
        # Initialize allocation
        allocations = {vehicle_id: [] for vehicle_id in vehicles}
        unallocated_tasks = []
        
        # Sort tasks by priority (assuming tasks have a 'priority' attribute)
        sorted_tasks = sorted(
            tasks.items(), 
            key=lambda x: getattr(x[1], 'priority', 0),
            reverse=True  # Higher priority first
        )
        
        # Track vehicle loads
        vehicle_loads = {vehicle_id: 0.0 for vehicle_id in vehicles}
        
        # Allocate tasks
        for task_id, task in sorted_tasks:
            # Find best vehicle for this task
            best_vehicle = None
            best_cost = float('inf')
            
            for vehicle_id, vehicle in vehicles.items():
                # Skip if vehicle has reached maximum tasks
                if len(allocations[vehicle_id]) >= self.config.max_tasks_per_vehicle:
                    continue
                
                # Skip if vehicle is unsuitable for task
                if self.config.use_vehicle_suitability and not self._is_vehicle_suitable(vehicle, task):
                    continue
                
                # Get assignment cost
                cost = cost_matrix.get((vehicle_id, task_id), float('inf'))
                
                # Consider current vehicle load for workload balancing
                if self.config.objective == AllocationObjective.BALANCE_WORKLOAD:
                    # Adjust cost based on current load
                    adjusted_cost = cost * (1.0 + vehicle_loads[vehicle_id] / 100.0)
                    if adjusted_cost < best_cost:
                        best_cost = adjusted_cost
                        best_vehicle = vehicle_id
                else:
                    # Simple cost comparison
                    if cost < best_cost:
                        best_cost = cost
                        best_vehicle = vehicle_id
            
            # Assign task if a suitable vehicle was found
            if best_vehicle is not None:
                allocations[best_vehicle].append(task_id)
                vehicle_loads[best_vehicle] += best_cost
            else:
                unallocated_tasks.append(task_id)
        
        # Calculate objective value
        objective_value = self.calculate_objective_value(allocations, tasks, vehicles, cost_matrix)
        
        # Determine status
        if not unallocated_tasks:
            status = AllocationStatus.SUCCESS
        elif len(unallocated_tasks) < len(tasks):
            status = AllocationStatus.PARTIAL
        else:
            status = AllocationStatus.FAILED
        
        # Create result
        result = AllocationResult(
            status=status,
            allocations=allocations,
            unallocated_tasks=unallocated_tasks,
            objective_value=objective_value,
            computation_time=time.time() - start_time,
            metadata={
                "vehicle_loads": vehicle_loads
            }
        )
        
        return result
    
    @timed("priority_reallocation")
    def reallocate(self, current_allocations: Dict[str, List[str]],
                  new_tasks: Dict[str, Any], available_vehicles: Dict[str, Any], 
                  cost_matrix: Optional[Dict[Tuple[str, str], float]] = None) -> AllocationResult:
        """
        Reallocate tasks when conditions change.
        
        This implementation keeps existing allocations and only assigns new tasks
        to available vehicles.
        
        Args:
            current_allocations: Current mapping from vehicle ID to task IDs
            new_tasks: Dict of new task IDs to task objects
            available_vehicles: Dict of available vehicle IDs to vehicle objects
            cost_matrix: Optional matrix of assignment costs
            
        Returns:
            AllocationResult: Updated allocation result
        """
        start_time = time.time()
        
        # Create a copy of current allocations
        updated_allocations = {vehicle_id: list(tasks) for vehicle_id, tasks in current_allocations.items()}
        
        # Initialize for new vehicles
        for vehicle_id in available_vehicles:
            if vehicle_id not in updated_allocations:
                updated_allocations[vehicle_id] = []
        
        # Calculate current vehicle loads
        vehicle_loads = {}
        for vehicle_id, task_ids in updated_allocations.items():
            if vehicle_id in available_vehicles:
                load = 0.0
                for task_id in task_ids:
                    if cost_matrix and (vehicle_id, task_id) in cost_matrix:
                        load += cost_matrix[(vehicle_id, task_id)]
                vehicle_loads[vehicle_id] = load
        
        # Sort new tasks by priority
        sorted_tasks = sorted(
            new_tasks.items(), 
            key=lambda x: getattr(x[1], 'priority', 0),
            reverse=True  # Higher priority first
        )
        
        # Allocate new tasks
        unallocated_tasks = []
        for task_id, task in sorted_tasks:
            # Find best vehicle for this task
            best_vehicle = None
            best_cost = float('inf')
            
            for vehicle_id in available_vehicles:
                # Skip if vehicle has reached maximum tasks
                if len(updated_allocations[vehicle_id]) >= self.config.max_tasks_per_vehicle:
                    continue
                
                # Skip if vehicle is unsuitable for task
                if self.config.use_vehicle_suitability and not self._is_vehicle_suitable(available_vehicles[vehicle_id], task):
                    continue
                
                # Get assignment cost
                cost = cost_matrix.get((vehicle_id, task_id), float('inf')) if cost_matrix else float('inf')
                
                # Consider current vehicle load for workload balancing
                if self.config.objective == AllocationObjective.BALANCE_WORKLOAD:
                    # Adjust cost based on current load
                    adjusted_cost = cost * (1.0 + vehicle_loads.get(vehicle_id, 0.0) / 100.0)
                    if adjusted_cost < best_cost:
                        best_cost = adjusted_cost
                        best_vehicle = vehicle_id
                else:
                    # Simple cost comparison
                    if cost < best_cost:
                        best_cost = cost
                        best_vehicle = vehicle_id
            
            # Assign task if a suitable vehicle was found
            if best_vehicle is not None:
                updated_allocations[best_vehicle].append(task_id)
                vehicle_loads[best_vehicle] = vehicle_loads.get(best_vehicle, 0.0) + best_cost
            else:
                unallocated_tasks.append(task_id)
        
        # Calculate objective value
        all_tasks = {**new_tasks}  # We don't have the original tasks here
        all_vehicles = {**available_vehicles}  # We don't have all vehicles here
        
        # Create a partial cost matrix for the calculation
        partial_cost_matrix = {}
        if cost_matrix:
            for (vehicle_id, task_id), cost in cost_matrix.items():
                if vehicle_id in updated_allocations and task_id in new_tasks:
                    partial_cost_matrix[(vehicle_id, task_id)] = cost
        
        objective_value = self.calculate_objective_value(
            updated_allocations, all_tasks, all_vehicles, partial_cost_matrix
        )
        
        # Determine status
        if not unallocated_tasks:
            status = AllocationStatus.SUCCESS
        elif len(unallocated_tasks) < len(new_tasks):
            status = AllocationStatus.PARTIAL
        else:
            status = AllocationStatus.FAILED
        
        # Create result
        result = AllocationResult(
            status=status,
            allocations=updated_allocations,
            unallocated_tasks=unallocated_tasks,
            objective_value=objective_value,
            computation_time=time.time() - start_time,
            metadata={
                "vehicle_loads": vehicle_loads
            }
        )
        
        return result
    
    def _create_default_cost_matrix(self, tasks: Dict[str, Any], 
                                  vehicles: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
        """
        Create a default cost matrix based on distances.
        
        Args:
            tasks: Dict of task IDs to task objects
            vehicles: Dict of vehicle IDs to vehicle objects
            
        Returns:
            Dict[Tuple[str, str], float]: Cost matrix
        """
        cost_matrix = {}
        
        for vehicle_id, vehicle in vehicles.items():
            for task_id, task in tasks.items():
                # Assuming tasks have a 'location' attribute and vehicles have a 'position' attribute
                if hasattr(task, 'location') and hasattr(vehicle, 'position'):
                    task_location = getattr(task, 'location')
                    vehicle_position = getattr(vehicle, 'position')
                    
                    # Calculate Euclidean distance as cost
                    if hasattr(task_location, 'distance_to'):
                        cost = task_location.distance_to(vehicle_position)
                    else:
                        # Fallback if distance_to method not available
                        dx = getattr(task_location, 'x', 0) - getattr(vehicle_position, 'x', 0)
                        dy = getattr(task_location, 'y', 0) - getattr(vehicle_position, 'y', 0)
                        cost = (dx**2 + dy**2) ** 0.5
                else:
                    # Default cost if location info not available
                    cost = 1.0
                
                cost_matrix[(vehicle_id, task_id)] = cost
        
        return cost_matrix
    
    def _is_vehicle_suitable(self, vehicle: Any, task: Any) -> bool:
        """
        Check if a vehicle is suitable for a task.
        
        Args:
            vehicle: Vehicle object
            task: Task object
            
        Returns:
            bool: True if vehicle is suitable for task
        """
        # Implement task-specific suitability rules here
        # For example:
        
        # Check vehicle type compatibility
        if hasattr(task, 'required_vehicle_type') and hasattr(vehicle, 'vehicle_type'):
            if task.required_vehicle_type != vehicle.vehicle_type:
                return False
        
        # Check vehicle capacity
        if hasattr(task, 'load_size') and hasattr(vehicle, 'capacity'):
            if task.load_size > vehicle.capacity:
                return False
        
        # Check vehicle equipment compatibility
        if hasattr(task, 'required_equipment') and hasattr(vehicle, 'equipment'):
            if not all(eq in vehicle.equipment for eq in task.required_equipment):
                return False
        
        # Check fuel level if configured
        if self.config.consider_fuel_level and hasattr(vehicle, 'fuel_level'):
            # Assuming tasks have an 'estimated_fuel_consumption' attribute
            if (hasattr(task, 'estimated_fuel_consumption') and 
                vehicle.fuel_level < task.estimated_fuel_consumption):
                return False
        
        # Check maintenance status if configured
        if self.config.consider_maintenance and hasattr(vehicle, 'maintenance_due'):
            if vehicle.maintenance_due:
                return False
        
        return True