"""
Base interfaces and classes for task allocation in the mining dispatch system.

This module provides the core abstractions for task allocation algorithms,
including interfaces and common data structures used across different
allocation strategies.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import abc
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Any, Optional, TypeVar, Generic
from dataclasses import dataclass, field
import time

from utils.logger import get_logger

# Get logger
logger = get_logger("coordination.allocation")


class AllocationStrategy(Enum):
    """Strategies for task allocation."""
    PRIORITY = auto()     # Priority-based allocation
    GREEDY = auto()       # Greedy allocation (minimal cost/distance)
    MIQP = auto()         # Mixed Integer Quadratic Programming
    AUCTION = auto()      # Auction-based allocation
    HUNGARIAN = auto()    # Hungarian algorithm (optimal assignment)


class AllocationStatus(Enum):
    """Status codes for allocation results."""
    SUCCESS = auto()           # Allocation completed successfully
    PARTIAL = auto()           # Only some tasks were allocated
    FAILED = auto()            # Allocation failed
    TIMEOUT = auto()           # Allocation timed out
    INFEASIBLE = auto()        # No feasible allocation exists
    NO_VEHICLES = auto()       # No vehicles available
    NO_TASKS = auto()          # No tasks to allocate


class AllocationObjective(Enum):
    """Optimization objectives for allocation."""
    MINIMIZE_MAKESPAN = auto()       # Minimize the total completion time
    MINIMIZE_DISTANCE = auto()       # Minimize total travel distance
    MINIMIZE_ENERGY = auto()         # Minimize energy consumption
    BALANCE_WORKLOAD = auto()        # Balance workload among vehicles
    MAXIMIZE_THROUGHPUT = auto()     # Maximize material throughput
    MULTI_OBJECTIVE = auto()         # Multiple weighted objectives


@dataclass
class AllocationConfig:
    """Configuration for task allocation."""
    strategy: AllocationStrategy = AllocationStrategy.PRIORITY  # Allocation strategy
    objective: AllocationObjective = AllocationObjective.MINIMIZE_MAKESPAN  # Optimization objective
    
    # Timeout settings
    timeout: float = 5.0  # Maximum time for allocation (seconds)
    
    # Constraints
    max_tasks_per_vehicle: int = 5  # Maximum tasks assigned to a vehicle
    consider_fuel_level: bool = True  # Consider vehicle fuel level
    consider_maintenance: bool = True  # Consider vehicle maintenance needs
    
    # Priority settings
    use_task_priority: bool = True  # Consider task priorities
    use_vehicle_suitability: bool = True  # Consider vehicle-task suitability
    
    # Algorithm-specific parameters
    miqp_gap: float = 0.05  # Optimality gap for MIQP solver
    miqp_presolve: bool = True  # Use presolve in MIQP
    
    # Multi-objective weights (if using MULTI_OBJECTIVE)
    makespan_weight: float = 1.0
    distance_weight: float = 0.5
    energy_weight: float = 0.2
    balance_weight: float = 0.3
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocationResult:
    """Result of a task allocation operation."""
    status: AllocationStatus = AllocationStatus.FAILED  # Allocation status
    allocations: Dict[str, List[str]] = field(default_factory=dict)  # Mapping from vehicle ID to task IDs
    unallocated_tasks: List[str] = field(default_factory=list)  # Tasks that couldn't be allocated
    objective_value: float = float('inf')  # Value of the objective function
    computation_time: float = 0.0  # Computation time in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class TaskAllocator(abc.ABC):
    """
    Base interface for task allocators.
    
    Task allocators are responsible for assigning tasks to vehicles
    according to various optimization criteria and constraints.
    """
    
    def __init__(self, config: Optional[AllocationConfig] = None):
        """
        Initialize the task allocator.
        
        Args:
            config: Allocator configuration
        """
        self.config = config or AllocationConfig()
        self.logger = get_logger(f"allocation.{self.__class__.__name__}")
    
    @abc.abstractmethod
    def allocate(self, tasks: Dict[str, Any], vehicles: Dict[str, Any], 
                cost_matrix: Optional[Dict[Tuple[str, str], float]] = None) -> AllocationResult:
        """
        Allocate tasks to vehicles.
        
        Args:
            tasks: Dict of task IDs to task objects
            vehicles: Dict of vehicle IDs to vehicle objects
            cost_matrix: Optional matrix of assignment costs.
                         Keys are (vehicle_id, task_id) tuples, values are costs.
            
        Returns:
            AllocationResult: Allocation result
        """
        pass
    
    @abc.abstractmethod
    def reallocate(self, current_allocations: Dict[str, List[str]],
                  new_tasks: Dict[str, Any], available_vehicles: Dict[str, Any], 
                  cost_matrix: Optional[Dict[Tuple[str, str], float]] = None) -> AllocationResult:
        """
        Reallocate tasks when conditions change.
        
        Args:
            current_allocations: Current mapping from vehicle ID to task IDs
            new_tasks: Dict of new task IDs to task objects
            available_vehicles: Dict of available vehicle IDs to vehicle objects
            cost_matrix: Optional matrix of assignment costs
            
        Returns:
            AllocationResult: Updated allocation result
        """
        pass
    
    def get_strategy(self) -> AllocationStrategy:
        """
        Get the allocator's strategy.
        
        Returns:
            AllocationStrategy: Allocation strategy
        """
        return self.config.strategy
    
    def set_config(self, config: AllocationConfig) -> None:
        """
        Update the allocator configuration.
        
        Args:
            config: New configuration
        """
        self.config = config
    
    def validate_allocation(self, allocation: Dict[str, List[str]], 
                           tasks: Dict[str, Any], vehicles: Dict[str, Any]) -> bool:
        """
        Validate that an allocation meets all constraints.
        
        Args:
            allocation: Mapping from vehicle ID to task IDs
            tasks: Dict of task IDs to task objects
            vehicles: Dict of vehicle IDs to vehicle objects
            
        Returns:
            bool: True if allocation is valid
        """
        # Check if all tasks are allocated
        allocated_tasks = set()
        for task_list in allocation.values():
            allocated_tasks.update(task_list)
        
        if not all(task_id in allocated_tasks for task_id in tasks):
            self.logger.warning("Not all tasks are allocated")
            return False
        
        # Check maximum tasks per vehicle constraint
        for vehicle_id, task_list in allocation.items():
            if len(task_list) > self.config.max_tasks_per_vehicle:
                self.logger.warning(f"Vehicle {vehicle_id} has too many tasks ({len(task_list)})")
                return False
        
        # Check if all allocated vehicles exist
        if not all(vehicle_id in vehicles for vehicle_id in allocation):
            self.logger.warning("Allocation references non-existent vehicles")
            return False
        
        # Check if all allocated tasks exist
        for task_list in allocation.values():
            if not all(task_id in tasks for task_id in task_list):
                self.logger.warning("Allocation references non-existent tasks")
                return False
        
        return True
    
    def calculate_objective_value(self, allocation: Dict[str, List[str]],
                                tasks: Dict[str, Any], vehicles: Dict[str, Any],
                                cost_matrix: Dict[Tuple[str, str], float]) -> float:
        """
        Calculate the objective value for an allocation.
        
        Args:
            allocation: Mapping from vehicle ID to task IDs
            tasks: Dict of task IDs to task objects
            vehicles: Dict of vehicle IDs to vehicle objects
            cost_matrix: Matrix of assignment costs
            
        Returns:
            float: Objective value (lower is better)
        """
        objective = 0.0
        
        # Different calculations based on objective
        if self.config.objective == AllocationObjective.MINIMIZE_MAKESPAN:
            # Makespan is the maximum completion time across all vehicles
            vehicle_times = {}
            
            for vehicle_id, task_ids in allocation.items():
                total_time = 0.0
                for task_id in task_ids:
                    if (vehicle_id, task_id) in cost_matrix:
                        total_time += cost_matrix[(vehicle_id, task_id)]
                vehicle_times[vehicle_id] = total_time
            
            objective = max(vehicle_times.values()) if vehicle_times else 0.0
            
        elif self.config.objective == AllocationObjective.MINIMIZE_DISTANCE:
            # Sum of all travel distances
            total_distance = 0.0
            
            for vehicle_id, task_ids in allocation.items():
                for task_id in task_ids:
                    if (vehicle_id, task_id) in cost_matrix:
                        total_distance += cost_matrix[(vehicle_id, task_id)]
            
            objective = total_distance
            
        elif self.config.objective == AllocationObjective.MULTI_OBJECTIVE:
            # Weighted sum of multiple objectives
            makespan = 0.0
            total_distance = 0.0
            
            # Calculate makespan
            vehicle_times = {}
            for vehicle_id, task_ids in allocation.items():
                total_time = 0.0
                for task_id in task_ids:
                    if (vehicle_id, task_id) in cost_matrix:
                        total_time += cost_matrix[(vehicle_id, task_id)]
                vehicle_times[vehicle_id] = total_time
            
            makespan = max(vehicle_times.values()) if vehicle_times else 0.0
            
            # Calculate total distance
            for vehicle_id, task_ids in allocation.items():
                for task_id in task_ids:
                    if (vehicle_id, task_id) in cost_matrix:
                        total_distance += cost_matrix[(vehicle_id, task_id)]
            
            # Calculate workload balance
            workload_std = 0.0
            if vehicle_times:
                avg_workload = sum(vehicle_times.values()) / len(vehicle_times)
                workload_std = (sum((t - avg_workload) ** 2 for t in vehicle_times.values()) / len(vehicle_times)) ** 0.5
            
            # Combine objectives with weights
            objective = (
                self.config.makespan_weight * makespan +
                self.config.distance_weight * total_distance +
                self.config.balance_weight * workload_std
            )
        
        return objective