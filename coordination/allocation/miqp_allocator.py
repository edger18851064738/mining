import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

"""
Mixed Integer Quadratic Programming (MIQP) task allocator for the mining dispatch system.

Implements an optimal task allocation strategy using mixed integer quadratic programming
to minimize objectives such as makespan, total distance, or a weighted combination.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import time
import numpy as np

from utils.logger import get_logger, timed

from coordination.allocation.base import (
    TaskAllocator, AllocationConfig, AllocationResult, 
    AllocationStatus, AllocationStrategy, AllocationObjective
)

# Get logger
logger = get_logger("allocation.miqp")


class MIQPAllocator(TaskAllocator):
    """
    Mixed Integer Quadratic Programming (MIQP) based task allocator.
    
    Finds optimal task allocations by formulating the problem as a mixed
    integer quadratic program and solving it with an optimization solver.
    """
    
    def __init__(self, config: Optional[AllocationConfig] = None):
        """
        Initialize the MIQP allocator.
        
        Args:
            config: Allocator configuration
        """
        super().__init__(config)
        
        # Update strategy in config if needed
        if self.config.strategy != AllocationStrategy.MIQP:
            self.config.strategy = AllocationStrategy.MIQP
            logger.info("Updated allocation strategy to MIQP")
        
        # Try to import optimization libraries
        self.solver_available = False
        try:
            import pulp
            self.solver = "pulp"
            self.solver_available = True
            logger.info("Using PuLP as the optimization solver")
        except ImportError:
            try:
                import gurobipy
                self.solver = "gurobi"
                self.solver_available = True
                logger.info("Using Gurobi as the optimization solver")
            except ImportError:
                try:
                    import cvxpy
                    self.solver = "cvxpy"
                    self.solver_available = True
                    logger.info("Using CVXPY as the optimization solver")
                except ImportError:
                    logger.warning("No optimization solver found. Install PuLP, Gurobi, or CVXPY to use MIQP allocator.")
                    self.solver = None
    
    @timed("miqp_allocation")
    def allocate(self, tasks: Dict[str, Any], vehicles: Dict[str, Any], 
                cost_matrix: Optional[Dict[Tuple[str, str], float]] = None) -> AllocationResult:
        """
        Allocate tasks to vehicles using mixed integer quadratic programming.
        
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
        
        # Check if solver is available
        if not self.solver_available:
            logger.error("No optimization solver available. Install PuLP, Gurobi, or CVXPY.")
            return AllocationResult(
                status=AllocationStatus.FAILED,
                computation_time=time.time() - start_time,
                metadata={"error": "No optimization solver available"}
            )
        
        # Create default cost matrix if not provided
        if cost_matrix is None:
            cost_matrix = self._create_default_cost_matrix(tasks, vehicles)
        
        # Solve using the available solver
        if self.solver == "pulp":
            return self._solve_with_pulp(tasks, vehicles, cost_matrix, start_time)
        elif self.solver == "gurobi":
            return self._solve_with_gurobi(tasks, vehicles, cost_matrix, start_time)
        elif self.solver == "cvxpy":
            return self._solve_with_cvxpy(tasks, vehicles, cost_matrix, start_time)
        else:
            logger.error("Solver is set but not properly initialized")
            return AllocationResult(
                status=AllocationStatus.FAILED,
                computation_time=time.time() - start_time,
                metadata={"error": "Solver initialization error"}
            )
    
    @timed("miqp_reallocation")
    def reallocate(self, current_allocations: Dict[str, List[str]],
                  new_tasks: Dict[str, Any], available_vehicles: Dict[str, Any], 
                  cost_matrix: Optional[Dict[Tuple[str, str], float]] = None) -> AllocationResult:
        """
        Reallocate tasks when conditions change.
        
        This implementation creates a new optimization problem that includes:
        1. Existing allocations as fixed constraints
        2. New tasks to be allocated to available vehicles
        
        Args:
            current_allocations: Current mapping from vehicle ID to task IDs
            new_tasks: Dict of new task IDs to task objects
            available_vehicles: Dict of available vehicle IDs to vehicle objects
            cost_matrix: Optional matrix of assignment costs
            
        Returns:
            AllocationResult: Updated allocation result
        """
        start_time = time.time()
        
        # Validate inputs
        if not new_tasks:
            return AllocationResult(
                allocations=current_allocations,
                status=AllocationStatus.NO_TASKS,
                computation_time=time.time() - start_time
            )
        
        if not available_vehicles:
            return AllocationResult(
                allocations=current_allocations,
                status=AllocationStatus.NO_VEHICLES,
                unallocated_tasks=list(new_tasks.keys()),
                computation_time=time.time() - start_time
            )
        
        # Check if solver is available
        if not self.solver_available:
            logger.error("No optimization solver available. Install PuLP, Gurobi, or CVXPY.")
            return AllocationResult(
                allocations=current_allocations,
                status=AllocationStatus.FAILED,
                unallocated_tasks=list(new_tasks.keys()),
                computation_time=time.time() - start_time,
                metadata={"error": "No optimization solver available"}
            )
        
        # Create default cost matrix if not provided
        if cost_matrix is None:
            cost_matrix = self._create_default_cost_matrix(new_tasks, available_vehicles)
        
        # Allocate only new tasks to available vehicles
        if self.solver == "pulp":
            result = self._solve_with_pulp(new_tasks, available_vehicles, cost_matrix, start_time)
        elif self.solver == "gurobi":
            result = self._solve_with_gurobi(new_tasks, available_vehicles, cost_matrix, start_time)
        elif self.solver == "cvxpy":
            result = self._solve_with_cvxpy(new_tasks, available_vehicles, cost_matrix, start_time)
        else:
            logger.error("Solver is set but not properly initialized")
            return AllocationResult(
                allocations=current_allocations,
                status=AllocationStatus.FAILED,
                unallocated_tasks=list(new_tasks.keys()),
                computation_time=time.time() - start_time,
                metadata={"error": "Solver initialization error"}
            )
        
        # Merge allocations
        merged_allocations = dict(current_allocations)
        
        for vehicle_id, task_ids in result.allocations.items():
            if vehicle_id in merged_allocations:
                merged_allocations[vehicle_id].extend(task_ids)
            else:
                merged_allocations[vehicle_id] = task_ids
        
        # Update result with merged allocations
        result.allocations = merged_allocations
        
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
                
                # Add vehicle suitability factor
                if self.config.use_vehicle_suitability and not self._is_vehicle_suitable(vehicle, task):
                    cost = float('inf')
                
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
        # Same implementation as in PriorityAllocator
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
    
    def _solve_with_pulp(self, tasks: Dict[str, Any], vehicles: Dict[str, Any],
                       cost_matrix: Dict[Tuple[str, str], float], start_time: float) -> AllocationResult:
        """
        Solve allocation problem using PuLP solver.
        
        Args:
            tasks: Dict of task IDs to task objects
            vehicles: Dict of vehicle IDs to vehicle objects
            cost_matrix: Matrix of assignment costs
            start_time: Start time for computation time tracking
            
        Returns:
            AllocationResult: Allocation result
        """
        try:
            import pulp
        except ImportError:
            logger.error("PuLP is not installed. Please install it with 'pip install pulp'")
            return AllocationResult(
                status=AllocationStatus.FAILED,
                computation_time=time.time() - start_time,
                metadata={"error": "PuLP not installed"}
            )
        
        # Create problem
        problem = pulp.LpProblem("TaskAllocation", pulp.LpMinimize)
        
        # Create variables
        # Binary variable x[v,t] = 1 if vehicle v is assigned to task t, 0 otherwise
        x = {}
        for vehicle_id in vehicles:
            for task_id in tasks:
                x[(vehicle_id, task_id)] = pulp.LpVariable(
                    f"x_{vehicle_id}_{task_id}", 
                    cat=pulp.LpBinary
                )
        
        # Add constraints
        
        # Each task must be assigned to exactly one vehicle
        for task_id in tasks:
            problem += pulp.lpSum(x[(vehicle_id, task_id)] for vehicle_id in vehicles) == 1
        
        # Each vehicle can be assigned at most max_tasks_per_vehicle tasks
        for vehicle_id in vehicles:
            problem += pulp.lpSum(x[(vehicle_id, task_id)] for task_id in tasks) <= self.config.max_tasks_per_vehicle
        
        # Set objective function based on allocation objective
        if self.config.objective == AllocationObjective.MINIMIZE_MAKESPAN:
            # Need a variable for makespan
            makespan = pulp.LpVariable("makespan", lowBound=0)
            
            # Calculate load for each vehicle
            vehicle_loads = {}
            for vehicle_id in vehicles:
                vehicle_loads[vehicle_id] = pulp.lpSum(
                    x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                    for task_id in tasks
                )
                
                # Makespan constraint for each vehicle
                problem += makespan >= vehicle_loads[vehicle_id]
            
            # Objective: Minimize makespan
            problem += makespan
            
        elif self.config.objective == AllocationObjective.MINIMIZE_DISTANCE:
            # Objective: Minimize total distance/cost
            problem += pulp.lpSum(
                x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                for vehicle_id in vehicles for task_id in tasks
            )
            
        elif self.config.objective == AllocationObjective.BALANCE_WORKLOAD:
            # Need variables for vehicle loads and their deviation
            vehicle_loads = {}
            load_deviations = {}
            
            # Calculate load for each vehicle
            for vehicle_id in vehicles:
                vehicle_loads[vehicle_id] = pulp.lpSum(
                    x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                    for task_id in tasks
                )
            
            # Average load
            avg_load = pulp.lpSum(vehicle_loads[vehicle_id] for vehicle_id in vehicles) / len(vehicles)
            
            # Load deviations
            for vehicle_id in vehicles:
                # Create variables for absolute difference
                load_deviations[vehicle_id] = pulp.LpVariable(f"dev_{vehicle_id}", lowBound=0)
                
                # Add constraints for absolute difference
                problem += load_deviations[vehicle_id] >= vehicle_loads[vehicle_id] - avg_load
                problem += load_deviations[vehicle_id] >= avg_load - vehicle_loads[vehicle_id]
            
            # Objective: Minimize sum of deviations (balance) and total distance
            problem += (
                self.config.balance_weight * pulp.lpSum(load_deviations[vehicle_id] for vehicle_id in vehicles) +
                self.config.distance_weight * pulp.lpSum(
                    x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                    for vehicle_id in vehicles for task_id in tasks
                )
            )
            
        elif self.config.objective == AllocationObjective.MULTI_OBJECTIVE:
            # Need a variable for makespan
            makespan = pulp.LpVariable("makespan", lowBound=0)
            
            # Calculate load for each vehicle
            vehicle_loads = {}
            for vehicle_id in vehicles:
                vehicle_loads[vehicle_id] = pulp.lpSum(
                    x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                    for task_id in tasks
                )
                
                # Makespan constraint for each vehicle
                problem += makespan >= vehicle_loads[vehicle_id]
            
            # Workload balancing
            load_deviations = {}
            avg_load = pulp.lpSum(vehicle_loads[vehicle_id] for vehicle_id in vehicles) / len(vehicles)
            
            for vehicle_id in vehicles:
                # Create variables for absolute difference
                load_deviations[vehicle_id] = pulp.LpVariable(f"dev_{vehicle_id}", lowBound=0)
                
                # Add constraints for absolute difference
                problem += load_deviations[vehicle_id] >= vehicle_loads[vehicle_id] - avg_load
                problem += load_deviations[vehicle_id] >= avg_load - vehicle_loads[vehicle_id]
            
            # Total distance
            total_distance = pulp.lpSum(
                x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                for vehicle_id in vehicles for task_id in tasks
            )
            
            # Multi-objective function
            problem += (
                self.config.makespan_weight * makespan +
                self.config.distance_weight * total_distance +
                self.config.balance_weight * pulp.lpSum(load_deviations[vehicle_id] for vehicle_id in vehicles)
            )
        
        # Solve the problem
        problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=self.config.timeout))
        
        # Check if solution was found
        if problem.status == pulp.LpStatusOptimal or problem.status == pulp.LpStatusNotSolved:
            # Get allocation from solution
            allocations = {vehicle_id: [] for vehicle_id in vehicles}
            unallocated_tasks = []
            
            for vehicle_id in vehicles:
                for task_id in tasks:
                    if pulp.value(x[(vehicle_id, task_id)]) > 0.5:  # Binary variable should be close to 1
                        allocations[vehicle_id].append(task_id)
            
            # Check for unallocated tasks
            all_allocated_tasks = []
            for tasks_list in allocations.values():
                all_allocated_tasks.extend(tasks_list)
            
            unallocated_tasks = [task_id for task_id in tasks if task_id not in all_allocated_tasks]
            
            # Calculate objective value
            objective_value = pulp.value(problem.objective)
            
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
                    "solver_status": pulp.LpStatus[problem.status],
                    "iterations": problem.solutionCplex.get_problem_stats().get('iterations', 0) if hasattr(problem, 'solutionCplex') else 0
                }
            )
            
            return result
        else:
            logger.warning(f"Failed to find optimal solution: {pulp.LpStatus[problem.status]}")
            return AllocationResult(
                status=AllocationStatus.INFEASIBLE,
                computation_time=time.time() - start_time,
                metadata={"solver_status": pulp.LpStatus[problem.status]}
            )
    
    def _solve_with_gurobi(self, tasks: Dict[str, Any], vehicles: Dict[str, Any],
                        cost_matrix: Dict[Tuple[str, str], float], start_time: float) -> AllocationResult:
        """
        Solve allocation problem using Gurobi solver.
        
        Args:
            tasks: Dict of task IDs to task objects
            vehicles: Dict of vehicle IDs to vehicle objects
            cost_matrix: Matrix of assignment costs
            start_time: Start time for computation time tracking
            
        Returns:
            AllocationResult: Allocation result
        """
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            logger.error("Gurobi is not installed. Please install it with 'pip install gurobipy'")
            return AllocationResult(
                status=AllocationStatus.FAILED,
                computation_time=time.time() - start_time,
                metadata={"error": "Gurobi not installed"}
            )
        
        # Create a Gurobi model
        model = gp.Model("TaskAllocation")
        model.setParam('OutputFlag', 0)  # Suppress output
        model.setParam('TimeLimit', self.config.timeout)
        model.setParam('MIPGap', self.config.miqp_gap)
        
        # Create variables
        # Binary variable x[v,t] = 1 if vehicle v is assigned to task t, 0 otherwise
        x = {}
        for vehicle_id in vehicles:
            for task_id in tasks:
                x[(vehicle_id, task_id)] = model.addVar(
                    vtype=GRB.BINARY, 
                    name=f"x_{vehicle_id}_{task_id}"
                )
        
        # Add constraints
        
        # Each task must be assigned to exactly one vehicle
        for task_id in tasks:
            model.addConstr(
                gp.quicksum(x[(vehicle_id, task_id)] for vehicle_id in vehicles) == 1,
                name=f"task_{task_id}_assigned"
            )
        
        # Each vehicle can be assigned at most max_tasks_per_vehicle tasks
        for vehicle_id in vehicles:
            model.addConstr(
                gp.quicksum(x[(vehicle_id, task_id)] for task_id in tasks) <= self.config.max_tasks_per_vehicle,
                name=f"vehicle_{vehicle_id}_capacity"
            )
        
        # Set objective function based on allocation objective
        if self.config.objective == AllocationObjective.MINIMIZE_MAKESPAN:
            # Need a variable for makespan
            makespan = model.addVar(lb=0, name="makespan")
            
            # Calculate load for each vehicle
            for vehicle_id in vehicles:
                vehicle_load = gp.quicksum(
                    x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                    for task_id in tasks
                )
                
                # Makespan constraint for each vehicle
                model.addConstr(makespan >= vehicle_load, name=f"makespan_{vehicle_id}")
            
            # Objective: Minimize makespan
            model.setObjective(makespan, GRB.MINIMIZE)
            
        elif self.config.objective == AllocationObjective.MINIMIZE_DISTANCE:
            # Objective: Minimize total distance/cost
            model.setObjective(
                gp.quicksum(
                    x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                    for vehicle_id in vehicles for task_id in tasks
                ),
                GRB.MINIMIZE
            )
            
        elif self.config.objective == AllocationObjective.BALANCE_WORKLOAD:
            # Variables for vehicle loads
            vehicle_loads = {}
            for vehicle_id in vehicles:
                vehicle_loads[vehicle_id] = gp.quicksum(
                    x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                    for task_id in tasks
                )
            
            # Average load expression
            n_vehicles = len(vehicles)
            avg_load = gp.quicksum(vehicle_loads[vehicle_id] for vehicle_id in vehicles) / n_vehicles
            
            # Variables for absolute deviations
            load_deviations = {}
            for vehicle_id in vehicles:
                load_deviations[vehicle_id] = model.addVar(lb=0, name=f"dev_{vehicle_id}")
                
                # Constraints for absolute difference
                model.addConstr(load_deviations[vehicle_id] >= vehicle_loads[vehicle_id] - avg_load)
                model.addConstr(load_deviations[vehicle_id] >= avg_load - vehicle_loads[vehicle_id])
            
            # Total distance
            total_distance = gp.quicksum(
                x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                for vehicle_id in vehicles for task_id in tasks
            )
            
            # Objective: Minimize weighted sum of load deviations and total distance
            model.setObjective(
                self.config.balance_weight * gp.quicksum(load_deviations[vehicle_id] for vehicle_id in vehicles) +
                self.config.distance_weight * total_distance,
                GRB.MINIMIZE
            )
            
        elif self.config.objective == AllocationObjective.MULTI_OBJECTIVE:
            # Need a variable for makespan
            makespan = model.addVar(lb=0, name="makespan")
            
            # Calculate load for each vehicle
            vehicle_loads = {}
            for vehicle_id in vehicles:
                vehicle_loads[vehicle_id] = gp.quicksum(
                    x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                    for task_id in tasks
                )
                
                # Makespan constraint for each vehicle
                model.addConstr(makespan >= vehicle_loads[vehicle_id], name=f"makespan_{vehicle_id}")
            
            # Variables for load deviations (for balancing)
            n_vehicles = len(vehicles)
            avg_load = gp.quicksum(vehicle_loads[vehicle_id] for vehicle_id in vehicles) / n_vehicles
            
            load_deviations = {}
            for vehicle_id in vehicles:
                load_deviations[vehicle_id] = model.addVar(lb=0, name=f"dev_{vehicle_id}")
                
                # Constraints for absolute difference
                model.addConstr(load_deviations[vehicle_id] >= vehicle_loads[vehicle_id] - avg_load)
                model.addConstr(load_deviations[vehicle_id] >= avg_load - vehicle_loads[vehicle_id])
            
            # Total distance
            total_distance = gp.quicksum(
                x[(vehicle_id, task_id)] * cost_matrix.get((vehicle_id, task_id), float('inf'))
                for vehicle_id in vehicles for task_id in tasks
            )
            
            # Multi-objective function
            model.setObjective(
                self.config.makespan_weight * makespan +
                self.config.distance_weight * total_distance +
                self.config.balance_weight * gp.quicksum(load_deviations[vehicle_id] for vehicle_id in vehicles),
                GRB.MINIMIZE
            )
        
        # Optimize the model
        model.optimize()
        
        # Check if solution was found
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            # Get allocation from solution
            allocations = {vehicle_id: [] for vehicle_id in vehicles}
            
            for vehicle_id in vehicles:
                for task_id in tasks:
                    if x[(vehicle_id, task_id)].X > 0.5:  # Binary variable should be close to 1
                        allocations[vehicle_id].append(task_id)
            
            # Check for unallocated tasks
            all_allocated_tasks = []
            for tasks_list in allocations.values():
                all_allocated_tasks.extend(tasks_list)
            
            unallocated_tasks = [task_id for task_id in tasks if task_id not in all_allocated_tasks]
            
            # Calculate objective value
            objective_value = model.objVal
            
            # Determine status
            if model.status == GRB.OPTIMAL and not unallocated_tasks:
                status = AllocationStatus.SUCCESS
            elif model.status == GRB.TIME_LIMIT and not unallocated_tasks:
                status = AllocationStatus.SUCCESS
            elif unallocated_tasks:
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
                    "solver_status": str(model.status),
                    "iterations": model.IterCount,
                    "mip_gap": model.MIPGap
                }
            )
            
            return result
        else:
            logger.warning(f"Failed to find optimal solution: status {model.status}")
            return AllocationResult(
                status=AllocationStatus.INFEASIBLE,
                computation_time=time.time() - start_time,
                metadata={"solver_status": str(model.status)}
            )
    
    def _solve_with_cvxpy(self, tasks: Dict[str, Any], vehicles: Dict[str, Any],
                       cost_matrix: Dict[Tuple[str, str], float], start_time: float) -> AllocationResult:
        """
        Solve allocation problem using CVXPY solver.
        
        Args:
            tasks: Dict of task IDs to task objects
            vehicles: Dict of vehicle IDs to vehicle objects
            cost_matrix: Matrix of assignment costs
            start_time: Start time for computation time tracking
            
        Returns:
            AllocationResult: Allocation result
        """
        try:
            import cvxpy as cp
            import numpy as np
        except ImportError:
            logger.error("CVXPY is not installed. Please install it with 'pip install cvxpy'")
            return AllocationResult(
                status=AllocationStatus.FAILED,
                computation_time=time.time() - start_time,
                metadata={"error": "CVXPY not installed"}
            )
        
        # Create cost matrix as numpy array
        vehicle_ids = list(vehicles.keys())
        task_ids = list(tasks.keys())
        
        cost_arr = np.full((len(vehicle_ids), len(task_ids)), float('inf'))
        for i, vehicle_id in enumerate(vehicle_ids):
            for j, task_id in enumerate(task_ids):
                cost_arr[i, j] = cost_matrix.get((vehicle_id, task_id), float('inf'))
        
        # Create binary variable for vehicle-task assignment
        X = cp.Variable((len(vehicle_ids), len(task_ids)), boolean=True)
        
        # Create constraints
        constraints = []
        
        # Each task must be assigned to exactly one vehicle
        for j in range(len(task_ids)):
            constraints.append(cp.sum(X[:, j]) == 1)
        
        # Each vehicle can be assigned at most max_tasks_per_vehicle tasks
        for i in range(len(vehicle_ids)):
            constraints.append(cp.sum(X[i, :]) <= self.config.max_tasks_per_vehicle)
        
        # Set objective function based on allocation objective
        if self.config.objective == AllocationObjective.MINIMIZE_MAKESPAN:
            # Need a variable for makespan
            makespan = cp.Variable()
            
            # Vehicle loads
            for i in range(len(vehicle_ids)):
                vehicle_load = cp.sum(cp.multiply(X[i, :], cost_arr[i, :]))
                constraints.append(makespan >= vehicle_load)
            
            # Objective: Minimize makespan
            objective = cp.Minimize(makespan)
            
        elif self.config.objective == AllocationObjective.MINIMIZE_DISTANCE:
            # Objective: Minimize total distance/cost
            objective = cp.Minimize(cp.sum(cp.multiply(X, cost_arr)))
            
        elif self.config.objective == AllocationObjective.BALANCE_WORKLOAD:
            # Vehicle loads
            vehicle_loads = [cp.sum(cp.multiply(X[i, :], cost_arr[i, :])) for i in range(len(vehicle_ids))]
            
            # Average load
            avg_load = cp.sum(vehicle_loads) / len(vehicle_ids)
            
            # Load deviations
            load_deviations = [cp.Variable() for _ in range(len(vehicle_ids))]
            for i in range(len(vehicle_ids)):
                constraints.append(load_deviations[i] >= vehicle_loads[i] - avg_load)
                constraints.append(load_deviations[i] >= avg_load - vehicle_loads[i])
            
            # Objective: Minimize weighted sum of deviations and total distance
            objective = cp.Minimize(
                self.config.balance_weight * cp.sum(load_deviations) +
                self.config.distance_weight * cp.sum(cp.multiply(X, cost_arr))
            )
            
        elif self.config.objective == AllocationObjective.MULTI_OBJECTIVE:
            # Variable for makespan
            makespan = cp.Variable()
            
            # Vehicle loads
            vehicle_loads = [cp.sum(cp.multiply(X[i, :], cost_arr[i, :])) for i in range(len(vehicle_ids))]
            
            # Makespan constraints
            for i in range(len(vehicle_ids)):
                constraints.append(makespan >= vehicle_loads[i])
            
            # Average load
            avg_load = cp.sum(vehicle_loads) / len(vehicle_ids)
            
            # Load deviations
            load_deviations = [cp.Variable() for _ in range(len(vehicle_ids))]
            for i in range(len(vehicle_ids)):
                constraints.append(load_deviations[i] >= vehicle_loads[i] - avg_load)
                constraints.append(load_deviations[i] >= avg_load - vehicle_loads[i])
            
            # Total distance
            total_distance = cp.sum(cp.multiply(X, cost_arr))
            
            # Multi-objective
            objective = cp.Minimize(
                self.config.makespan_weight * makespan +
                self.config.distance_weight * total_distance +
                self.config.balance_weight * cp.sum(load_deviations)
            )
        
        # Create and solve the problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.GLPK_MI, verbose=False, time_limit=self.config.timeout)
        except cp.SolverError:
            # Try with alternative solver
            try:
                problem.solve(solver=cp.CBC, verbose=False)
            except cp.SolverError:
                # If still fails, try more solvers
                problem.solve(verbose=False)
        
        # Check if solution was found
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Get allocation from solution
            allocations = {vehicle_id: [] for vehicle_id in vehicle_ids}
            
            for i, vehicle_id in enumerate(vehicle_ids):
                for j, task_id in enumerate(task_ids):
                    if X.value[i, j] > 0.5:  # Binary variable should be close to 1
                        allocations[vehicle_id].append(task_id)
            
            # Check for unallocated tasks
            all_allocated_tasks = []
            for tasks_list in allocations.values():
                all_allocated_tasks.extend(tasks_list)
            
            unallocated_tasks = [task_id for task_id in tasks if task_id not in all_allocated_tasks]
            
            # Calculate objective value
            objective_value = problem.value
            
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
                    "solver_status": problem.status,
                    "solver": problem.solver_stats.solver_name if hasattr(problem, 'solver_stats') else 'unknown'
                }
            )
            
            return result
        else:
            logger.warning(f"Failed to find optimal solution: {problem.status}")
            return AllocationResult(
                status=AllocationStatus.INFEASIBLE,
                computation_time=time.time() - start_time,
                metadata={"solver_status": problem.status}
            )