"""
Hybrid A* path planning implementation.

Provides a hybrid approach combining discrete A* search with continuous
motion primitives, suitable for vehicles with non-holonomic constraints.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import math
import time
import heapq
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import numpy as np

from utils.geo.coordinates import Point2D
from utils.math.trajectories import Path, PathType
from utils.math.vectors import Vector2D
from utils.logger import get_logger, timed

from algorithms.planning.common import (
    PlanningStatus, MotionType, DrivingDirection, PathSegment, 
    PlanningResult, PlanningConfig, PlanningConstraints,
    merge_path_segments, discretize_path_with_headings
)
from algorithms.planning.reeds_shepp import ReedsSheppPlanner

# Get logger
logger = get_logger("planning.hybrid_astar")


@dataclass
class HybridAStarNode:
    """Node for Hybrid A* search."""
    x: float                    # x-coordinate
    y: float                    # y-coordinate
    heading: float              # heading in radians
    g_cost: float = 0.0         # cost from start
    f_cost: float = 0.0         # f = g + h (total estimated cost)
    is_forward: bool = True     # direction of motion
    steer_angle: float = 0.0    # steering angle
    parent: Optional['HybridAStarNode'] = None  # parent node
    
    def __eq__(self, other):
        """Compare nodes based on position and heading."""
        if not isinstance(other, HybridAStarNode):
            return False
        return (abs(self.x - other.x) < 0.1 and 
                abs(self.y - other.y) < 0.1 and 
                abs(self.heading - other.heading) < 0.1)
    
    def __hash__(self):
        """Hash based on discretized position and heading."""
        # Discretize coordinates and heading for hashing
        grid_x = int(self.x * 5)  # ~20cm grid
        grid_y = int(self.y * 5)
        grid_h = int(self.heading * 36 / (2 * math.pi))  # ~10° resolution
        return hash((grid_x, grid_y, grid_h))
    
    def __lt__(self, other):
        """Compare nodes based on f_cost for priority queue."""
        return self.f_cost < other.f_cost


class MotionPrimitive:
    """
    Motion primitive for vehicle movement.
    
    Represents a basic movement action for the vehicle.
    """
    
    def __init__(self, steer_angle: float, is_forward: bool, motion_type: MotionType):
        """
        Initialize a motion primitive.
        
        Args:
            steer_angle: Steering angle in radians
            is_forward: Whether the motion is forward
            motion_type: Type of motion
        """
        self.steer_angle = steer_angle
        self.is_forward = is_forward
        self.motion_type = motion_type
        
        # Derived direction
        self.direction = (DrivingDirection.FORWARD if is_forward 
                         else DrivingDirection.BACKWARD)
    
    def __repr__(self) -> str:
        """String representation of the motion primitive."""
        direction = "Forward" if self.is_forward else "Backward"
        return f"MotionPrimitive({direction}, {self.steer_angle:.2f}rad, {self.motion_type.name})"


class HybridAStarPlanner:
    """
    Hybrid A* path planner for vehicle navigation.
    
    Combines discrete grid-based search with continuous motion primitives
    to find feasible paths for vehicles with non-holonomic constraints.
    """
    
    def __init__(self, config: Optional[PlanningConfig] = None):
        """
        Initialize the Hybrid A* planner.
        
        Args:
            config: Planning configuration
        """
        self.config = config or PlanningConfig()
        
        # Motion parameters
        self.step_size = self.config.step_size
        self.min_turning_radius = self.config.constraints.min_turning_radius
        
        # Grid parameters
        self.grid_resolution = self.config.grid_resolution
        
        # Initialize Reeds-Shepp planner for heuristic
        self.rs_planner = ReedsSheppPlanner(self.config)
        
        # Generate motion primitives
        self.motion_primitives = self._generate_motion_primitives()
        
        logger.info(f"Initialized Hybrid A* planner with {len(self.motion_primitives)} motion primitives")
    
    def _generate_motion_primitives(self) -> List[MotionPrimitive]:
        """
        Generate motion primitives for node expansion.
        
        Returns:
            List[MotionPrimitive]: List of motion primitives
        """
        primitives = []
        
        # Maximum steering angle
        max_steer = self.config.constraints.max_steering_angle
        
        # Forward motion
        primitives.append(MotionPrimitive(0.0, True, MotionType.FORWARD))  # Straight forward
        primitives.append(MotionPrimitive(max_steer, True, MotionType.ARC_LEFT))  # Max left turn
        primitives.append(MotionPrimitive(-max_steer, True, MotionType.ARC_RIGHT))  # Max right turn
        primitives.append(MotionPrimitive(max_steer/2, True, MotionType.ARC_LEFT))  # Half left turn
        primitives.append(MotionPrimitive(-max_steer/2, True, MotionType.ARC_RIGHT))  # Half right turn
        
        # Backward motion (if allowed)
        if self.config.constraints.allow_reverse:
            primitives.append(MotionPrimitive(0.0, False, MotionType.BACKWARD))  # Straight backward
            primitives.append(MotionPrimitive(max_steer, False, MotionType.ARC_LEFT))  # Max left backward
            primitives.append(MotionPrimitive(-max_steer, False, MotionType.ARC_RIGHT))  # Max right backward
            primitives.append(MotionPrimitive(max_steer/2, False, MotionType.ARC_LEFT))  # Half left backward
            primitives.append(MotionPrimitive(-max_steer/2, False, MotionType.ARC_RIGHT))  # Half right backward
        
        return primitives
    
    @timed("hybrid_astar_planning")
    def plan(self, start_x: float, start_y: float, start_heading: float,
             goal_x: float, goal_y: float, goal_heading: float,
             obstacles: List[Any] = None) -> PlanningResult:
        """
        Plan a path from start to goal using Hybrid A*.
        
        Args:
            start_x: Starting x-coordinate
            start_y: Starting y-coordinate
            start_heading: Starting heading in radians
            goal_x: Goal x-coordinate
            goal_y: Goal y-coordinate
            goal_heading: Goal heading in radians
            obstacles: List of obstacles
            
        Returns:
            PlanningResult: Planning result with path and metadata
        """
        start_time = time.time()
        
        # Initialize start and goal nodes
        start_node = HybridAStarNode(
            x=start_x, 
            y=start_y, 
            heading=start_heading,
            g_cost=0.0,
            is_forward=True
        )
        
        goal_node = HybridAStarNode(
            x=goal_x, 
            y=goal_y, 
            heading=goal_heading
        )
        
        # Calculate heuristic for start node
        h_cost = self._heuristic(start_node, goal_node)
        start_node.f_cost = h_cost
        
        # Initialize open and closed sets
        open_set = []  # Priority queue
        heapq.heappush(open_set, (start_node.f_cost, id(start_node), start_node))
        closed_set = set()  # Set of visited nodes
        
        # Initialize grid-based closed set for efficient collision checking
        grid_closed = {}
        
        # Track statistics
        iterations = 0
        max_iterations = self.config.max_iterations
        time_limit = self.config.time_limit
        
        # Main search loop
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Check time limit
            if time.time() - start_time > time_limit:
                logger.warning(f"Hybrid A* search timed out after {iterations} iterations")
                return PlanningResult(
                    status=PlanningStatus.TIMEOUT,
                    computation_time=time.time() - start_time,
                    iterations=iterations
                )
            
            # Get node with lowest f_cost
            _, _, current = heapq.heappop(open_set)
            
            # Check if goal reached
            if self._is_goal(current, goal_node):
                logger.info(f"Goal reached after {iterations} iterations")
                path_result = self._reconstruct_path(current)
                
                # Calculate computation time
                computation_time = time.time() - start_time
                
                return PlanningResult(
                    path=path_result.path,
                    segments=path_result.segments,
                    status=PlanningStatus.SUCCESS,
                    computation_time=computation_time,
                    iterations=iterations,
                    cost=current.g_cost,
                    metadata={
                        "nodes_explored": iterations,
                        "final_heading_error": abs(self._normalize_angle(
                            current.heading - goal_heading)),
                        "direction_changes": path_result.metadata.get("direction_changes", 0)
                    }
                )
            
            # Skip if node already in closed set
            node_hash = hash(current)
            if node_hash in closed_set:
                continue
            
            # Add to closed set
            closed_set.add(node_hash)
            
            # Get grid cell for current node
            grid_x = int(current.x / self.grid_resolution)
            grid_y = int(current.y / self.grid_resolution)
            grid_h = int(current.heading * 36 / (2 * math.pi))  # ~10° resolution
            grid_key = (grid_x, grid_y, grid_h)
            
            # Skip if we've already found a better path to this grid cell
            if grid_key in grid_closed and grid_closed[grid_key] <= current.g_cost:
                continue
            
            # Update grid closed set
            grid_closed[grid_key] = current.g_cost
            
            # Expand node using motion primitives
            for motion in self.motion_primitives:
                # Skip motion if would cause too many direction changes
                if (current.parent and 
                    current.is_forward != motion.is_forward and
                    self._count_direction_changes(current) >= self.config.constraints.direction_changes_limit - 1):
                    continue
                
                # Generate successor node
                successor = self._apply_motion(current, motion)
                
                # Skip if successor is None or collides with obstacles
                if not successor or (obstacles and self._check_collision(successor, obstacles)):
                    continue
                
                # Calculate costs
                g_cost = current.g_cost + self._motion_cost(current, successor, motion)
                h_cost = self._heuristic(successor, goal_node)
                f_cost = g_cost + h_cost
                
                # Update costs
                successor.g_cost = g_cost
                successor.f_cost = f_cost
                
                # Add to open set
                heapq.heappush(open_set, (successor.f_cost, id(successor), successor))
        
        # No path found
        if iterations >= max_iterations:
            logger.warning(f"Hybrid A* search reached maximum iterations ({max_iterations})")
            status = PlanningStatus.FAILURE
        else:
            logger.warning("Hybrid A* search failed (no path found)")
            status = PlanningStatus.NO_SOLUTION
        
        return PlanningResult(
            status=status,
            computation_time=time.time() - start_time,
            iterations=iterations
        )
    
    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-pi, pi].
        
        Args:
            angle: Angle in radians
            
        Returns:
            float: Normalized angle
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle <= -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _is_goal(self, current: HybridAStarNode, goal: HybridAStarNode) -> bool:
        """
        Check if current node is close enough to goal.
        
        Args:
            current: Current node
            goal: Goal node
            
        Returns:
            bool: True if goal reached
        """
        # Distance to goal
        dx = current.x - goal.x
        dy = current.y - goal.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Heading difference
        heading_diff = abs(self._normalize_angle(current.heading - goal.heading))
        
        # Goal tolerance
        position_tolerance = self.grid_resolution * 2
        heading_tolerance = math.radians(15)  # 15 degrees
        
        return distance < position_tolerance and heading_diff < heading_tolerance
    
    def _heuristic(self, node: HybridAStarNode, goal: HybridAStarNode) -> float:
        """
        Calculate heuristic cost from node to goal.
        
        Args:
            node: Current node
            goal: Goal node
            
        Returns:
            float: Heuristic cost
        """
        # Use Reeds-Shepp curve length as heuristic if enabled
        if self.config.constraints.allow_reverse:
            try:
                # Plan Reeds-Shepp path
                rs_result = self.rs_planner.plan(
                    node.x, node.y, node.heading,
                    goal.x, goal.y, goal.heading
                )
                
                if rs_result.status == PlanningStatus.SUCCESS:
                    # Add a small weight to the RS path cost
                    return rs_result.cost * self.config.heuristic_weight
            except Exception as e:
                logger.warning(f"Error computing RS heuristic: {str(e)}")
        
        # Fallback to Euclidean distance + heading difference
        dx = goal.x - node.x
        dy = goal.y - node.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Heading difference (normalized to [0, pi])
        heading_diff = abs(self._normalize_angle(goal.heading - node.heading))
        heading_cost = heading_diff * self.min_turning_radius
        
        return (distance + heading_cost) * self.config.heuristic_weight
    
    def _apply_motion(self, node: HybridAStarNode, motion: MotionPrimitive) -> Optional[HybridAStarNode]:
        """
        Apply a motion primitive to a node.
        
        Args:
            node: Current node
            motion: Motion primitive to apply
            
        Returns:
            Optional[HybridAStarNode]: New node, or None if invalid
        """
        # Vehicle parameters
        L = self.config.constraints.vehicle_length  # wheelbase
        
        # Get direction multiplier (1 for forward, -1 for backward)
        direction = 1 if motion.is_forward else -1
        
        # Calculate new heading and position
        # Simple bicycle model
        beta = math.atan(math.tan(motion.steer_angle) / 2)  # Slip angle at center of mass
        
        # New heading after moving step_size distance
        new_heading = node.heading + direction * self.step_size * math.tan(motion.steer_angle) / L
        new_heading = self._normalize_angle(new_heading)
        
        # New position
        new_x = node.x + self.step_size * math.cos(node.heading + direction * beta)
        new_y = node.y + self.step_size * math.sin(node.heading + direction * beta)
        
        # Create new node
        successor = HybridAStarNode(
            x=new_x,
            y=new_y,
            heading=new_heading,
            is_forward=motion.is_forward,
            steer_angle=motion.steer_angle,
            parent=node
        )
        
        return successor
    
    def _motion_cost(self, from_node: HybridAStarNode, to_node: HybridAStarNode, 
                    motion: MotionPrimitive) -> float:
        """
        Calculate cost of a motion.
        
        Args:
            from_node: Starting node
            to_node: Ending node
            motion: Motion primitive applied
            
        Returns:
            float: Motion cost
        """
        # Base cost is the distance traveled
        cost = self.step_size
        
        # Additional cost for steering
        steer_cost = abs(motion.steer_angle) * 0.1
        
        # Additional cost for reversing (if configured)
        reverse_cost = 0.0
        if not motion.is_forward:
            reverse_cost = self.step_size * 0.5
        
        # Additional cost for changing direction
        direction_change_cost = 0.0
        if from_node.parent and from_node.is_forward != motion.is_forward:
            direction_change_cost = self.step_size * 2.0
        
        return cost + steer_cost + reverse_cost + direction_change_cost
    
    def _check_collision(self, node: HybridAStarNode, obstacles: List[Any]) -> bool:
        """
        Check if a node collides with obstacles.
        
        Args:
            node: Node to check
            obstacles: List of obstacles
            
        Returns:
            bool: True if collision detected
        """
        # Get vehicle dimensions
        vehicle_width = self.config.constraints.vehicle_width
        vehicle_length = self.config.constraints.vehicle_length
        
        # Simplify as a circle for quick check
        vehicle_radius = math.sqrt((vehicle_width/2)**2 + (vehicle_length/2)**2)
        
        # Check against each obstacle
        for obstacle in obstacles:
            if hasattr(obstacle, 'distance_to_point'):
                # Use obstacle's distance function if available
                point = Point2D(node.x, node.y)
                distance = obstacle.distance_to_point(point)
                if distance < vehicle_radius:
                    return True
            elif hasattr(obstacle, 'contains_point'):
                # Use contains_point method if available
                point = Point2D(node.x, node.y)
                if obstacle.contains_point(point):
                    return True
            else:
                # Assume obstacle has x, y attributes
                if hasattr(obstacle, 'x') and hasattr(obstacle, 'y'):
                    dx = node.x - obstacle.x
                    dy = node.y - obstacle.y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Check if circle around vehicle collides with circle around obstacle
                    obstacle_radius = getattr(obstacle, 'radius', 1.0)
                    if distance < vehicle_radius + obstacle_radius:
                        return True
        
        return False
    
    def _count_direction_changes(self, node: HybridAStarNode) -> int:
        """
        Count the number of direction changes in the path to a node.
        
        Args:
            node: Node to analyze
            
        Returns:
            int: Number of direction changes
        """
        count = 0
        current = node
        
        while current.parent and current.parent.parent:
            if current.is_forward != current.parent.is_forward:
                count += 1
            current = current.parent
            
        return count
    
    def _reconstruct_path(self, goal_node: HybridAStarNode) -> PlanningResult:
        """
        Reconstruct the path from start to goal.
        
        Args:
            goal_node: Goal node with parent links
            
        Returns:
            PlanningResult: Planning result with path
        """
        # Collect nodes in reverse order
        nodes = []
        current = goal_node
        
        while current:
            nodes.append(current)
            current = current.parent
        
        # Reverse to get start-to-goal order
        nodes.reverse()
        
        # Extract segments
        segments = []
        direction_changes = 0
        
        # Process nodes into segments
        current_segment_points = []
        current_direction = nodes[0].is_forward if nodes else True
        current_steer = nodes[0].steer_angle if nodes else 0.0
        
        for i, node in enumerate(nodes):
            # Add point to current segment
            current_segment_points.append(Point2D(node.x, node.y))
            
            # Check if we need to start a new segment
            if i < len(nodes) - 1:
                next_node = nodes[i + 1]
                
                if (node.is_forward != next_node.is_forward or 
                    abs(node.steer_angle - next_node.steer_angle) > 0.1):
                    
                    # Finish current segment
                    if current_segment_points:
                        # Determine motion type
                        if abs(current_steer) < 0.01:
                            motion_type = (MotionType.FORWARD if current_direction 
                                          else MotionType.BACKWARD)
                        else:
                            if current_steer > 0:
                                motion_type = MotionType.ARC_LEFT
                            else:
                                motion_type = MotionType.ARC_RIGHT
                        
                        # Create segment
                        direction = (DrivingDirection.FORWARD if current_direction 
                                    else DrivingDirection.BACKWARD)
                        
                        # Skip segments with only one point
                        if len(current_segment_points) > 1:
                            segments.append(PathSegment(
                                points=current_segment_points,
                                motion_type=motion_type,
                                direction=direction,
                                length=self.step_size * (len(current_segment_points) - 1),
                                curvature=abs(current_steer) / self.config.constraints.vehicle_length
                                         if abs(current_steer) > 0.01 else 0.0
                            ))
                    
                    # Start new segment
                    current_segment_points = [Point2D(node.x, node.y)]
                    
                    # Check for direction change
                    if current_direction != next_node.is_forward:
                        direction_changes += 1
                    
                    # Update current direction and steering
                    current_direction = next_node.is_forward
                    current_steer = next_node.steer_angle
        
        # Add final segment
        if current_segment_points and len(current_segment_points) > 1:
            # Determine motion type
            if abs(current_steer) < 0.01:
                motion_type = (MotionType.FORWARD if current_direction 
                              else MotionType.BACKWARD)
            else:
                if current_steer > 0:
                    motion_type = MotionType.ARC_LEFT
                else:
                    motion_type = MotionType.ARC_RIGHT
            
            # Create segment
            direction = (DrivingDirection.FORWARD if current_direction 
                        else DrivingDirection.BACKWARD)
            
            segments.append(PathSegment(
                points=current_segment_points,
                motion_type=motion_type,
                direction=direction,
                length=self.step_size * (len(current_segment_points) - 1),
                curvature=abs(current_steer) / self.config.constraints.vehicle_length
                         if abs(current_steer) > 0.01 else 0.0
            ))
        
        # Merge segments to create path
        path = merge_path_segments(segments)
        
        # Add metadata
        if path and path.metadata:
            path.metadata["direction_changes"] = direction_changes
        
        # Create planning result
        result = PlanningResult(
            path=path,
            segments=segments,
            status=PlanningStatus.SUCCESS,
            cost=goal_node.g_cost,
            metadata={
                "direction_changes": direction_changes,
                "segments": len(segments)
            }
        )
        
        return result
    
    def plan_path(self, start: Point2D, start_heading: float,
                 goal: Point2D, goal_heading: float,
                 obstacles: List[Any] = None) -> PlanningResult:
        """
        Plan a path using Hybrid A* (alternative interface).
        
        Args:
            start: Start point
            start_heading: Start heading in radians
            goal: Goal point
            goal_heading: Goal heading in radians
            obstacles: List of obstacles
            
        Returns:
            PlanningResult: Planning result
        """
        return self.plan(
            start.x, start.y, start_heading,
            goal.x, goal.y, goal_heading,
            obstacles
        )