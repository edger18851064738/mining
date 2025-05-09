"""
Common structures and utilities for path planning algorithms.

This module provides shared data structures, result formats, configuration settings,
and utility functions used across different path planning implementations.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import math
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field

from utils.geo.coordinates import Point2D
from utils.math.trajectories import Path, PathType, PathSmoothingMethod
from utils.math.vectors import Vector2D


class PlanningStatus(Enum):
    """Status codes for planning results."""
    SUCCESS = auto()           # Planning completed successfully
    FAILURE = auto()           # Planning failed to find a path
    TIMEOUT = auto()           # Planning timed out
    START_INVALID = auto()     # Starting position is invalid
    GOAL_INVALID = auto()      # Goal position is invalid
    NO_SOLUTION = auto()       # No solution exists
    PARTIAL_SOLUTION = auto()  # Only a partial solution was found


class MotionType(Enum):
    """Types of vehicle motions."""
    FORWARD = auto()       # Forward motion
    BACKWARD = auto()      # Backward motion
    TURN_LEFT = auto()     # Left turn
    TURN_RIGHT = auto()    # Right turn
    ARC_LEFT = auto()      # Left arc
    ARC_RIGHT = auto()     # Right arc
    STATIONARY = auto()    # No motion


class DrivingDirection(Enum):
    """Driving direction for the vehicle."""
    FORWARD = 1
    BACKWARD = -1


@dataclass
class PlanningConstraints:
    """
    Constraints for path planning.
    
    Contains physical and operational constraints for the vehicle.
    """
    
    # Vehicle physical constraints
    min_turning_radius: float = 5.0    # Minimum turning radius in meters
    max_steering_angle: float = 0.7    # Maximum steering angle in radians
    vehicle_length: float = 5.0        # Vehicle length in meters
    vehicle_width: float = 2.0         # Vehicle width in meters
    max_speed: float = 5.0             # Maximum speed in m/s
    max_acceleration: float = 1.0      # Maximum acceleration in m/s²
    
    # Planning constraints
    allow_reverse: bool = True         # Whether reverse motion is allowed
    direction_changes_limit: int = 10  # Maximum number of direction changes
    
    # Safety margins
    obstacle_margin: float = 1.0       # Safety margin around obstacles
    vehicle_margin: float = 1.0        # Safety margin around other vehicles


@dataclass
class PlanningConfig:
    """
    Configuration for path planning algorithms.
    
    Contains algorithm-specific parameters and limits.
    """
    
    # General parameters
    max_iterations: int = 10000        # Maximum number of iterations
    time_limit: float = 5.0            # Time limit in seconds
    step_size: float = 1.0             # Step size for discretization
    
    # Grid parameters
    grid_resolution: float = 0.5       # Grid cell size for hybrid algorithms
    
    # Heuristic parameters
    heuristic_weight: float = 1.0      # Weight factor for heuristic function
    
    # Optimization parameters
    smoothing_factor: float = 0.5      # Factor for path smoothing
    path_simplification: bool = True   # Whether to simplify the path
    smoothing_iterations: int = 10     # Number of iterations for path smoothing
    
    # Sampling parameters
    sampling_radius: float = 10.0      # Radius for RRT sampling
    
    # Cost weights
    distance_weight: float = 1.0       # Weight for distance cost
    smoothness_weight: float = 0.5     # Weight for smoothness cost
    direction_change_weight: float = 2.0  # Weight for direction change cost
    
    # Constraints
    constraints: PlanningConstraints = field(default_factory=PlanningConstraints)


@dataclass
class PathSegment:
    """
    A segment of a path with associated metadata.
    
    Represents a portion of a path with a specific motion type and direction.
    """
    points: List[Point2D]               # Points in the segment
    motion_type: MotionType             # Type of motion
    direction: DrivingDirection         # Driving direction
    length: float = 0.0                 # Length of the segment
    curvature: float = 0.0              # Curvature of the segment
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class PlanningResult:
    """
    Result of a path planning operation.
    
    Contains the planned path, status, and metadata.
    """
    path: Optional[Path] = None                  # Complete path
    segments: List[PathSegment] = field(default_factory=list)  # Path segments
    status: PlanningStatus = PlanningStatus.FAILURE  # Planning status
    computation_time: float = 0.0                # Computation time in seconds
    iterations: int = 0                          # Number of iterations
    cost: float = float('inf')                   # Path cost
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


def merge_path_segments(segments: List[PathSegment]) -> Path:
    """
    Merge path segments into a single continuous path.
    
    Args:
        segments: List of path segments
        
    Returns:
        Path: Combined path
    """
    if not segments:
        return Path([])
    
    all_points = []
    
    # Collect all points from segments
    for i, segment in enumerate(segments):
        if i == 0:
            # Include all points from first segment
            all_points.extend(segment.points)
        else:
            # Skip first point of subsequent segments if it's the same as the last point of previous segment
            if segment.points and all_points and segment.points[0] == all_points[-1]:
                all_points.extend(segment.points[1:])
            else:
                all_points.extend(segment.points)
    
    # Determine path type
    path_type = PathType.HYBRID
    
    # Create metadata
    metadata = {
        "segments": len(segments),
        "segment_types": [segment.motion_type.name for segment in segments],
        "segment_directions": [segment.direction.name for segment in segments],
        "direction_changes": sum(1 for i in range(1, len(segments)) 
                               if segments[i].direction != segments[i-1].direction)
    }
    
    return Path(all_points, path_type, metadata)


def discretize_path_with_headings(path: Path, step_size: float) -> List[Tuple[Point2D, float]]:
    """
    Discretize a path into points with associated headings.
    
    Args:
        path: Input path
        step_size: Distance between points
        
    Returns:
        List[Tuple[Point2D, float]]: List of (point, heading) tuples
    """
    if len(path.points) < 2:
        return []
    
    result = []
    
    # Resample path by distance
    resampled_path = path.resample_by_distance(step_size)
    
    # Calculate headings
    for i in range(len(resampled_path.points)):
        if i == 0:
            # For first point, use direction to next point
            dx = resampled_path.points[1].x - resampled_path.points[0].x
            dy = resampled_path.points[1].y - resampled_path.points[0].y
        elif i == len(resampled_path.points) - 1:
            # For last point, use direction from previous point
            dx = resampled_path.points[i].x - resampled_path.points[i-1].x
            dy = resampled_path.points[i].y - resampled_path.points[i-1].y
        else:
            # For middle points, average direction from previous and to next
            dx1 = resampled_path.points[i].x - resampled_path.points[i-1].x
            dy1 = resampled_path.points[i].y - resampled_path.points[i-1].y
            dx2 = resampled_path.points[i+1].x - resampled_path.points[i].x
            dy2 = resampled_path.points[i+1].y - resampled_path.points[i].y
            
            # Average directions
            dx = (dx1 + dx2) / 2
            dy = (dy1 + dy2) / 2
        
        # Calculate heading
        heading = math.atan2(dy, dx)
        
        result.append((resampled_path.points[i], heading))
    
    return result


def check_collision(path: Path, obstacles: List[Any], vehicle_width: float, vehicle_length: float) -> bool:
    """
    Check if a path collides with any obstacles.
    
    Args:
        path: Path to check
        obstacles: List of obstacles
        vehicle_width: Width of the vehicle
        vehicle_length: Length of the vehicle
        
    Returns:
        bool: True if collision detected, False otherwise
    """
    # Calculate vehicle radius (approximating as circle)
    vehicle_radius = math.sqrt((vehicle_width/2)**2 + (vehicle_length/2)**2)
    
    for point in path.points:
        for obstacle in obstacles:
            if hasattr(obstacle, 'contains_point'):
                # Use contains_point method if available
                if obstacle.contains_point(point):
                    return True
            elif hasattr(obstacle, 'distance_to_point'):
                # Use distance_to_point method if available
                if obstacle.distance_to_point(point) < vehicle_radius:
                    return True
    
    return False


def calculate_path_cost(path: Path, config: PlanningConfig) -> float:
    """
    Calculate cost of a path based on multiple factors.
    
    Args:
        path: Path to evaluate
        config: Planning configuration with cost weights
        
    Returns:
        float: Total path cost
    """
    if not path.points:
        return float('inf')
    
    # Distance cost
    distance_cost = path.length * config.distance_weight
    
    # Smoothness cost (approximated by summing angle changes)
    smoothness_cost = 0.0
    if len(path.points) >= 3:
        for i in range(1, len(path.points) - 1):
            p1, p2, p3 = path.points[i-1], path.points[i], path.points[i+1]
            
            # Vectors from p2 to p1 and p3
            v1x, v1y = p1.x - p2.x, p1.y - p2.y
            v2x, v2y = p3.x - p2.x, p3.y - p2.y
            
            # Normalize vectors
            v1_len = math.sqrt(v1x**2 + v1y**2)
            v2_len = math.sqrt(v2x**2 + v2y**2)
            
            if v1_len > 0 and v2_len > 0:
                v1x, v1y = v1x / v1_len, v1y / v1_len
                v2x, v2y = v2x / v2_len, v2y / v2_len
                
                # Dot product gives cosine of angle
                cos_angle = v1x * v2x + v1y * v2y
                
                # Convert to angle and accumulate
                # (1 - cos_angle ranges from 0 to 2, with 0 being straight line)
                smoothness_cost += (1 - cos_angle)
    
    smoothness_cost *= config.smoothness_weight
    
    # Direction changes cost (from metadata if available)
    direction_change_cost = 0.0
    if path.metadata and "direction_changes" in path.metadata:
        direction_change_cost = path.metadata["direction_changes"] * config.direction_change_weight
    
    # Total cost
    total_cost = distance_cost + smoothness_cost + direction_change_cost
    
    return total_cost


def optimize_path(path: Path, config: PlanningConfig, obstacles: List[Any] = None) -> Path:
    """
    Optimize a path by smoothing and simplification.
    
    Args:
        path: Path to optimize
        config: Planning configuration
        obstacles: List of obstacles to avoid during optimization
        
    Returns:
        Path: Optimized path
    """
    if len(path.points) < 3:
        return Path(path.points.copy(), path.path_type, path.metadata.copy())
    
    # First, smooth the path
    smoothed_path = PathSmoothingMethod(
        path, 
        smoothing_factor=config.smoothing_factor, 
        iterations=config.smoothing_iterations
    )
    
    # Check if smoothed path collides with obstacles
    if obstacles and check_collision(
        smoothed_path, 
        obstacles, 
        config.constraints.vehicle_width, 
        config.constraints.vehicle_length
    ):
        # If collision, return original path
        return path
    
    return smoothed_path