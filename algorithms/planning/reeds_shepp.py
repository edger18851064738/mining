"""
Reeds-Shepp curves path planning implementation.

Provides an implementation of the Reeds-Shepp curves, which are optimal paths
for vehicles with a minimum turning radius constraint, allowing both forward
and backward motion.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import math
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import time

from utils.geo.coordinates import Point2D
from utils.math.trajectories import Path, PathType
from utils.math.vectors import Vector2D
from utils.logger import get_logger

from algorithms.planning.common import (
    PlanningStatus, MotionType, DrivingDirection, PathSegment, 
    PlanningResult, PlanningConfig, merge_path_segments
)

# Get logger
logger = get_logger("planning.reeds_shepp")


class RSCurveType(Enum):
    """Type of Reeds-Shepp curve segments."""
    STRAIGHT = auto()      # Straight line segment
    LEFT = auto()          # Left turn
    RIGHT = auto()         # Right turn


class RSPattern(Enum):
    """Reeds-Shepp path patterns."""
    # Format: [segment types...]_[segment directions...]
    # Where:
    # - S: Straight, L: Left, R: Right
    # - +: Forward, -: Backward
    
    # Straight patterns
    S_PLUS = "S+"                 # Straight forward
    S_MINUS = "S-"                # Straight backward
    
    # CSC patterns (Curve-Straight-Curve)
    LSL_PLUS = "LSL+++"           # Left-Straight-Left, all forward
    LSL_MINUS = "LSL---"          # Left-Straight-Left, all backward
    LSR_PLUS = "LSR+++"           # Left-Straight-Right, all forward
    LSR_MINUS = "LSR---"          # Left-Straight-Right, all backward
    RSL_PLUS = "RSL+++"           # Right-Straight-Left, all forward
    RSL_MINUS = "RSL---"          # Right-Straight-Left, all backward
    RSR_PLUS = "RSR+++"           # Right-Straight-Right, all forward
    RSR_MINUS = "RSR---"          # Right-Straight-Right, all backward
    
    # CCC patterns (Curve-Curve-Curve)
    LRL_PLUS = "LRL+++"           # Left-Right-Left, all forward
    LRL_MINUS = "LRL---"          # Left-Right-Left, all backward
    RLR_PLUS = "RLR+++"           # Right-Left-Right, all forward
    RLR_MINUS = "RLR---"          # Right-Left-Right, all backward
    
    # Mixed patterns
    LSL_MIXED = "LSL++-"          # Left-Straight-Left, mixed directions
    LSR_MIXED = "LSR++-"          # Left-Straight-Right, mixed directions
    RSL_MIXED = "RSL++-"          # Right-Straight-Left, mixed directions
    RSR_MIXED = "RSR++-"          # Right-Straight-Right, mixed directions
    
    # Additional patterns for completeness
    LRL_MIXED = "LRL++-"          # Left-Right-Left, mixed directions
    RLR_MIXED = "RLR++-"          # Right-Left-Right, mixed directions


@dataclass
class RSPathSegment:
    """A segment of a Reeds-Shepp path."""
    curve_type: RSCurveType             # Type of curve
    length: float                       # Length of the segment
    direction: DrivingDirection         # Driving direction


@dataclass
class RSPath:
    """Complete Reeds-Shepp path."""
    segments: List[RSPathSegment] = field(default_factory=list)  # Path segments
    total_length: float = 0.0           # Total path length
    pattern: Optional[RSPattern] = None  # Pattern type


class ReedsSheppPlanner:
    """
    Reeds-Shepp curve path planner for vehicles with minimum turning radius.
    
    Implements the Reeds-Shepp curves, which are optimal paths for vehicles 
    that can move forward and backward with a minimum turning radius constraint.
    """
    
    def __init__(self, config: Optional[PlanningConfig] = None):
        """
        Initialize the Reeds-Shepp planner.
        
        Args:
            config: Planning configuration
        """
        self.config = config or PlanningConfig()
        self.min_turning_radius = self.config.constraints.min_turning_radius
        logger.info(f"Initialized Reeds-Shepp planner with turning radius {self.min_turning_radius}")
    
    def plan(self, start_x: float, start_y: float, start_heading: float,
             goal_x: float, goal_y: float, goal_heading: float) -> PlanningResult:
        """
        Plan a Reeds-Shepp path from start to goal.
        
        Args:
            start_x: Starting x-coordinate
            start_y: Starting y-coordinate
            start_heading: Starting heading in radians
            goal_x: Goal x-coordinate
            goal_y: Goal y-coordinate
            goal_heading: Goal heading in radians
            
        Returns:
            PlanningResult: Planning result with path and metadata
        """
        start_time = time.time()
        
        # Transform goal to be relative to start position and orientation
        dx = goal_x - start_x
        dy = goal_y - start_y
        
        # Rotate coordinates so that start heading is along x-axis
        cos_angle = math.cos(-start_heading)
        sin_angle = math.sin(-start_heading)
        
        transformed_x = dx * cos_angle - dy * sin_angle
        transformed_y = dx * sin_angle + dy * cos_angle
        transformed_heading = goal_heading - start_heading
        
        # Normalize heading to [-pi, pi]
        transformed_heading = self._normalize_angle(transformed_heading)
        
        # Scale coordinates by turning radius
        scaled_x = transformed_x / self.min_turning_radius
        scaled_y = transformed_y / self.min_turning_radius
        
        logger.debug(f"Planning RS path: ({start_x}, {start_y}, {start_heading}) -> "
                    f"({goal_x}, {goal_y}, {goal_heading})")
        logger.debug(f"Transformed goal: ({scaled_x}, {scaled_y}, {transformed_heading})")
        
        # Find the optimal path
        rs_path = self._find_optimal_path(scaled_x, scaled_y, transformed_heading)
        
        if not rs_path or not rs_path.segments:
            # No path found
            logger.warning("No Reeds-Shepp path found")
            result = PlanningResult(
                status=PlanningStatus.NO_SOLUTION,
                computation_time=time.time() - start_time
            )
            return result
        
        # Convert the RS path to a sequence of points
        segments = self._convert_rs_path_to_segments(
            rs_path, start_x, start_y, start_heading
        )
        
        # Merge segments into a path
        path = merge_path_segments(segments)
        
        # Create the result
        result = PlanningResult(
            path=path,
            segments=segments,
            status=PlanningStatus.SUCCESS,
            computation_time=time.time() - start_time,
            cost=rs_path.total_length * self.min_turning_radius,
            metadata={
                "pattern": rs_path.pattern.name if rs_path.pattern else "UNKNOWN",
                "segment_lengths": [seg.length for seg in rs_path.segments],
                "segment_types": [seg.curve_type.name for seg in rs_path.segments],
                "segment_directions": [seg.direction.name for seg in rs_path.segments],
                "turning_radius": self.min_turning_radius,
                "direction_changes": sum(1 for i in range(1, len(rs_path.segments)) 
                                       if rs_path.segments[i].direction != rs_path.segments[i-1].direction)
            }
        )
        
        logger.info(f"Found RS path with length {result.cost:.2f}, "
                   f"pattern {rs_path.pattern.name if rs_path.pattern else 'UNKNOWN'}")
        
        return result
    
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
    
    def _find_optimal_path(self, x: float, y: float, phi: float) -> Optional[RSPath]:
        """
        Find the optimal Reeds-Shepp path.
        
        Args:
            x: Scaled x-coordinate of goal relative to start
            y: Scaled y-coordinate of goal relative to start
            phi: Heading difference in radians
            
        Returns:
            Optional[RSPath]: Optimal Reeds-Shepp path, or None if no path found
        """
        # Special case: straight line
        if abs(y) < 1e-6 and abs(phi) < 1e-6:
            # Straight path
            direction = DrivingDirection.FORWARD if x >= 0 else DrivingDirection.BACKWARD
            length = abs(x)
            
            rs_path = RSPath(
                segments=[
                    RSPathSegment(
                        curve_type=RSCurveType.STRAIGHT,
                        length=length,
                        direction=direction
                    )
                ],
                total_length=length,
                pattern=RSPattern.S_PLUS if direction == DrivingDirection.FORWARD else RSPattern.S_MINUS
            )
            
            return rs_path
        
        # Calculate all possible path types
        paths = []
        
        # CSC paths (Curve-Straight-Curve)
        paths.extend(self._compute_csc_paths(x, y, phi))
        
        # CCC paths (Curve-Curve-Curve)
        paths.extend(self._compute_ccc_paths(x, y, phi))
        
        if not paths:
            return None
        
        # Find the shortest path
        shortest_path = min(paths, key=lambda p: p.total_length)
        return shortest_path
    
    def _compute_csc_paths(self, x: float, y: float, phi: float) -> List[RSPath]:
        """
        Compute all CSC (Curve-Straight-Curve) paths.
        
        Args:
            x: Scaled x-coordinate of goal relative to start
            y: Scaled y-coordinate of goal relative to start
            phi: Heading difference in radians
            
        Returns:
            List[RSPath]: List of CSC paths
        """
        paths = []
        
        # LSL (Left-Straight-Left)
        lsl_path = self._compute_lsl(x, y, phi)
        if lsl_path:
            lsl_path.pattern = RSPattern.LSL_PLUS
            paths.append(lsl_path)
        
        # LSR (Left-Straight-Right)
        lsr_path = self._compute_lsr(x, y, phi)
        if lsr_path:
            lsr_path.pattern = RSPattern.LSR_PLUS
            paths.append(lsr_path)
        
        # RSL (Right-Straight-Left)
        rsl_path = self._compute_rsl(x, y, phi)
        if rsl_path:
            rsl_path.pattern = RSPattern.RSL_PLUS
            paths.append(rsl_path)
        
        # RSR (Right-Straight-Right)
        rsr_path = self._compute_rsr(x, y, phi)
        if rsr_path:
            rsr_path.pattern = RSPattern.RSR_PLUS
            paths.append(rsr_path)
        
        # Compute backward variants (reverse direction)
        # LSL_MINUS
        lsl_minus_path = self._compute_lsl(-x, -y, -phi)
        if lsl_minus_path:
            # Reverse all segment directions
            for segment in lsl_minus_path.segments:
                segment.direction = DrivingDirection.BACKWARD
            lsl_minus_path.pattern = RSPattern.LSL_MINUS
            paths.append(lsl_minus_path)
        
        # LSR_MINUS
        lsr_minus_path = self._compute_lsr(-x, -y, -phi)
        if lsr_minus_path:
            # Reverse all segment directions
            for segment in lsr_minus_path.segments:
                segment.direction = DrivingDirection.BACKWARD
            lsr_minus_path.pattern = RSPattern.LSR_MINUS
            paths.append(lsr_minus_path)
        
        # RSL_MINUS
        rsl_minus_path = self._compute_rsl(-x, -y, -phi)
        if rsl_minus_path:
            # Reverse all segment directions
            for segment in rsl_minus_path.segments:
                segment.direction = DrivingDirection.BACKWARD
            rsl_minus_path.pattern = RSPattern.RSL_MINUS
            paths.append(rsl_minus_path)
        
        # RSR_MINUS
        rsr_minus_path = self._compute_rsr(-x, -y, -phi)
        if rsr_minus_path:
            # Reverse all segment directions
            for segment in rsr_minus_path.segments:
                segment.direction = DrivingDirection.BACKWARD
            rsr_minus_path.pattern = RSPattern.RSR_MINUS
            paths.append(rsr_minus_path)
        
        return paths
    
    def _compute_lsl(self, x: float, y: float, phi: float) -> Optional[RSPath]:
        """
        Compute LSL path (Left-Straight-Left).
        
        Args:
            x: Scaled x-coordinate of goal relative to start
            y: Scaled y-coordinate of goal relative to start
            phi: Heading difference in radians
            
        Returns:
            Optional[RSPath]: LSL path if exists, None otherwise
        """
        # Center of the first circle (Left turn) is at (0, 1)
        # Center of the last circle (Left turn) is at (x - sin(phi), y - 1 + cos(phi))
        x_center2 = x - math.sin(phi)
        y_center2 = y - 1 + math.cos(phi)
        
        # Distance between circle centers
        d = math.sqrt(x_center2**2 + (y_center2 - 1)**2)
        
        # If centers are too close, no solution exists
        if d < 2:
            return None
        
        # Compute tangent lines between circles
        # For LSL, we use external tangent
        # Angle from first center to tangent point
        alpha = math.atan2(y_center2 - 1, x_center2)
        
        # Length of straight segment
        straight_length = d
        
        # Arc angles
        t1 = self._normalize_angle(alpha)
        t2 = self._normalize_angle(phi - alpha)
        
        # Create path segments
        segments = [
            RSPathSegment(RSCurveType.LEFT, t1, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.STRAIGHT, straight_length, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.LEFT, t2, DrivingDirection.FORWARD)
        ]
        
        # Calculate total length
        total_length = t1 + straight_length + t2
        
        return RSPath(segments=segments, total_length=total_length)
    
    def _compute_lsr(self, x: float, y: float, phi: float) -> Optional[RSPath]:
        """
        Compute LSR path (Left-Straight-Right).
        
        Args:
            x: Scaled x-coordinate of goal relative to start
            y: Scaled y-coordinate of goal relative to start
            phi: Heading difference in radians
            
        Returns:
            Optional[RSPath]: LSR path if exists, None otherwise
        """
        # Center of the first circle (Left turn) is at (0, 1)
        # Center of the last circle (Right turn) is at (x + sin(phi), y + 1 - cos(phi))
        x_center2 = x + math.sin(phi)
        y_center2 = y + 1 - math.cos(phi)
        
        # Distance between circle centers
        d = math.sqrt(x_center2**2 + (y_center2 - 1)**2)
        
        # If circles are overlapping, no solution exists
        if d < 2:
            return None
        
        # Compute tangent lines between circles
        # For LSR, we use cross tangent
        # Half the distance between tangent points
        h = 2 / d
        
        # Base angle for tangent
        alpha = math.asin(h)
        
        # Angle from first center to tangent point
        beta = math.atan2(y_center2 - 1, x_center2) - alpha
        
        # Angle from second center to tangent point
        gamma = beta + 2 * alpha
        
        # Length of straight segment
# Length of straight segment
        straight_length = math.sqrt(d**2 - 4)
        
        # Arc angles
        t1 = self._normalize_angle(beta)
        t2 = self._normalize_angle(gamma - phi)
        
        # Create path segments
        segments = [
            RSPathSegment(RSCurveType.LEFT, t1, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.STRAIGHT, straight_length, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.RIGHT, t2, DrivingDirection.FORWARD)
        ]
        
        # Calculate total length
        total_length = t1 + straight_length + t2
        
        return RSPath(segments=segments, total_length=total_length)
    
    def _compute_rsl(self, x: float, y: float, phi: float) -> Optional[RSPath]:
        """
        Compute RSL path (Right-Straight-Left).
        
        Args:
            x: Scaled x-coordinate of goal relative to start
            y: Scaled y-coordinate of goal relative to start
            phi: Heading difference in radians
            
        Returns:
            Optional[RSPath]: RSL path if exists, None otherwise
        """
        # Center of the first circle (Right turn) is at (0, -1)
        # Center of the last circle (Left turn) is at (x - sin(phi), y - 1 + cos(phi))
        x_center2 = x - math.sin(phi)
        y_center2 = y - 1 + cos(phi)
        
        # Distance between circle centers
        d = math.sqrt(x_center2**2 + (y_center2 + 1)**2)
        
        # If circles are overlapping, no solution exists
        if d < 2:
            return None
        
        # Compute tangent lines between circles
        # For RSL, we use cross tangent
        # Half the distance between tangent points
        h = 2 / d
        
        # Base angle for tangent
        alpha = math.asin(h)
        
        # Angle from first center to tangent point
        beta = math.atan2(y_center2 + 1, x_center2) + alpha
        
        # Angle from second center to tangent point
        gamma = beta - 2 * alpha
        
        # Length of straight segment
        straight_length = math.sqrt(d**2 - 4)
        
        # Arc angles
        t1 = self._normalize_angle(-beta)
        t2 = self._normalize_angle(-gamma + phi)
        
        # Create path segments
        segments = [
            RSPathSegment(RSCurveType.RIGHT, t1, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.STRAIGHT, straight_length, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.LEFT, t2, DrivingDirection.FORWARD)
        ]
        
        # Calculate total length
        total_length = t1 + straight_length + t2
        
        return RSPath(segments=segments, total_length=total_length)
    
    def _compute_rsr(self, x: float, y: float, phi: float) -> Optional[RSPath]:
        """
        Compute RSR path (Right-Straight-Right).
        
        Args:
            x: Scaled x-coordinate of goal relative to start
            y: Scaled y-coordinate of goal relative to start
            phi: Heading difference in radians
            
        Returns:
            Optional[RSPath]: RSR path if exists, None otherwise
        """
        # Center of the first circle (Right turn) is at (0, -1)
        # Center of the last circle (Right turn) is at (x + sin(phi), y + 1 - cos(phi))
        x_center2 = x + math.sin(phi)
        y_center2 = y + 1 - math.cos(phi)
        
        # Distance between circle centers
        d = math.sqrt(x_center2**2 + (y_center2 + 1)**2)
        
        # Compute tangent lines between circles
        # For RSR, we use external tangent
        # Angle from first center to tangent point
        alpha = math.atan2(y_center2 + 1, x_center2)
        
        # Length of straight segment
        straight_length = d
        
        # Arc angles
        t1 = self._normalize_angle(-alpha)
        t2 = self._normalize_angle(-phi + alpha)
        
        # Create path segments
        segments = [
            RSPathSegment(RSCurveType.RIGHT, t1, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.STRAIGHT, straight_length, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.RIGHT, t2, DrivingDirection.FORWARD)
        ]
        
        # Calculate total length
        total_length = t1 + straight_length + t2
        
        return RSPath(segments=segments, total_length=total_length)
    
    def _compute_ccc_paths(self, x: float, y: float, phi: float) -> List[RSPath]:
        """
        Compute all CCC (Curve-Curve-Curve) paths.
        
        Args:
            x: Scaled x-coordinate of goal relative to start
            y: Scaled y-coordinate of goal relative to start
            phi: Heading difference in radians
            
        Returns:
            List[RSPath]: List of CCC paths
        """
        paths = []
        
        # LRL (Left-Right-Left)
        lrl_path = self._compute_lrl(x, y, phi)
        if lrl_path:
            lrl_path.pattern = RSPattern.LRL_PLUS
            paths.append(lrl_path)
        
        # RLR (Right-Left-Right)
        rlr_path = self._compute_rlr(x, y, phi)
        if rlr_path:
            rlr_path.pattern = RSPattern.RLR_PLUS
            paths.append(rlr_path)
        
        # Compute backward variants
        # LRL_MINUS
        lrl_minus_path = self._compute_lrl(-x, -y, -phi)
        if lrl_minus_path:
            # Reverse all segment directions
            for segment in lrl_minus_path.segments:
                segment.direction = DrivingDirection.BACKWARD
            lrl_minus_path.pattern = RSPattern.LRL_MINUS
            paths.append(lrl_minus_path)
        
        # RLR_MINUS
        rlr_minus_path = self._compute_rlr(-x, -y, -phi)
        if rlr_minus_path:
            # Reverse all segment directions
            for segment in rlr_minus_path.segments:
                segment.direction = DrivingDirection.BACKWARD
            rlr_minus_path.pattern = RSPattern.RLR_MINUS
            paths.append(rlr_minus_path)
        
        return paths
    
    def _compute_lrl(self, x: float, y: float, phi: float) -> Optional[RSPath]:
        """
        Compute LRL path (Left-Right-Left).
        
        Args:
            x: Scaled x-coordinate of goal relative to start
            y: Scaled y-coordinate of goal relative to start
            phi: Heading difference in radians
            
        Returns:
            Optional[RSPath]: LRL path if exists, None otherwise
        """
        # Center of first circle (Left turn) is at (0, 1)
        # Center of last circle (Left turn) is at (x - sin(phi), y - 1 + cos(phi))
        x_center1 = 0
        y_center1 = 1
        x_center2 = x - math.sin(phi)
        y_center2 = y - 1 + math.cos(phi)
        
        # Distance between centers
        d = math.sqrt((x_center2 - x_center1)**2 + (y_center2 - y_center1)**2)
        
        # If circles are too far apart, no LRL solution exists
        if d > 4:
            return None
        
        # Calculate the midpoint of the path
        # For LRL, the middle segment is a Right turn
        # The center of this circle is determined by the tangent points on the first and last circles
        
        # Angle from first circle center to the line connecting the centers
        alpha = math.atan2(y_center2 - y_center1, x_center2 - x_center1)
        
        # Angle offset based on the distance
        beta = math.acos(d / 4)
        
        # Angles for each segment
        t1 = self._normalize_angle(alpha + beta)
        t2 = self._normalize_angle(2 * beta)
        t3 = self._normalize_angle(phi - alpha - beta)
        
        # Create path segments
        segments = [
            RSPathSegment(RSCurveType.LEFT, t1, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.RIGHT, t2, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.LEFT, t3, DrivingDirection.FORWARD)
        ]
        
        # Calculate total length
        total_length = t1 + t2 + t3
        
        return RSPath(segments=segments, total_length=total_length)
    
    def _compute_rlr(self, x: float, y: float, phi: float) -> Optional[RSPath]:
        """
        Compute RLR path (Right-Left-Right).
        
        Args:
            x: Scaled x-coordinate of goal relative to start
            y: Scaled y-coordinate of goal relative to start
            phi: Heading difference in radians
            
        Returns:
            Optional[RSPath]: RLR path if exists, None otherwise
        """
        # Center of first circle (Right turn) is at (0, -1)
        # Center of last circle (Right turn) is at (x + sin(phi), y + 1 - cos(phi))
        x_center1 = 0
        y_center1 = -1
        x_center2 = x + math.sin(phi)
        y_center2 = y + 1 - math.cos(phi)
        
        # Distance between centers
        d = math.sqrt((x_center2 - x_center1)**2 + (y_center2 - y_center1)**2)
        
        # If circles are too far apart, no RLR solution exists
        if d > 4:
            return None
        
        # Calculate the midpoint of the path
        # For RLR, the middle segment is a Left turn
        
        # Angle from first circle center to the line connecting the centers
        alpha = math.atan2(y_center2 - y_center1, x_center2 - x_center1)
        
        # Angle offset based on the distance
        beta = math.acos(d / 4)
        
        # Angles for each segment
        t1 = self._normalize_angle(alpha - beta)
        t2 = self._normalize_angle(2 * beta)
        t3 = self._normalize_angle(phi - alpha + beta)
        
        # Create path segments
        segments = [
            RSPathSegment(RSCurveType.RIGHT, t1, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.LEFT, t2, DrivingDirection.FORWARD),
            RSPathSegment(RSCurveType.RIGHT, t3, DrivingDirection.FORWARD)
        ]
        
        # Calculate total length
        total_length = t1 + t2 + t3
        
        return RSPath(segments=segments, total_length=total_length)
    
    def _convert_rs_path_to_segments(self, rs_path: RSPath, start_x: float, start_y: float, 
                                   start_heading: float) -> List[PathSegment]:
        """
        Convert a Reed-Shepp path to a sequence of PathSegments with actual points.
        
        Args:
            rs_path: Reed-Shepp path
            start_x: Starting x-coordinate
            start_y: Starting y-coordinate
            start_heading: Starting heading in radians
            
        Returns:
            List[PathSegment]: List of path segments with points
        """
        segments = []
        
        # Current pose
        x, y = start_x, start_y
        heading = start_heading
        
        for i, segment in enumerate(rs_path.segments):
            # Convert the segment to points
            if segment.curve_type == RSCurveType.STRAIGHT:
                # Straight segment
                points = self._generate_straight_segment(
                    x, y, heading, 
                    segment.length * self.min_turning_radius,
                    segment.direction
                )
                
                # Set motion type
                if segment.direction == DrivingDirection.FORWARD:
                    motion_type = MotionType.FORWARD
                else:
                    motion_type = MotionType.BACKWARD
                
            elif segment.curve_type == RSCurveType.LEFT:
                # Left turn
                points = self._generate_arc_segment(
                    x, y, heading,
                    segment.length,
                    self.min_turning_radius,
                    True,  # Left turn
                    segment.direction
                )
                
                # Set motion type
                if segment.direction == DrivingDirection.FORWARD:
                    motion_type = MotionType.ARC_LEFT if abs(segment.length) > 0.01 else MotionType.TURN_LEFT
                else:
                    motion_type = MotionType.ARC_LEFT if abs(segment.length) > 0.01 else MotionType.TURN_LEFT
                
            elif segment.curve_type == RSCurveType.RIGHT:
                # Right turn
                points = self._generate_arc_segment(
                    x, y, heading,
                    segment.length,
                    self.min_turning_radius,
                    False,  # Right turn
                    segment.direction
                )
                
                # Set motion type
                if segment.direction == DrivingDirection.FORWARD:
                    motion_type = MotionType.ARC_RIGHT if abs(segment.length) > 0.01 else MotionType.TURN_RIGHT
                else:
                    motion_type = MotionType.ARC_RIGHT if abs(segment.length) > 0.01 else MotionType.TURN_RIGHT
            
            # Skip empty segments
            if not points:
                continue
            
            # Update current pose
            x, y = points[-1].x, points[-1].y
            
            # Update heading based on segment type
            if segment.curve_type == RSCurveType.STRAIGHT:
                # Heading doesn't change for straight segments
                pass
            elif segment.curve_type == RSCurveType.LEFT:
                # Left turn, heading increases
                heading += segment.length * (1 if segment.direction == DrivingDirection.FORWARD else -1)
            elif segment.curve_type == RSCurveType.RIGHT:
                # Right turn, heading decreases
                heading -= segment.length * (1 if segment.direction == DrivingDirection.FORWARD else -1)
            
            # Normalize heading
            heading = self._normalize_angle(heading)
            
            # Create path segment
            path_segment = PathSegment(
                points=points,
                motion_type=motion_type,
                direction=segment.direction,
                length=segment.length * self.min_turning_radius,
                curvature=1.0 / self.min_turning_radius if segment.curve_type != RSCurveType.STRAIGHT else 0.0,
                metadata={
                    "curve_type": segment.curve_type.name,
                    "segment_index": i
                }
            )
            
            segments.append(path_segment)
        
        return segments
    
    def _generate_straight_segment(self, x: float, y: float, heading: float, 
                                  length: float, direction: DrivingDirection) -> List[Point2D]:
        """
        Generate points for a straight segment.
        
        Args:
            x: Starting x-coordinate
            y: Starting y-coordinate
            heading: Heading in radians
            length: Length of the segment
            direction: Driving direction
            
        Returns:
            List[Point2D]: List of points
        """
        # Adjust step size
        step_size = min(0.5, abs(length) / 10)
        if step_size < 0.01:
            step_size = 0.01
        
        points = []
        
        # Distance signs
        sign = 1 if direction == DrivingDirection.FORWARD else -1
        
        # Calculate direction vector
        dx = math.cos(heading) * sign
        dy = math.sin(heading) * sign
        
        # Number of steps
        num_steps = max(2, int(abs(length) / step_size))
        
        # Generate points
        for i in range(num_steps + 1):
            t = i / num_steps * abs(length)
            point_x = x + dx * t
            point_y = y + dy * t
            points.append(Point2D(point_x, point_y))
        
        return points
    
    def _generate_arc_segment(self, x: float, y: float, heading: float, 
                             angle: float, radius: float, is_left: bool,
                             direction: DrivingDirection) -> List[Point2D]:
        """
        Generate points for an arc segment.
        
        Args:
            x: Starting x-coordinate
            y: Starting y-coordinate
            heading: Heading in radians
            angle: Arc angle in radians
            radius: Turning radius
            is_left: Whether the turn is left
            direction: Driving direction
            
        Returns:
            List[Point2D]: List of points
        """
        # Skip if angle is too small
        if abs(angle) < 1e-6:
            return [Point2D(x, y)]
        
        # Calculate arc center
        if is_left:
            # Left turn: center is to the left of the vehicle
            center_x = x - radius * math.sin(heading)
            center_y = y + radius * math.cos(heading)
        else:
            # Right turn: center is to the right of the vehicle
            center_x = x + radius * math.sin(heading)
            center_y = y - radius * math.cos(heading)
        
        # Starting angle from center to initial point
        start_angle = math.atan2(y - center_y, x - center_x)
        
        # Adjust step size
        step_size = min(0.1, abs(angle) / 10)
        if step_size < 0.01:
            step_size = 0.01
        
        points = []
        
        # Angle signs
        angle_sign = 1 if is_left else -1
        if direction == DrivingDirection.BACKWARD:
            angle_sign *= -1
        
        # Number of steps
        num_steps = max(2, int(abs(angle) / step_size))
        
        # Generate points
        for i in range(num_steps + 1):
            t = i / num_steps * abs(angle)
            current_angle = start_angle + angle_sign * t
            
            point_x = center_x + radius * math.cos(current_angle)
            point_y = center_y + radius * math.sin(current_angle)
            
            points.append(Point2D(point_x, point_y))
        
        return points