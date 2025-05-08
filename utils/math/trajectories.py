"""
Trajectory generation and manipulation utilities for the mining dispatch system.

Provides functions for generating, smoothing, and analyzing vehicle trajectories.
"""

import math
import numpy as np
from typing import List, Tuple, Union, Optional, Callable
from enum import Enum, auto
from scipy.interpolate import splprep, splev

from utils.geo.coordinates import Point2D, Point3D
from utils.math.vectors import Vector2D, Vector3D


class PathSmoothingMethod(Enum):
    """Enumeration of available path smoothing methods."""
    BEZIER = auto()
    SPLINE = auto()
    MOVING_AVERAGE = auto()
    

class PathType(Enum):
    """Enumeration of path/trajectory types."""
    STRAIGHT_LINE = auto()
    BEZIER_CURVE = auto()
    SPLINE = auto()
    REEDS_SHEPP = auto()
    DUBINS = auto()
    HYBRID = auto()


class Path:
    """
    Path representation with points and optional metadata.
    
    A path is a sequence of points representing a route or trajectory.
    """
    
    def __init__(self, points: List[Union[Point2D, Tuple[float, float]]], 
                path_type: PathType = PathType.STRAIGHT_LINE,
                metadata: dict = None):
        """
        Initialize a path.
        
        Args:
            points: List of points in the path
            path_type: Type of path (default: STRAIGHT_LINE)
            metadata: Optional metadata dictionary
        """
        # Convert all points to Point2D
        self.points = [p if isinstance(p, Point2D) else Point2D(*p) for p in points]
        self.path_type = path_type
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        """Number of points in the path."""
        return len(self.points)
    
    def __getitem__(self, index) -> Point2D:
        """Get point at index."""
        return self.points[index]
    
    def __iter__(self):
        """Iterate over points."""
        return iter(self.points)
    
    def __repr__(self) -> str:
        """String representation of the path."""
        return f"Path({len(self.points)} points, type={self.path_type.name})"
    
    @property
    def length(self) -> float:
        """
        Calculate the total length of the path.
        
        Returns:
            float: Total path length
        """
        if len(self.points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(self.points) - 1):
            total_length += self.points[i].distance_to(self.points[i+1])
        
        return total_length
    
    @property
    def start_point(self) -> Point2D:
        """Get the first point of the path."""
        if not self.points:
            raise ValueError("Path is empty")
        return self.points[0]
    
    @property
    def end_point(self) -> Point2D:
        """Get the last point of the path."""
        if not self.points:
            raise ValueError("Path is empty")
        return self.points[-1]
    
    def append(self, point: Union[Point2D, Tuple[float, float]]) -> None:
        """
        Append a point to the path.
        
        Args:
            point: Point to append
        """
        if not isinstance(point, Point2D):
            point = Point2D(*point)
        self.points.append(point)
    
    def extend(self, points: List[Union[Point2D, Tuple[float, float]]]) -> None:
        """
        Extend the path with a list of points.
        
        Args:
            points: Points to append
        """
        for point in points:
            self.append(point)
    
    def reverse(self) -> 'Path':
        """
        Create a new path with points in reverse order.
        
        Returns:
            Path: New path with reversed order
        """
        return Path(self.points[::-1], self.path_type, self.metadata.copy())
    
    def subsample(self, step: int = 1) -> 'Path':
        """
        Create a subsampled version of the path.
        
        Args:
            step: Step size for subsampling
            
        Returns:
            Path: Subsampled path
        """
        if step <= 0:
            raise ValueError("Step must be positive")
        
        return Path(self.points[::step], self.path_type, self.metadata.copy())
    
    def resample_by_distance(self, distance: float) -> 'Path':
        """
        Resample path with points at regular distance intervals.
        
        Args:
            distance: Distance between points
            
        Returns:
            Path: Resampled path
            
        Raises:
            ValueError: If distance is not positive
        """
        if distance <= 0:
            raise ValueError("Distance must be positive")
        
        if len(self.points) < 2:
            return Path(self.points.copy(), self.path_type, self.metadata.copy())
        
        # Create new point list starting with the first point
        new_points = [self.points[0]]
        
        # Current distance from last sampled point
        current_distance = 0
        
        for i in range(1, len(self.points)):
            # Calculate segment length
            segment_length = self.points[i-1].distance_to(self.points[i])
            
            # Process this segment
            remaining_segment = segment_length
            start_point = self.points[i-1]
            end_point = self.points[i]
            
            # Vector from start to end of segment
            segment_vector = Vector2D.from_points(start_point, end_point)
            segment_dir = segment_vector.normalized
            
            while remaining_segment > 0:
                # Distance needed to reach next sample point
                distance_needed = distance - current_distance
                
                if distance_needed <= remaining_segment:
                    # We can place a new point within this segment
                    new_point = Point2D(
                        start_point.x + segment_dir.x * distance_needed,
                        start_point.y + segment_dir.y * distance_needed
                    )
                    new_points.append(new_point)
                    
                    # Update state
                    remaining_segment -= distance_needed
                    start_point = new_point
                    current_distance = 0
                else:
                    # Can't reach next sample point in this segment
                    current_distance += remaining_segment
                    break
        
        # Always include the last point
        if not new_points[-1] == self.points[-1]:
            new_points.append(self.points[-1])
        
        return Path(new_points, self.path_type, self.metadata.copy())
    
    def smooth(self, method: PathSmoothingMethod = PathSmoothingMethod.SPLINE, 
              smoothing_factor: float = 0.5, num_points: int = 100) -> 'Path':
        """
        Create a smoothed version of the path.
        
        Args:
            method: Smoothing method to use
            smoothing_factor: Smoothing intensity (0.0 to 1.0)
            num_points: Number of points in the output path
            
        Returns:
            Path: Smoothed path
        """
        if len(self.points) < 3:
            return Path(self.points.copy(), self.path_type, self.metadata.copy())
        
        if method == PathSmoothingMethod.BEZIER:
            return self._smooth_bezier(num_points)
        elif method == PathSmoothingMethod.SPLINE:
            return self._smooth_spline(smoothing_factor, num_points)
        elif method == PathSmoothingMethod.MOVING_AVERAGE:
            return self._smooth_moving_average(int(smoothing_factor * 10))
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    def _smooth_bezier(self, num_points: int) -> 'Path':
        """Smooth path using Bezier curves."""
        # Extract x and y coordinates
        x = [p.x for p in self.points]
        y = [p.y for p in self.points]
        
        # Create parameter t (0 to 1)
        t = np.linspace(0, 1, num_points)
        
        # Create Bezier curve using De Casteljau's algorithm
        n = len(self.points) - 1
        
        # Initialize with original points
        points_x = np.copy(x)
        points_y = np.copy(y)
        
        # Output points
        result_x = np.zeros(num_points)
        result_y = np.zeros(num_points)
        
        # For each parameter value
        for j, tj in enumerate(t):
            # Temporary arrays for this iteration
            temp_x = np.copy(points_x)
            temp_y = np.copy(points_y)
            
            # Apply De Casteljau's algorithm
            for i in range(1, n + 1):
                for k in range(n + 1 - i):
                    temp_x[k] = (1 - tj) * temp_x[k] + tj * temp_x[k + 1]
                    temp_y[k] = (1 - tj) * temp_y[k] + tj * temp_y[k + 1]
            
            # First element is the point on the curve
            result_x[j] = temp_x[0]
            result_y[j] = temp_y[0]
        
        # Create path from Bezier curve points
        bezier_points = [Point2D(result_x[i], result_y[i]) for i in range(num_points)]
        return Path(bezier_points, PathType.BEZIER_CURVE, self.metadata.copy())
    
    def _smooth_spline(self, smoothing_factor: float, num_points: int) -> 'Path':
        """Smooth path using B-spline."""
        # Extract x and y coordinates
        x = [p.x for p in self.points]
        y = [p.y for p in self.points]
        
        # Calculate spline parameters
        k = min(3, len(self.points) - 1)  # Spline degree
        s = smoothing_factor * len(self.points)  # Smoothing factor
        
        # Create the spline
        tck, u = splprep([x, y], s=s, k=k)
        
        # Evaluate the spline at evenly spaced points
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        
        # Create path from spline points
        spline_points = [Point2D(x_new[i], y_new[i]) for i in range(num_points)]
        return Path(spline_points, PathType.SPLINE, self.metadata.copy())
    
    def _smooth_moving_average(self, window_size: int) -> 'Path':
        """Smooth path using moving average."""
        if window_size < 2:
            return Path(self.points.copy(), self.path_type, self.metadata.copy())
        
        # Adjust window size if needed
        window_size = min(window_size, len(self.points))
        
        # Create new smoothed points
        smoothed_points = []
        
        # Always keep the first and last points unchanged
        smoothed_points.append(self.points[0])
        
        # Apply moving average to middle points
        half_window = window_size // 2
        for i in range(1, len(self.points) - 1):
            # Calculate window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(len(self.points), i + half_window + 1)
            
            # Calculate average within window
            avg_x = sum(p.x for p in self.points[start_idx:end_idx]) / (end_idx - start_idx)
            avg_y = sum(p.y for p in self.points[start_idx:end_idx]) / (end_idx - start_idx)
            
            smoothed_points.append(Point2D(avg_x, avg_y))
        
        # Add last point
        if len(self.points) > 1:
            smoothed_points.append(self.points[-1])
        
        return Path(smoothed_points, self.path_type, self.metadata.copy())


class Trajectory(Path):
    """
    Extended path with time information for each point.
    
    A trajectory is a path with timestamps, representing the planned
    position of a vehicle over time.
    """
    
    def __init__(self, points: List[Union[Point2D, Tuple[float, float]]],
                timestamps: List[float] = None,
                velocities: List[float] = None,
                path_type: PathType = PathType.STRAIGHT_LINE,
                metadata: dict = None):
        """
        Initialize a trajectory.
        
        Args:
            points: List of points in the trajectory
            timestamps: List of timestamps for each point (seconds)
            velocities: List of velocities at each point (m/s)
            path_type: Type of path
            metadata: Optional metadata dictionary
        """
        super().__init__(points, path_type, metadata)
        
        # Initialize timestamps and velocities
        if timestamps is None:
            self.timestamps = [0.0] * len(points)
        else:
            if len(timestamps) != len(points):
                raise ValueError("Number of timestamps must match number of points")
            self.timestamps = list(timestamps)
        
        if velocities is None:
            self.velocities = [0.0] * len(points)
        else:
            if len(velocities) != len(points):
                raise ValueError("Number of velocities must match number of points")
            self.velocities = list(velocities)
    
    def __repr__(self) -> str:
        """String representation of the trajectory."""
        duration = self.duration if len(self.timestamps) > 1 else 0
        return f"Trajectory({len(self.points)} points, duration={duration:.2f}s, type={self.path_type.name})"
    
    @property
    def duration(self) -> float:
        """
        Calculate the total duration of the trajectory.
        
        Returns:
            float: Duration in seconds
        """
        if not self.timestamps:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]
    
    @property
    def average_velocity(self) -> float:
        """
        Calculate the average velocity over the trajectory.
        
        Returns:
            float: Average velocity in m/s
        """
        if self.duration == 0 or len(self.points) < 2:
            return 0.0
        return self.length / self.duration
    
    def position_at_time(self, time: float) -> Optional[Point2D]:
        """
        Interpolate position at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Optional[Point2D]: Interpolated position, or None if time is out of range
        """
        if not self.timestamps or time < self.timestamps[0] or time > self.timestamps[-1]:
            return None
        
        # Find bracketing timestamps
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] <= time <= self.timestamps[i + 1]:
                # Linear interpolation
                t1, t2 = self.timestamps[i], self.timestamps[i + 1]
                p1, p2 = self.points[i], self.points[i + 1]
                
                # Calculate interpolation parameter (0 to 1)
                alpha = (time - t1) / (t2 - t1) if t2 > t1 else 0.0
                
                # Interpolate
                x = p1.x + alpha * (p2.x - p1.x)
                y = p1.y + alpha * (p2.y - p1.y)
                
                return Point2D(x, y)
        
        # Should not reach here, but just in case
        return None
    
    def velocity_at_time(self, time: float) -> Optional[float]:
        """
        Interpolate velocity at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Optional[float]: Interpolated velocity, or None if time is out of range
        """
        if not self.timestamps or time < self.timestamps[0] or time > self.timestamps[-1]:
            return None
        
        # Find bracketing timestamps
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] <= time <= self.timestamps[i + 1]:
                # Linear interpolation
                t1, t2 = self.timestamps[i], self.timestamps[i + 1]
                v1, v2 = self.velocities[i], self.velocities[i + 1]
                
                # Calculate interpolation parameter (0 to 1)
                alpha = (time - t1) / (t2 - t1) if t2 > t1 else 0.0
                
                # Interpolate
                return v1 + alpha * (v2 - v1)
        
        # Should not reach here, but just in case
        return None
    
    def heading_at_time(self, time: float) -> Optional[float]:
        """
        Calculate heading (angle) at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Optional[float]: Heading in radians, or None if time is out of range
        """
        if not self.timestamps or time < self.timestamps[0] or time > self.timestamps[-1]:
            return None
        
        # Find bracketing timestamps
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] <= time <= self.timestamps[i + 1]:
                # Get positions
                p1, p2 = self.points[i], self.points[i + 1]
                
                # Calculate heading
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                
                # Handle zero movement
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    # Try next segment if available
                    if i < len(self.timestamps) - 2:
                        p3 = self.points[i + 2]
                        dx = p3.x - p1.x
                        dy = p3.y - p1.y
                    else:
                        # Use previous segment if available
                        if i > 0:
                            p0 = self.points[i - 1]
                            dx = p1.x - p0.x
                            dy = p1.y - p0.y
                
                return math.atan2(dy, dx)
        
        # Should not reach here, but just in case
        return None
    
    @classmethod
    def from_path_with_velocity(cls, path: Path, velocity: float) -> 'Trajectory':
        """
        Create a trajectory from a path and constant velocity.
        
        Args:
            path: Input path
            velocity: Constant velocity (m/s)
            
        Returns:
            Trajectory: Trajectory with timestamps
            
        Raises:
            ValueError: If velocity is not positive
        """
        if velocity <= 0:
            raise ValueError("Velocity must be positive")
        
        if len(path.points) < 2:
            timestamps = [0.0] * len(path.points)
            velocities = [velocity] * len(path.points)
            return cls(path.points.copy(), timestamps, velocities, path.path_type, path.metadata.copy())
        
        # Calculate timestamps based on distance and velocity
        timestamps = [0.0]
        total_distance = 0.0
        
        for i in range(1, len(path.points)):
            distance = path.points[i-1].distance_to(path.points[i])
            total_distance += distance
            time = total_distance / velocity
            timestamps.append(time)
        
        # Use constant velocity for all points
        velocities = [velocity] * len(path.points)
        
        return cls(path.points.copy(), timestamps, velocities, path.path_type, path.metadata.copy())
    
    @classmethod
    def from_path_with_acceleration(cls, path: Path, 
                                   initial_velocity: float,
                                   acceleration: float,
                                   max_velocity: float = float('inf')) -> 'Trajectory':
        """
        Create a trajectory from a path with constant acceleration.
        
        Args:
            path: Input path
            initial_velocity: Starting velocity (m/s)
            acceleration: Constant acceleration (m/s²)
            max_velocity: Maximum velocity (m/s)
            
        Returns:
            Trajectory: Trajectory with timestamps and velocities
            
        Raises:
            ValueError: If initial_velocity is negative or max_velocity is not positive
        """
        if initial_velocity < 0:
            raise ValueError("Initial velocity cannot be negative")
        if max_velocity <= 0:
            raise ValueError("Maximum velocity must be positive")
        
        if len(path.points) < 2:
            timestamps = [0.0] * len(path.points)
            velocities = [initial_velocity] * len(path.points)
            return cls(path.points.copy(), timestamps, velocities, path.path_type, path.metadata.copy())
        
        # Calculate timestamps and velocities
        timestamps = [0.0]
        velocities = [initial_velocity]
        
        current_time = 0.0
        current_velocity = initial_velocity
        
        for i in range(1, len(path.points)):
            # Distance for this segment
            distance = path.points[i-1].distance_to(path.points[i])
            
            if abs(acceleration) < 1e-6:
                # Constant velocity case
                current_velocity = min(current_velocity, max_velocity)
                segment_time = distance / current_velocity
                current_time += segment_time
                timestamps.append(current_time)
                velocities.append(current_velocity)
            else:
                # Accelerating case
                # Use kinematic equations: v² = v₀² + 2a·s
                v0_squared = current_velocity**2
                v_squared = v0_squared + 2 * acceleration * distance
                
                # Handle deceleration to stop
                if v_squared <= 0 and acceleration < 0:
                    # Calculate distance to stop
                    stop_distance = v0_squared / (2 * abs(acceleration))
                    
                    if stop_distance < distance:
                        # We'll stop before reaching the next point
                        # Time to stop
                        stop_time = current_velocity / abs(acceleration)
                        current_time += stop_time
                        
                        # Remaining distance at zero velocity
                        remaining_distance = distance - stop_distance
                        
                        # We need to start moving again (assume we reverse acceleration)
                        # Time to cover remaining distance
                        t_remaining = math.sqrt(2 * remaining_distance / abs(acceleration))
                        current_time += t_remaining
                        
                        # Final velocity
                        current_velocity = abs(acceleration) * t_remaining
                    else:
                        # We'll still be moving at the end
                        # Final velocity
                        current_velocity = math.sqrt(max(0, v_squared))
                        
                        # Time for this segment
                        segment_time = (current_velocity - velocities[-1]) / acceleration
                        current_time += segment_time
                else:
                    # Normal case
                    # Final velocity (capped by max_velocity)
                    current_velocity = min(math.sqrt(max(0, v_squared)), max_velocity)
                    
                    # Time for this segment
                    if abs(current_velocity - velocities[-1]) < 1e-6:
                        # Velocity didn't change (likely at max_velocity)
                        segment_time = distance / current_velocity
                    else:
                        # v = v₀ + a·t, so t = (v - v₀) / a
                        segment_time = (current_velocity - velocities[-1]) / acceleration
                    
                    current_time += segment_time
                
                timestamps.append(current_time)
                velocities.append(current_velocity)
        
        return cls(path.points.copy(), timestamps, velocities, path.path_type, path.metadata.copy())


def generate_straight_line_path(start: Point2D, end: Point2D, num_points: int = 10) -> Path:
    """
    Generate a straight line path between two points.
    
    Args:
        start: Start point
        end: End point
        num_points: Number of points to generate
        
    Returns:
        Path: Straight line path
    """
    if num_points < 2:
        raise ValueError("Number of points must be at least 2")
    
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = start.x + t * (end.x - start.x)
        y = start.y + t * (end.y - start.y)
        points.append(Point2D(x, y))
    
    return Path(points, PathType.STRAIGHT_LINE)


def generate_circular_arc_path(center: Point2D, radius: float, 
                              start_angle: float, end_angle: float, 
                              num_points: int = 36) -> Path:
    """
    Generate a circular arc path.
    
    Args:
        center: Center of the circle
        radius: Radius of the circle
        start_angle: Start angle in radians
        end_angle: End angle in radians
        num_points: Number of points to generate
        
    Returns:
        Path: Circular arc path
    """
    if num_points < 2:
        raise ValueError("Number of points must be at least 2")
    
    # Ensure positive angle difference
    while end_angle < start_angle:
        end_angle += 2 * math.pi
    
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        angle = start_angle + t * (end_angle - start_angle)
        
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        
        points.append(Point2D(x, y))
    
    return Path(points)


def generate_bezier_path(control_points: List[Point2D], num_points: int = 100) -> Path:
    """
    Generate a Bezier curve path from control points.
    
    Args:
        control_points: List of control points
        num_points: Number of points to generate
        
    Returns:
        Path: Bezier curve path
    """
    if len(control_points) < 2:
        raise ValueError("At least 2 control points are required")
    
    if num_points < 2:
        raise ValueError("Number of output points must be at least 2")
    
    # Extract x and y coordinates
    x = [p.x for p in control_points]
    y = [p.y for p in control_points]
    
    # Parameter t from 0 to 1
    t_values = np.linspace(0, 1, num_points)
    
    # Compute Bezier curve points
    n = len(control_points) - 1  # Degree of the curve
    
    # Generate binomial coefficients
    coeffs = [math.comb(n, i) for i in range(n + 1)]
    
    # Generate points
    points = []
    for t in t_values:
        px = 0.0
        py = 0.0
        
        for i in range(n + 1):
            # Bernstein polynomial
            b = coeffs[i] * (t ** i) * ((1 - t) ** (n - i))
            
            # Add weighted contribution
            px += b * x[i]
            py += b * y[i]
        
        points.append(Point2D(px, py))
    
    return Path(points, PathType.BEZIER_CURVE)


def generate_s_curve_path(start: Point2D, end: Point2D, 
                         curve_height: float, 
                         num_points: int = 100) -> Path:
    """
    Generate an S-curve path between two points.
    
    Args:
        start: Start point
        end: End point
        curve_height: Maximum height of the curve from the straight line
        num_points: Number of points to generate
        
    Returns:
        Path: S-curve path
    """
    if num_points < 2:
        raise ValueError("Number of points must be at least 2")
    
    # Vector from start to end
    direction = Vector2D.from_points(start, end)
    
    # Normal vector (perpendicular to direction)
    if math.isclose(direction.magnitude, 0.0):
        # Default to up direction if start and end are the same
        normal = Vector2D(0, 1)
    else:
        normal = direction.perpendicular().normalized
    
    # Middle point
    middle_x = (start.x + end.x) / 2
    middle_y = (start.y + end.y) / 2
    
    # Control points for cubic Bezier
    control1 = Point2D(
        (2 * start.x + end.x) / 3 + normal.x * curve_height,
        (2 * start.y + end.y) / 3 + normal.y * curve_height
    )
    
    control2 = Point2D(
        (start.x + 2 * end.x) / 3 - normal.x * curve_height,
        (start.y + 2 * end.y) / 3 - normal.y * curve_height
    )
    
    # Generate Bezier with 4 control points
    return generate_bezier_path([start, control1, control2, end], num_points)


def resample_path_by_curvature(path: Path, base_distance: float = 1.0,
                             curvature_factor: float = 0.5,
                             min_points: int = 10) -> Path:
    """
    Resample a path with density proportional to local curvature.
    
    Args:
        path: Input path
        base_distance: Base distance between points
        curvature_factor: Factor to adjust spacing based on curvature
        min_points: Minimum number of points in the output
        
    Returns:
        Path: Resampled path
    """
    if len(path.points) < 3:
        return Path(path.points.copy(), path.path_type, path.metadata.copy())
    
    # Calculate approximate curvature at each point
    curvatures = []
    
    # Add first point
    # Use forward difference for first point
    v1 = Vector2D.from_points(path.points[0], path.points[1])
    v2 = Vector2D.from_points(path.points[1], path.points[2])
    angle_diff = abs(signed_angle_2d(v1, v2))
    curvatures.append(angle_diff)
    
    # Middle points
    for i in range(1, len(path.points) - 1):
        prev_point = path.points[i-1]
        curr_point = path.points[i]
        next_point = path.points[i+1]
        
        # Vectors from current point to neighbors
        v1 = Vector2D.from_points(curr_point, prev_point)
        v2 = Vector2D.from_points(curr_point, next_point)
        
        # Angle between vectors as measure of curvature
        angle_diff = abs(signed_angle_2d(v1, v2))
        curvatures.append(angle_diff)
    
    # Add last point
    # Use backward difference for last point
    v1 = Vector2D.from_points(path.points[-3], path.points[-2])
    v2 = Vector2D.from_points(path.points[-2], path.points[-1])
    angle_diff = abs(signed_angle_2d(v1, v2))
    curvatures.append(angle_diff)
    
    # Normalize curvatures
    max_curvature = max(curvatures)
    if max_curvature > 0:
        normalized_curvatures = [c / max_curvature for c in curvatures]
    else:
        normalized_curvatures = [0.0] * len(curvatures)
    
    # Generate new points
    new_points = [path.points[0]]  # Start with first point
    
    for i in range(len(path.points) - 1):
        start_point = path.points[i]
        end_point = path.points[i+1]
        segment_length = start_point.distance_to(end_point)
        
        # Calculate locally adjusted spacing
        avg_curvature = (normalized_curvatures[i] + normalized_curvatures[i+1]) / 2
        local_spacing = base_distance / (1 + curvature_factor * avg_curvature)
        
        # Calculate number of intermediate points
        num_intermediate = max(0, int(segment_length / local_spacing) - 1)
        
        # Add intermediate points
        for j in range(1, num_intermediate + 1):
            t = j / (num_intermediate + 1)
            x = start_point.x + t * (end_point.x - start_point.x)
            y = start_point.y + t * (end_point.y - start_point.y)
            new_points.append(Point2D(x, y))
        
        # Add endpoint (except for the last segment, where it's already the last point)
        if i < len(path.points) - 2:
            new_points.append(end_point)
    
    # Always add the last point
    new_points.append(path.points[-1])
    
    # Ensure minimum number of points
    if len(new_points) < min_points:
        # Fall back to uniform resampling
        return path.resample_by_distance(path.length / (min_points - 1))
    
    return Path(new_points, path.path_type, path.metadata.copy())


def signed_angle_2d(v1: Vector2D, v2: Vector2D) -> float:
    """
    Calculate the signed angle from v1 to v2.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        float: Signed angle in radians (-π to π)
    """
    # Normalize vectors
    if math.isclose(v1.magnitude, 0.0) or math.isclose(v2.magnitude, 0.0):
        return 0.0
    
    v1_norm = v1.normalized
    v2_norm = v2.normalized
    
    # Calculate cross product (z component)
    cross_z = v1_norm.cross_scalar(v2_norm)
    
    # Calculate dot product
    dot = v1_norm.dot(v2_norm)
    
    # Calculate signed angle
    angle = math.atan2(cross_z, dot)
    
    return angle


def calculate_path_curvature(path: Path) -> List[float]:
    """
    Calculate the curvature at each point in the path.
    
    Args:
        path: Input path
        
    Returns:
        List[float]: Curvature at each point
    """
    if len(path.points) < 3:
        return [0.0] * len(path.points)
    
    curvatures = []
    
    # Use forward difference for first point
    v1 = Vector2D.from_points(path.points[0], path.points[1])
    v2 = Vector2D.from_points(path.points[1], path.points[2])
    angle = abs(signed_angle_2d(v1, v2))
    dist = path.points[0].distance_to(path.points[1])
    curvature = angle / dist if dist > 0 else 0.0
    curvatures.append(curvature)
    
    # Middle points
    for i in range(1, len(path.points) - 1):
        prev_point = path.points[i-1]
        curr_point = path.points[i]
        next_point = path.points[i+1]
        
        # Approximate curvature using Menger curvature (circumcircle of three points)
        a = prev_point.distance_to(curr_point)
        b = curr_point.distance_to(next_point)
        c = next_point.distance_to(prev_point)
        
        # Semi-perimeter
        s = (a + b + c) / 2
        
        # Area using Heron's formula
        try:
            area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        except ValueError:
            # Degenerate case
            area = 0
        
        # Curvature = 4 * area / (a * b * c)
        if a > 0 and b > 0 and c > 0:
            curvature = 4 * area / (a * b * c)
        else:
            curvature = 0.0
            
        curvatures.append(curvature)
    
    # Use backward difference for last point
    v1 = Vector2D.from_points(path.points[-3], path.points[-2])
    v2 = Vector2D.from_points(path.points[-2], path.points[-1])
    angle = abs(signed_angle_2d(v1, v2))
    dist = path.points[-2].distance_to(path.points[-1])
    curvature = angle / dist if dist > 0 else 0.0
    curvatures.append(curvature)
    
    return curvatures


def merge_paths(paths: List[Path], connect: bool = True) -> Path:
    """
    Merge multiple paths into a single path.
    
    Args:
        paths: List of paths to merge
        connect: If True, add connecting segments between paths
        
    Returns:
        Path: Merged path
    """
    if not paths:
        return Path([])
    
    merged_points = []
    
    for i, path in enumerate(paths):
        if i == 0 or not connect:
            # Add all points from first path or if not connecting
            merged_points.extend(path.points)
        else:
            # For subsequent paths, add a connecting segment if needed
            prev_end = paths[i-1].end_point
            curr_start = path.start_point
            
            if prev_end != curr_start:
                # Add connecting segment
                merged_points.append(curr_start)
            
            # Add all remaining points
            merged_points.extend(path.points[1:])
    
    # Determine path type
    if all(p.path_type == paths[0].path_type for p in paths):
        path_type = paths[0].path_type
    else:
        path_type = PathType.HYBRID
    
    # Merge metadata from all paths
    merged_metadata = {}
    for path in paths:
        if path.metadata:
            merged_metadata.update(path.metadata)
    
    return Path(merged_points, path_type, merged_metadata)