"""
Base vehicle definitions for the mining dispatch system.

Defines the abstract vehicle interface and common vehicle behavior.
All specific vehicle types should inherit from these base classes.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from abc import ABC, abstractmethod
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from datetime import datetime
import math

from utils.geo.coordinates import Point2D
from utils.math.vectors import Vector2D
from utils.io.serialization import Serializable


class Vehicle(ABC, Serializable):
    """
    Abstract base class for all vehicle types.
    
    Defines the common interface and behavior that all vehicles must implement.
    """
    
    def __init__(self, vehicle_id: Optional[str] = None):
        """
        Initialize a vehicle.
        
        Args:
            vehicle_id: Unique identifier for the vehicle (generated if None)
        """
        self.vehicle_id = vehicle_id or str(uuid.uuid4())
        self.creation_time = datetime.now()
        self._current_location = Point2D(0, 0)
        self._heading = 0.0  # Heading in radians (0 is east, π/2 is north)
        self._current_path = []
        self._path_index = 0
    
    @property
    def current_location(self) -> Point2D:
        """Get the current location of the vehicle."""
        return self._current_location
    
    @current_location.setter
    def current_location(self, location: Union[Point2D, Tuple[float, float]]) -> None:
        """
        Set the current location of the vehicle.
        
        Args:
            location: New location (Point2D or tuple)
        """
        if isinstance(location, tuple) and len(location) >= 2:
            self._current_location = Point2D(location[0], location[1])
        elif isinstance(location, Point2D):
            self._current_location = location
        else:
            raise ValueError(f"Invalid location format: {location}")
    
    @property
    def heading(self) -> float:
        """Get the current heading angle in radians."""
        return self._heading
    
    @heading.setter
    def heading(self, angle: float) -> None:
        """
        Set the heading angle.
        
        Args:
            angle: Heading angle in radians
        """
        # Normalize angle to [0, 2π)
        self._heading = angle % (2 * math.pi)
    
    @property
    def heading_degrees(self) -> float:
        """Get the current heading angle in degrees."""
        return math.degrees(self._heading)
    
    @heading_degrees.setter
    def heading_degrees(self, angle: float) -> None:
        """
        Set the heading angle in degrees.
        
        Args:
            angle: Heading angle in degrees
        """
        self.heading = math.radians(angle)
    
    @property
    def current_path(self) -> List[Point2D]:
        """Get the current path being followed."""
        return self._current_path
    
    @current_path.setter
    def current_path(self, path: List[Union[Point2D, Tuple[float, float]]]) -> None:
        """
        Set the current path.
        
        Args:
            path: List of points forming the path
        """
        # Convert all points to Point2D
        self._current_path = []
        for point in path:
            if isinstance(point, tuple) and len(point) >= 2:
                self._current_path.append(Point2D(point[0], point[1]))
            elif isinstance(point, Point2D):
                self._current_path.append(point)
            else:
                raise ValueError(f"Invalid path point format: {point}")
        
        # Reset path index
        self._path_index = 0
    
    @property
    def path_index(self) -> int:
        """Get the current index in the path."""
        return self._path_index
    
    @path_index.setter
    def path_index(self, index: int) -> None:
        """
        Set the current index in the path.
        
        Args:
            index: Path index
            
        Raises:
            IndexError: If index is out of range
        """
        if not self._current_path:
            self._path_index = 0
        elif 0 <= index < len(self._current_path):
            self._path_index = index
        else:
            raise IndexError(f"Path index {index} out of range (0-{len(self._current_path)-1})")
    
    @property
    def remaining_path(self) -> List[Point2D]:
        """Get the remaining points in the current path."""
        if not self._current_path or self._path_index >= len(self._current_path):
            return []
        return self._current_path[self._path_index:]
    
    @property
    def next_waypoint(self) -> Optional[Point2D]:
        """Get the next waypoint in the path, or None if end of path."""
        if not self._current_path or self._path_index >= len(self._current_path) - 1:
            return None
        return self._current_path[self._path_index + 1]
    
    @property
    def is_at_path_end(self) -> bool:
        """Check if vehicle has reached the end of its path."""
        return (not self._current_path or 
                self._path_index >= len(self._current_path) - 1)
    
    @property
    @abstractmethod
    def state(self) -> str:
        """
        Get the current state of the vehicle.
        
        Returns:
            str: State identifier
        """
        pass
    
    @abstractmethod
    def update_position(self) -> None:
        """
        Update the vehicle's position based on its movement model.
        
        This method represents one simulation step for the vehicle.
        """
        pass
    
    @abstractmethod
    def calculate_path_to(self, destination: Point2D) -> List[Point2D]:
        """
        Calculate a path from current location to a destination.
        
        Args:
            destination: Target location
            
        Returns:
            List[Point2D]: Calculated path
        """
        pass
    
    @abstractmethod
    def assign_path(self, path: List[Union[Point2D, Tuple[float, float]]]) -> None:
        """
        Assign a new path to the vehicle.
        
        Args:
            path: List of points forming the path
        """
        pass
    
    @abstractmethod
    def assign_task(self, task) -> None:
        """
        Assign a task to the vehicle.
        
        Args:
            task: Task to assign
        """
        pass
    
    def move_to_next_waypoint(self) -> bool:
        """
        Move to the next waypoint in the path.
        
        Returns:
            bool: True if moved successfully, False if at path end
        """
        if self.is_at_path_end:
            return False
        
        self._path_index += 1
        if self._path_index < len(self._current_path):
            self._current_location = self._current_path[self._path_index]
            
            # Update heading if there's a next point
            if self._path_index + 1 < len(self._current_path):
                next_point = self._current_path[self._path_index + 1]
                dx = next_point.x - self._current_location.x
                dy = next_point.y - self._current_location.y
                self._heading = math.atan2(dy, dx)
            
            return True
        else:
            # Reached end of path
            return False
    
    def distance_to(self, point: Union[Point2D, Tuple[float, float]]) -> float:
        """
        Calculate distance from vehicle's current location to a point.
        
        Args:
            point: Target point
            
        Returns:
            float: Distance to point
        """
        target = point
        if isinstance(point, tuple) and len(point) >= 2:
            target = Point2D(point[0], point[1])
        
        return self._current_location.distance_to(target)
    
    def direction_to(self, point: Union[Point2D, Tuple[float, float]]) -> Vector2D:
        """
        Calculate direction vector from vehicle's current location to a point.
        
        Args:
            point: Target point
            
        Returns:
            Vector2D: Direction vector (normalized)
        """
        target = point
        if isinstance(point, tuple) and len(point) >= 2:
            target = Point2D(point[0], point[1])
        
        direction = Vector2D(target.x - self._current_location.x, 
                           target.y - self._current_location.y)
        
        # Return normalized vector (or zero vector if distance is zero)
        if direction.magnitude > 0:
            return direction.normalized
        else:
            return direction
    
    def angle_to(self, point: Union[Point2D, Tuple[float, float]]) -> float:
        """
        Calculate angle between vehicle's heading and direction to point.
        
        Args:
            point: Target point
            
        Returns:
            float: Angle difference in radians
        """
        target = point
        if isinstance(point, tuple) and len(point) >= 2:
            target = Point2D(point[0], point[1])
        
        direction = Vector2D(target.x - self._current_location.x, 
                           target.y - self._current_location.y)
        
        if direction.magnitude == 0:
            return 0
        
        # Calculate angle of direction vector
        angle = math.atan2(direction.y, direction.x)
        
        # Calculate difference from current heading
        diff = angle - self._heading
        
        # Normalize to [-π, π]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
            
        return diff
    
    def reset_path(self) -> None:
        """Clear the current path."""
        self._current_path = []
        self._path_index = 0
    
    def __repr__(self) -> str:
        """String representation of the vehicle."""
        return f"{self.__class__.__name__}(id={self.vehicle_id}, pos={self._current_location})"


class ConstrainedVehicle(Vehicle):
    """
    Base class for vehicles with physical constraints.
    
    Extends the base Vehicle class with physical properties and constraints
    like dimensions, turning radius, and speed limits.
    """
    
    def __init__(self, vehicle_id: Optional[str] = None, 
                 max_speed: float = 5.0,
                 max_acceleration: float = 2.0,
                 max_deceleration: float = 4.0,
                 turning_radius: float = 10.0,
                 length: float = 5.0,
                 width: float = 2.0):
        """
        Initialize a constrained vehicle.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            max_speed: Maximum speed in m/s
            max_acceleration: Maximum acceleration in m/s²
            max_deceleration: Maximum deceleration in m/s²
            turning_radius: Minimum turning radius in meters
            length: Vehicle length in meters
            width: Vehicle width in meters
        """
        super().__init__(vehicle_id)
        
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.turning_radius = turning_radius
        self.length = length
        self.width = width
        
        self._current_speed = 0.0
        self._target_speed = 0.0
    
    @property
    def current_speed(self) -> float:
        """Get the current speed of the vehicle."""
        return self._current_speed
    
    @current_speed.setter
    def current_speed(self, speed: float) -> None:
        """
        Set the current speed of the vehicle.
        
        Args:
            speed: Speed in m/s (clamped to max_speed)
        """
        self._current_speed = max(0, min(speed, self.max_speed))
    
    @property
    def target_speed(self) -> float:
        """Get the target speed of the vehicle."""
        return self._target_speed
    
    @target_speed.setter
    def target_speed(self, speed: float) -> None:
        """
        Set the target speed of the vehicle.
        
        Args:
            speed: Target speed in m/s (clamped to max_speed)
        """
        self._target_speed = max(0, min(speed, self.max_speed))
    
    def update_speed(self, dt: float) -> None:
        """
        Update the vehicle's speed toward target speed.
        
        Args:
            dt: Time step in seconds
        """
        if self._current_speed < self._target_speed:
            # Accelerate
            acceleration = min(self.max_acceleration, 
                              (self._target_speed - self._current_speed) / dt)
            new_speed = self._current_speed + acceleration * dt
            self._current_speed = min(new_speed, self._target_speed)
        elif self._current_speed > self._target_speed:
            # Decelerate
            deceleration = min(self.max_deceleration, 
                              (self._current_speed - self._target_speed) / dt)
            new_speed = self._current_speed - deceleration * dt
            self._current_speed = max(new_speed, self._target_speed)
    
    def get_stopping_distance(self) -> float:
        """
        Calculate the stopping distance at current speed.
        
        Returns:
            float: Stopping distance in meters
        """
        # s = v²/(2*a) where s=distance, v=velocity, a=deceleration
        return (self._current_speed ** 2) / (2 * self.max_deceleration)
    
    def get_turning_distance(self, angle: float) -> float:
        """
        Calculate the distance required to turn by a given angle.
        
        Args:
            angle: Angle in radians
            
        Returns:
            float: Distance in meters
        """
        # Arc length = radius * angle
        return self.turning_radius * abs(angle)
    
    def get_bounding_points(self) -> List[Point2D]:
        """
        Get the corner points of the vehicle's bounding box.
        
        Returns:
            List[Point2D]: Four corner points
        """
        half_length = self.length / 2
        half_width = self.width / 2
        
        # Calculate corner offsets from center
        cos_heading = math.cos(self._heading)
        sin_heading = math.sin(self._heading)
        
        # Define corners relative to center and heading
        corners = [
            (half_length * cos_heading - half_width * sin_heading,
             half_length * sin_heading + half_width * cos_heading),
            (half_length * cos_heading + half_width * sin_heading,
             half_length * sin_heading - half_width * cos_heading),
            (-half_length * cos_heading + half_width * sin_heading,
             -half_length * sin_heading - half_width * cos_heading),
            (-half_length * cos_heading - half_width * sin_heading,
             -half_length * sin_heading + half_width * cos_heading)
        ]
        
        # Convert to absolute positions
        return [
            Point2D(self._current_location.x + dx, self._current_location.y + dy)
            for dx, dy in corners
        ]
    
    def estimate_arrival_time(self, destination: Point2D) -> float:
        """
        Estimate time to reach a destination at maximum speed.
        
        Args:
            destination: Target location
            
        Returns:
            float: Estimated time in seconds
        """
        distance = self.distance_to(destination)
        
        if self.max_speed <= 0:
            return float('inf')
            
        # Simple estimate assuming constant max speed
        return distance / self.max_speed