"""
Vector mathematics utilities for the mining dispatch system.

Provides vector operations used in path planning, collision detection,
and other geometric calculations.
"""

import math
from typing import Tuple, List, Union, Optional
import numpy as np

from utils.geo.coordinates import Point2D, Point3D


class Vector2D:
    """
    2D vector representation with basic vector operations.
    
    Used for representing directions, velocities, and forces in 2D space.
    """
    
    def __init__(self, x: float, y: float):
        """
        Initialize a 2D vector.
        
        Args:
            x: x-component
            y: y-component
        """
        self.x = float(x)
        self.y = float(y)
    
    def __repr__(self) -> str:
        """String representation of the vector."""
        return f"Vector2D({self.x:.2f}, {self.y:.2f})"
    
    def __eq__(self, other) -> bool:
        """Check if two vectors are equal."""
        if not isinstance(other, Vector2D):
            return False
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)
    
    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        """Vector addition."""
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        """Vector subtraction."""
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2D':
        """Scalar multiplication."""
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector2D':
        """Scalar multiplication (right multiplication)."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector2D':
        """Scalar division."""
        if math.isclose(scalar, 0.0):
            raise ZeroDivisionError("Division by zero")
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def __neg__(self) -> 'Vector2D':
        """Vector negation."""
        return Vector2D(-self.x, -self.y)
    
    @property
    def magnitude(self) -> float:
        """Magnitude (length) of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    @property
    def magnitude_squared(self) -> float:
        """Squared magnitude of the vector (more efficient when only comparing lengths)."""
        return self.x * self.x + self.y * self.y
    
    @property
    def normalized(self) -> 'Vector2D':
        """
        Normalized vector (unit vector in the same direction).
        
        Returns:
            Vector2D: Normalized vector
            
        Raises:
            ValueError: If vector has zero magnitude
        """
        mag = self.magnitude
        if math.isclose(mag, 0.0):
            raise ValueError("Cannot normalize a zero vector")
        return self / mag
    
    def dot(self, other: 'Vector2D') -> float:
        """
        Dot product with another vector.
        
        Args:
            other: Another vector
            
        Returns:
            float: Dot product
        """
        return self.x * other.x + self.y * other.y
    
    def cross_scalar(self, other: 'Vector2D') -> float:
        """
        Cross product magnitude with another vector.
        
        For 2D vectors, the cross product is a scalar.
        
        Args:
            other: Another vector
            
        Returns:
            float: Cross product magnitude (z-component of 3D cross product)
        """
        return self.x * other.y - self.y * other.x
    
    def angle_with(self, other: 'Vector2D') -> float:
        """
        Calculate the angle between this vector and another.
        
        Args:
            other: Another vector
            
        Returns:
            float: Angle in radians
            
        Raises:
            ValueError: If either vector has zero magnitude
        """
        if math.isclose(self.magnitude, 0.0) or math.isclose(other.magnitude, 0.0):
            raise ValueError("Cannot compute angle with zero vector")
            
        # Use dot product formula: cos(theta) = (a·b) / (|a|·|b|)
        dot_product = self.dot(other)
        cos_angle = dot_product / (self.magnitude * other.magnitude)
        
        # Clamp to [-1, 1] to handle floating-point errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        return math.acos(cos_angle)
    
    def rotate(self, angle_rad: float) -> 'Vector2D':
        """
        Rotate the vector by a given angle.
        
        Args:
            angle_rad: Rotation angle in radians
            
        Returns:
            Vector2D: Rotated vector
        """
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        new_x = self.x * cos_angle - self.y * sin_angle
        new_y = self.x * sin_angle + self.y * cos_angle
        
        return Vector2D(new_x, new_y)
    
    def project_onto(self, other: 'Vector2D') -> 'Vector2D':
        """
        Project this vector onto another vector.
        
        Args:
            other: Vector to project onto
            
        Returns:
            Vector2D: Projection of this vector onto other
            
        Raises:
            ValueError: If other vector has zero magnitude
        """
        if math.isclose(other.magnitude, 0.0):
            raise ValueError("Cannot project onto a zero vector")
            
        dot = self.dot(other)
        other_mag_squared = other.magnitude_squared
        scalar = dot / other_mag_squared
        
        return other * scalar
    
    def perpendicular(self) -> 'Vector2D':
        """
        Get a vector perpendicular to this one (90 degrees counterclockwise).
        
        Returns:
            Vector2D: Perpendicular vector
        """
        return Vector2D(-self.y, self.x)
    
    def distance_to_line(self, line_start: Point2D, line_end: Point2D) -> float:
        """
        Calculate the shortest distance from the point represented by this vector
        to a line segment.
        
        Args:
            line_start: Start point of the line segment
            line_end: End point of the line segment
            
        Returns:
            float: Shortest distance to the line segment
        """
        # Convert to vectors
        point_vec = Vector2D(self.x, self.y)
        line_start_vec = Vector2D(line_start.x, line_start.y)
        line_end_vec = Vector2D(line_end.x, line_end.y)
        
        # Vector from line_start to line_end
        line_vec = line_end_vec - line_start_vec
        
        # Vector from line_start to point
        point_vec_rel = point_vec - line_start_vec
        
        # Check if line is a point
        if math.isclose(line_vec.magnitude, 0.0):
            return point_vec_rel.magnitude
        
        # Calculate projection parameter
        t = point_vec_rel.dot(line_vec) / line_vec.magnitude_squared
        
        if t < 0.0:
            # Closest to line_start
            return point_vec_rel.magnitude
        elif t > 1.0:
            # Closest to line_end
            return (point_vec - line_end_vec).magnitude
        else:
            # Closest to point on line
            closest_point_vec = line_start_vec + (line_vec * t)
            return (point_vec - closest_point_vec).magnitude
    
    def to_point2d(self) -> Point2D:
        """
        Convert the vector to a Point2D.
        
        Returns:
            Point2D: Point with same x, y coordinates
        """
        return Point2D(self.x, self.y)
    
    @classmethod
    def from_points(cls, start: Point2D, end: Point2D) -> 'Vector2D':
        """
        Create a vector from one point to another.
        
        Args:
            start: Start point
            end: End point
            
        Returns:
            Vector2D: Vector from start to end
        """
        return cls(end.x - start.x, end.y - start.y)
    
    @classmethod
    def from_magnitude_angle(cls, magnitude: float, angle_rad: float) -> 'Vector2D':
        """
        Create a vector from magnitude and angle.
        
        Args:
            magnitude: Length of the vector
            angle_rad: Angle in radians (0 is along positive x-axis)
            
        Returns:
            Vector2D: New vector
        """
        return cls(magnitude * math.cos(angle_rad), magnitude * math.sin(angle_rad))


class Vector3D:
    """
    3D vector representation with basic vector operations.
    
    Used for representing directions, velocities, and forces in 3D space.
    """
    
    def __init__(self, x: float, y: float, z: float):
        """
        Initialize a 3D vector.
        
        Args:
            x: x-component
            y: y-component
            z: z-component
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __repr__(self) -> str:
        """String representation of the vector."""
        return f"Vector3D({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
    def __eq__(self, other) -> bool:
        """Check if two vectors are equal."""
        if not isinstance(other, Vector3D):
            return False
        return (math.isclose(self.x, other.x) and 
                math.isclose(self.y, other.y) and 
                math.isclose(self.z, other.z))
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        """Vector addition."""
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        """Vector subtraction."""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        """Scalar multiplication."""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector3D':
        """Scalar multiplication (right multiplication)."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3D':
        """Scalar division."""
        if math.isclose(scalar, 0.0):
            raise ZeroDivisionError("Division by zero")
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self) -> 'Vector3D':
        """Vector negation."""
        return Vector3D(-self.x, -self.y, -self.z)
    
    @property
    def magnitude(self) -> float:
        """Magnitude (length) of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    @property
    def magnitude_squared(self) -> float:
        """Squared magnitude of the vector (more efficient when only comparing lengths)."""
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    @property
    def normalized(self) -> 'Vector3D':
        """
        Normalized vector (unit vector in the same direction).
        
        Returns:
            Vector3D: Normalized vector
            
        Raises:
            ValueError: If vector has zero magnitude
        """
        mag = self.magnitude
        if math.isclose(mag, 0.0):
            raise ValueError("Cannot normalize a zero vector")
        return self / mag
    
    def dot(self, other: 'Vector3D') -> float:
        """
        Dot product with another vector.
        
        Args:
            other: Another vector
            
        Returns:
            float: Dot product
        """
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """
        Cross product with another vector.
        
        Args:
            other: Another vector
            
        Returns:
            Vector3D: Cross product vector
        """
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def angle_with(self, other: 'Vector3D') -> float:
        """
        Calculate the angle between this vector and another.
        
        Args:
            other: Another vector
            
        Returns:
            float: Angle in radians
            
        Raises:
            ValueError: If either vector has zero magnitude
        """
        if math.isclose(self.magnitude, 0.0) or math.isclose(other.magnitude, 0.0):
            raise ValueError("Cannot compute angle with zero vector")
            
        # Use dot product formula: cos(theta) = (a·b) / (|a|·|b|)
        dot_product = self.dot(other)
        cos_angle = dot_product / (self.magnitude * other.magnitude)
        
        # Clamp to [-1, 1] to handle floating-point errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        return math.acos(cos_angle)
    
    def project_onto(self, other: 'Vector3D') -> 'Vector3D':
        """
        Project this vector onto another vector.
        
        Args:
            other: Vector to project onto
            
        Returns:
            Vector3D: Projection of this vector onto other
            
        Raises:
            ValueError: If other vector has zero magnitude
        """
        if math.isclose(other.magnitude, 0.0):
            raise ValueError("Cannot project onto a zero vector")
            
        dot = self.dot(other)
        other_mag_squared = other.magnitude_squared
        scalar = dot / other_mag_squared
        
        return other * scalar
    
    def to_point3d(self) -> Point3D:
        """
        Convert the vector to a Point3D.
        
        Returns:
            Point3D: Point with same x, y, z coordinates
        """
        return Point3D(self.x, self.y, self.z)
    
    def to_vector2d(self) -> Vector2D:
        """
        Convert to a 2D vector, discarding the z-component.
        
        Returns:
            Vector2D: 2D vector with same x and y components
        """
        return Vector2D(self.x, self.y)
    
    @classmethod
    def from_points(cls, start: Point3D, end: Point3D) -> 'Vector3D':
        """
        Create a vector from one point to another.
        
        Args:
            start: Start point
            end: End point
            
        Returns:
            Vector3D: Vector from start to end
        """
        return cls(end.x - start.x, end.y - start.y, end.z - start.z)
    
    @classmethod
    def from_vector2d(cls, vector2d: Vector2D, z: float = 0.0) -> 'Vector3D':
        """
        Create a 3D vector from a 2D vector and a z component.
        
        Args:
            vector2d: 2D vector
            z: z-component value
            
        Returns:
            Vector3D: 3D vector with added z-component
        """
        return cls(vector2d.x, vector2d.y, z)


def normal_vector_2d(p1: Point2D, p2: Point2D) -> Vector2D:
    """
    Calculate a normal vector to the line segment from p1 to p2.
    
    Args:
        p1: First point
        p2: Second point
        
    Returns:
        Vector2D: Unit normal vector
        
    Raises:
        ValueError: If p1 and p2 are the same point
    """
    direction = Vector2D.from_points(p1, p2)
    if math.isclose(direction.magnitude, 0.0):
        raise ValueError("Cannot compute normal vector between identical points")
    
    # Get perpendicular vector and normalize
    normal = direction.perpendicular()
    return normal.normalized


def interpolate_vectors(v1: Vector2D, v2: Vector2D, t: float) -> Vector2D:
    """
    Linearly interpolate between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        t: Interpolation parameter (0.0 to 1.0)
        
    Returns:
        Vector2D: Interpolated vector
    """
    return v1 * (1 - t) + v2 * t


def angle_between_vectors(v1: Vector2D, v2: Vector2D) -> float:
    """
    Calculate the angle between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        float: Angle in radians
        
    Raises:
        ValueError: If either vector has zero magnitude
    """
    return v1.angle_with(v2)


def signed_angle_2d(v1: Vector2D, v2: Vector2D) -> float:
    """
    Calculate the signed angle from v1 to v2 (positive if counterclockwise).
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        float: Signed angle in radians (-π to π)
        
    Raises:
        ValueError: If either vector has zero magnitude
    """
    if math.isclose(v1.magnitude, 0.0) or math.isclose(v2.magnitude, 0.0):
        raise ValueError("Cannot compute angle with zero vector")
    
    # Use atan2 to get the signed angle
    angle = math.atan2(v2.y, v2.x) - math.atan2(v1.y, v1.x)
    
    # Normalize to [-π, π]
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle <= -math.pi:
        angle += 2 * math.pi
    
    return angle


def reflect_vector(vector: Vector2D, normal: Vector2D) -> Vector2D:
    """
    Reflect a vector across a normal vector.
    
    Args:
        vector: Vector to reflect
        normal: Normal vector (will be normalized)
        
    Returns:
        Vector2D: Reflected vector
        
    Raises:
        ValueError: If normal has zero magnitude
    """
    if math.isclose(normal.magnitude, 0.0):
        raise ValueError("Normal vector cannot have zero magnitude")
    
    normal_unit = normal.normalized
    dot_product = vector.dot(normal_unit)
    return vector - (2 * dot_product * normal_unit)