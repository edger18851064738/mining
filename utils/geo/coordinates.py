"""
Coordinate system definitions and point classes for the mining dispatch system.

This module provides standardized point classes and operations to ensure
consistent coordinate handling throughout the system.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import math
from typing import Tuple, List, Union, Optional, TypeVar, Generic

# Type variable for generic point operations
P = TypeVar('P', bound='BasePoint')

class BasePoint:
    """Base class for all point types"""
    
    def distance_to(self, other: 'BasePoint') -> float:
        """Calculate distance to another point"""
        raise NotImplementedError("Subclasses must implement distance_to")
    
    def as_tuple(self) -> tuple:
        """Return point coordinates as a tuple"""
        raise NotImplementedError("Subclasses must implement as_tuple")


class Point2D(BasePoint):
    """
    Two-dimensional point class for representing locations in a plane.
    
    Standard point format for all 2D operations in the system.
    """
    
    def __init__(self, x: float, y: float):
        """
        Initialize a 2D point.
        
        Args:
            x: x-coordinate
            y: y-coordinate
        """
        self.x = float(x)
        self.y = float(y)
    
    def __repr__(self) -> str:
        """String representation of the point."""
        return f"Point2D({self.x:.2f}, {self.y:.2f})"
    
    def __eq__(self, other) -> bool:
        """Check if two points are equal."""
        if not isinstance(other, Point2D):
            return False
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)
    
    def __hash__(self) -> int:
        """Hash value for using points in dictionaries and sets."""
        return hash((round(self.x, 6), round(self.y, 6)))
    
    def distance_to(self, other: 'Point2D') -> float:
        """
        Calculate Euclidean distance to another point.
        
        Args:
            other: Another Point2D object
            
        Returns:
            float: Euclidean distance between points
        """
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def as_tuple(self) -> Tuple[float, float]:
        """Return point coordinates as a tuple."""
        return (self.x, self.y)
    
    @classmethod
    def from_tuple(cls, coords: Tuple[float, float]) -> 'Point2D':
        """
        Create a Point2D from a tuple of coordinates.
        
        Args:
            coords: Tuple of (x, y) coordinates
            
        Returns:
            Point2D: A new Point2D object
        """
        return cls(coords[0], coords[1])
    
    @classmethod
    def from_any(cls, point: Union[Tuple[float, float], List[float], 'Point2D', 'Point3D']) -> 'Point2D':
        """
        Convert various point representations to a Point2D.
        
        Args:
            point: Point representation, which can be a tuple, list, Point2D, or Point3D
            
        Returns:
            Point2D: A new Point2D object
            
        Raises:
            ValueError: If the input format is not recognized
        """
        if isinstance(point, tuple) and len(point) >= 2:
            return cls(point[0], point[1])
        elif isinstance(point, list) and len(point) >= 2:
            return cls(point[0], point[1])
        elif isinstance(point, Point2D):
            return point
        elif hasattr(point, 'x') and hasattr(point, 'y'):
            return cls(point.x, point.y)
        else:
            raise ValueError(f"Cannot convert {point} to Point2D")
    
    def midpoint(self, other: 'Point2D') -> 'Point2D':
        """
        Calculate the midpoint between this point and another.
        
        Args:
            other: Another Point2D object
            
        Returns:
            Point2D: Midpoint between the two points
        """
        return Point2D((self.x + other.x) / 2, (self.y + other.y) / 2)
    
    def translate(self, dx: float, dy: float) -> 'Point2D':
        """
        Translate the point by the given offsets.
        
        Args:
            dx: X-offset
            dy: Y-offset
            
        Returns:
            Point2D: New translated point
        """
        return Point2D(self.x + dx, self.y + dy)


class Point3D(BasePoint):
    """
    Three-dimensional point class for representing locations in 3D space.
    
    Used for positions with elevation or points with orientation.
    """
    
    def __init__(self, x: float, y: float, z: float):
        """
        Initialize a 3D point.
        
        Args:
            x: x-coordinate
            y: y-coordinate
            z: z-coordinate (often elevation or orientation)
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __repr__(self) -> str:
        """String representation of the point."""
        return f"Point3D({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
    def __eq__(self, other) -> bool:
        """Check if two points are equal."""
        if not isinstance(other, Point3D):
            return False
        return (math.isclose(self.x, other.x) and 
                math.isclose(self.y, other.y) and 
                math.isclose(self.z, other.z))
    
    def __hash__(self) -> int:
        """Hash value for using points in dictionaries and sets."""
        return hash((round(self.x, 6), round(self.y, 6), round(self.z, 6)))
    
    def distance_to(self, other: 'Point3D') -> float:
        """
        Calculate Euclidean distance to another point in 3D space.
        
        Args:
            other: Another Point3D object
            
        Returns:
            float: Euclidean distance between points
        """
        return math.sqrt((self.x - other.x)**2 + 
                         (self.y - other.y)**2 + 
                         (self.z - other.z)**2)
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Return point coordinates as a tuple."""
        return (self.x, self.y, self.z)
    
    def to_2d(self) -> Point2D:
        """
        Convert to a 2D point, discarding the z-coordinate.
        
        Returns:
            Point2D: A 2D point with the same x and y coordinates
        """
        return Point2D(self.x, self.y)
    
    @classmethod
    def from_tuple(cls, coords: Tuple[float, float, float]) -> 'Point3D':
        """
        Create a Point3D from a tuple of coordinates.
        
        Args:
            coords: Tuple of (x, y, z) coordinates
            
        Returns:
            Point3D: A new Point3D object
        """
        return cls(coords[0], coords[1], coords[2])
    
    @classmethod
    def from_2d(cls, point: Point2D, z: float = 0.0) -> 'Point3D':
        """
        Create a 3D point from a 2D point and a z value.
        
        Args:
            point: A Point2D object
            z: Z-coordinate value (default: 0.0)
            
        Returns:
            Point3D: A 3D point with z-coordinate added
        """
        return cls(point.x, point.y, z)
    
    @classmethod
    def from_any(cls, point: Union[Tuple, List, 'Point2D', 'Point3D'], 
                default_z: float = 0.0) -> 'Point3D':
        """
        Convert various point representations to a Point3D.
        
        Args:
            point: Point representation, which can be a tuple, list, Point2D, or Point3D
            default_z: Default z-coordinate to use if not present in input (default: 0.0)
            
        Returns:
            Point3D: A new Point3D object
            
        Raises:
            ValueError: If the input format is not recognized
        """
        if isinstance(point, tuple):
            if len(point) >= 3:
                return cls(point[0], point[1], point[2])
            elif len(point) == 2:
                return cls(point[0], point[1], default_z)
        elif isinstance(point, list):
            if len(point) >= 3:
                return cls(point[0], point[1], point[2])
            elif len(point) == 2:
                return cls(point[0], point[1], default_z)
        elif isinstance(point, Point2D):
            return cls(point.x, point.y, default_z)
        elif isinstance(point, Point3D):
            return point
        elif hasattr(point, 'x') and hasattr(point, 'y'):
            z = getattr(point, 'z', default_z)
            return cls(point.x, point.y, z)
        
        raise ValueError(f"Cannot convert {point} to Point3D")


class BoundingBox:
    """
    Rectangular area defined by minimum and maximum coordinates.
    
    Used for collision detection and spatial queries.
    """
    
    def __init__(self, min_point: Point2D, max_point: Point2D):
        """
        Initialize a bounding box.
        
        Args:
            min_point: Minimum coordinates (lower-left corner)
            max_point: Maximum coordinates (upper-right corner)
        """
        self.min_point = min_point
        self.max_point = max_point
    
    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.max_point.x - self.min_point.x
    
    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.max_point.y - self.min_point.y
    
    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Point2D:
        """Center point of the bounding box."""
        return Point2D(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2
        )
    
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if the bounding box contains a point.
        
        Args:
            point: The point to check
            
        Returns:
            bool: True if the point is inside the bounding box
        """
        return (self.min_point.x <= point.x <= self.max_point.x and
                self.min_point.y <= point.y <= self.max_point.y)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """
        Check if this bounding box intersects with another.
        
        Args:
            other: Another BoundingBox
            
        Returns:
            bool: True if the bounding boxes intersect
        """
        return not (self.max_point.x < other.min_point.x or
                    self.min_point.x > other.max_point.x or
                    self.max_point.y < other.min_point.y or
                    self.min_point.y > other.max_point.y)
    
    @classmethod
    def from_points(cls, points: List[Point2D]) -> 'BoundingBox':
        """
        Create a bounding box that encompasses all given points.
        
        Args:
            points: List of points
            
        Returns:
            BoundingBox: Bounding box containing all points
            
        Raises:
            ValueError: If the points list is empty
        """
        if not points:
            raise ValueError("Cannot create a bounding box from an empty list of points")
            
        min_x = min(p.x for p in points)
        min_y = min(p.y for p in points)
        max_x = max(p.x for p in points)
        max_y = max(p.y for p in points)
        
        return cls(Point2D(min_x, min_y), Point2D(max_x, max_y))
    
    def expand(self, margin: float) -> 'BoundingBox':
        """
        Expand the bounding box by a margin in all directions.
        
        Args:
            margin: Distance to expand in each direction
            
        Returns:
            BoundingBox: Expanded bounding box
        """
        return BoundingBox(
            Point2D(self.min_point.x - margin, self.min_point.y - margin),
            Point2D(self.max_point.x + margin, self.max_point.y + margin)
        )

# Function to convert various coordinate formats to standard Point2D
def normalize_to_point2d(coord: Union[tuple, list, Point2D, Point3D, object]) -> Point2D:
    """
    Convert various coordinate representations to a standard Point2D.
    
    Args:
        coord: Coordinate in various formats (tuple, list, Point2D, Point3D, or object with x,y attributes)
        
    Returns:
        Point2D: Standardized 2D point
        
    Raises:
        ValueError: If the coordinate format is not recognized
    """
    try:
        if isinstance(coord, Point2D):
            return coord
        elif isinstance(coord, Point3D):
            return coord.to_2d()
        elif isinstance(coord, (tuple, list)) and len(coord) >= 2:
            return Point2D(coord[0], coord[1])
        elif hasattr(coord, 'x') and hasattr(coord, 'y'):
            return Point2D(coord.x, coord.y)
        else:
            raise ValueError(f"Cannot convert {coord} to Point2D")
    except Exception as e:
        raise ValueError(f"Error converting {coord} to Point2D: {str(e)}")

# Function to convert various coordinate formats to standard Point3D
def normalize_to_point3d(coord: Union[tuple, list, Point2D, Point3D, object], 
                         default_z: float = 0.0) -> Point3D:
    """
    Convert various coordinate representations to a standard Point3D.
    
    Args:
        coord: Coordinate in various formats
        default_z: Default z-coordinate if not present in input
        
    Returns:
        Point3D: Standardized 3D point
        
    Raises:
        ValueError: If the coordinate format is not recognized
    """
    try:
        return Point3D.from_any(coord, default_z)
    except Exception as e:
        raise ValueError(f"Error converting {coord} to Point3D: {str(e)}")