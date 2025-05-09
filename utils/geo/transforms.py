"""
Coordinate transformation utilities for the mining dispatch system.

Provides functions for converting between different coordinate systems,
including grid/meter conversions and transformations between local and global coordinates.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import math
from typing import Tuple, List, Union, Dict, Optional, Any
import numpy as np

from utils.geo.coordinates import Point2D, Point3D, normalize_to_point2d, normalize_to_point3d


class CoordinateTransformer:
    """
    Transformer for converting between different coordinate systems.
    
    Provides methods for converting between:
    - Grid and meter coordinates
    - Local and global coordinates
    
    Uses a configurable grid size and origin point.
    """
    
    def __init__(self, grid_size: float = 1.0, origin: Point2D = None):
        """
        Initialize the coordinate transformer.
        
        Args:
            grid_size: Size of a grid cell in meters (default: 1.0)
            origin: Origin point in the grid system (default: (0,0))
        """
        self.grid_size = float(grid_size)
        self.origin = origin or Point2D(0.0, 0.0)
    
    def configure(self, grid_size: float = None, origin: Point2D = None) -> None:
        """
        Configure transformer settings.
        
        Args:
            grid_size: New grid cell size in meters (optional)
            origin: New origin point (optional)
        """
        if grid_size is not None:
            self.grid_size = float(grid_size)
        
        if origin is not None:
            self.origin = normalize_to_point2d(origin)
    
    def grid_to_meters(self, grid_point: Union[Point2D, Tuple, List]) -> Point2D:
        """
        Convert grid coordinates to meter coordinates.
        
        Args:
            grid_point: Point in grid coordinates
            
        Returns:
            Point2D: Point in meter coordinates
        """
        point = normalize_to_point2d(grid_point)
        
        meter_x = (point.x - self.origin.x) * self.grid_size
        meter_y = (point.y - self.origin.y) * self.grid_size
        
        return Point2D(meter_x, meter_y)
    
    def meters_to_grid(self, meter_point: Union[Point2D, Tuple, List]) -> Point2D:
        """
        Convert meter coordinates to grid coordinates.
        
        Args:
            meter_point: Point in meter coordinates
            
        Returns:
            Point2D: Point in grid coordinates
        """
        point = normalize_to_point2d(meter_point)
        
        grid_x = point.x / self.grid_size + self.origin.x
        grid_y = point.y / self.grid_size + self.origin.y
        
        return Point2D(grid_x, grid_y)
    
    def meters_to_grid_rounded(self, meter_point: Union[Point2D, Tuple, List]) -> Point2D:
        """
        Convert meter coordinates to grid coordinates, rounding to nearest grid cell.
        
        Args:
            meter_point: Point in meter coordinates
            
        Returns:
            Point2D: Point in grid coordinates, rounded to nearest grid cell
        """
        grid_point = self.meters_to_grid(meter_point)
        return Point2D(round(grid_point.x), round(grid_point.y))


def rotate_point(point: Point2D, angle_rad: float, origin: Point2D = None) -> Point2D:
    """
    Rotate a point around an origin by a given angle.
    
    Args:
        point: Point to rotate
        angle_rad: Rotation angle in radians
        origin: Origin of rotation (default: (0,0))
        
    Returns:
        Point2D: Rotated point
    """
    if origin is None:
        origin = Point2D(0, 0)
    
    # Translate point to origin
    translated_x = point.x - origin.x
    translated_y = point.y - origin.y
    
    # Rotate
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    rotated_x = translated_x * cos_angle - translated_y * sin_angle
    rotated_y = translated_x * sin_angle + translated_y * cos_angle
    
    # Translate back
    rotated_x += origin.x
    rotated_y += origin.y
    
    return Point2D(rotated_x, rotated_y)


def rotate_path(path: List[Point2D], angle_rad: float, origin: Point2D = None) -> List[Point2D]:
    """
    Rotate a path (list of points) around an origin by a given angle.
    
    Args:
        path: List of points to rotate
        angle_rad: Rotation angle in radians
        origin: Origin of rotation (default: centroid of path)
        
    Returns:
        List[Point2D]: Rotated path
    """
    if not path:
        return []
    
    # If origin is not specified, use centroid of path
    if origin is None:
        centroid_x = sum(p.x for p in path) / len(path)
        centroid_y = sum(p.y for p in path) / len(path)
        origin = Point2D(centroid_x, centroid_y)
    
    # Rotate each point
    return [rotate_point(p, angle_rad, origin) for p in path]


def scale_point(point: Point2D, scale_x: float, scale_y: float = None, origin: Point2D = None) -> Point2D:
    """
    Scale a point relative to an origin.
    
    Args:
        point: Point to scale
        scale_x: Scale factor for x coordinate
        scale_y: Scale factor for y coordinate (default: same as scale_x)
        origin: Origin of scaling (default: (0,0))
        
    Returns:
        Point2D: Scaled point
    """
    if scale_y is None:
        scale_y = scale_x
    
    if origin is None:
        origin = Point2D(0, 0)
    
    # Translate point to origin
    translated_x = point.x - origin.x
    translated_y = point.y - origin.y
    
    # Scale
    scaled_x = translated_x * scale_x
    scaled_y = translated_y * scale_y
    
    # Translate back
    scaled_x += origin.x
    scaled_y += origin.y
    
    return Point2D(scaled_x, scaled_y)


def transform_point(point: Point2D, transform_matrix: np.ndarray) -> Point2D:
    """
    Apply a transformation matrix to a point.
    
    Args:
        point: Point to transform
        transform_matrix: 3x3 homogeneous transformation matrix
        
    Returns:
        Point2D: Transformed point
        
    Raises:
        ValueError: If transform_matrix is not a 3x3 matrix
    """
    if transform_matrix.shape != (3, 3):
        raise ValueError("Transform matrix must be 3x3")
    
    # Convert to homogeneous coordinates
    homogeneous = np.array([point.x, point.y, 1.0])
    
    # Apply transformation
    transformed = np.dot(transform_matrix, homogeneous)
    
    # Convert back from homogeneous coordinates
    return Point2D(transformed[0] / transformed[2], transformed[1] / transformed[2])


def transform_path(path: List[Point2D], transform_matrix: np.ndarray) -> List[Point2D]:
    """
    Apply a transformation matrix to a path (list of points).
    
    Args:
        path: List of points to transform
        transform_matrix: 3x3 homogeneous transformation matrix
        
    Returns:
        List[Point2D]: Transformed path
    """
    return [transform_point(p, transform_matrix) for p in path]


def create_transformation_matrix(translation: Tuple[float, float] = (0, 0),
                                rotation: float = 0,
                                scale: Tuple[float, float] = (1, 1)) -> np.ndarray:
    """
    Create a 2D transformation matrix combining translation, rotation, and scale.
    
    Args:
        translation: (tx, ty) translation vector
        rotation: Rotation angle in radians
        scale: (sx, sy) scale factors
        
    Returns:
        np.ndarray: 3x3 homogeneous transformation matrix
    """
    # Create rotation matrix
    cos_angle = math.cos(rotation)
    sin_angle = math.sin(rotation)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    
    # Create scale matrix
    scale_matrix = np.array([
        [scale[0], 0, 0],
        [0, scale[1], 0],
        [0, 0, 1]
    ])
    
    # Create translation matrix
    translation_matrix = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])
    
    # Combine matrices: translation * rotation * scale
    # Order matters! This applies scale first, then rotation, then translation
    return np.dot(translation_matrix, np.dot(rotation_matrix, scale_matrix))


def discretize_points(path: List[Point2D], grid_size: float = 1.0) -> List[Point2D]:
    """
    Discretize a path to grid points based on grid size.
    
    Args:
        path: List of points to discretize
        grid_size: Grid cell size
        
    Returns:
        List[Point2D]: Discretized path
    """
    if not path:
        return []
    
    discretized = []
    for point in path:
        grid_x = round(point.x / grid_size)
        grid_y = round(point.y / grid_size)
        discretized.append(Point2D(grid_x, grid_y))
    
    # Remove consecutive duplicates
    result = [discretized[0]]
    for i in range(1, len(discretized)):
        if discretized[i] != discretized[i-1]:
            result.append(discretized[i])
    
    return result


def interpolate_points(start: Point2D, end: Point2D, num_points: int = 10) -> List[Point2D]:
    """
    Generate evenly spaced points along a line from start to end.
    
    Args:
        start: Start point
        end: End point
        num_points: Number of points to generate (including start and end)
        
    Returns:
        List[Point2D]: Interpolated points
        
    Raises:
        ValueError: If num_points is less than 2
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    
    result = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = start.x + t * (end.x - start.x)
        y = start.y + t * (end.y - start.y)
        result.append(Point2D(x, y))
    
    return result


def bresenham_line(start: Point2D, end: Point2D) -> List[Point2D]:
    """
    Generate all grid points along a line from start to end using Bresenham's algorithm.
    
    Args:
        start: Start point (will be rounded to integers)
        end: End point (will be rounded to integers)
        
    Returns:
        List[Point2D]: List of grid points along the line
    """
    # Round to integers
    x1, y1 = round(start.x), round(start.y)
    x2, y2 = round(end.x), round(end.y)
    
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append(Point2D(x1, y1))
        if x1 == x2 and y1 == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return points


def world_to_local(point: Point2D, reference: Point2D, reference_heading: float) -> Point2D:
    """
    Convert a point from world coordinates to local coordinates relative to a reference frame.
    
    Args:
        point: Point in world coordinates
        reference: Origin of local coordinate system in world coordinates
        reference_heading: Heading angle of local coordinate system in radians
        
    Returns:
        Point2D: Point in local coordinates
    """
    # Translate
    translated_x = point.x - reference.x
    translated_y = point.y - reference.y
    
    # Rotate (inverse of heading)
    cos_angle = math.cos(-reference_heading)
    sin_angle = math.sin(-reference_heading)
    
    local_x = translated_x * cos_angle - translated_y * sin_angle
    local_y = translated_x * sin_angle + translated_y * cos_angle
    
    return Point2D(local_x, local_y)


def local_to_world(point: Point2D, reference: Point2D, reference_heading: float) -> Point2D:
    """
    Convert a point from local coordinates to world coordinates.
    
    Args:
        point: Point in local coordinates
        reference: Origin of local coordinate system in world coordinates
        reference_heading: Heading angle of local coordinate system in radians
        
    Returns:
        Point2D: Point in world coordinates
    """
    # Rotate
    cos_angle = math.cos(reference_heading)
    sin_angle = math.sin(reference_heading)
    
    rotated_x = point.x * cos_angle - point.y * sin_angle
    rotated_y = point.x * sin_angle + point.y * cos_angle
    
    # Translate
    world_x = rotated_x + reference.x
    world_y = rotated_y + reference.y
    
    return Point2D(world_x, world_y)


class GridMap:
    """
    Utility for mapping between continuous coordinates and grid cells.
    
    Provides efficient conversion between different coordinate representations.
    """
    
    def __init__(self, origin: Point2D = None, cell_size: float = 1.0):
        """
        Initialize the grid map.
        
        Args:
            origin: Origin point of the grid in world coordinates
            cell_size: Size of a grid cell in world units
        """
        self.origin = origin or Point2D(0, 0)
        self.cell_size = cell_size
    
    def world_to_grid(self, point: Point2D) -> Tuple[int, int]:
        """
        Convert a world coordinate to grid cell indices.
        
        Args:
            point: Point in world coordinates
            
        Returns:
            Tuple[int, int]: Grid cell indices (row, col)
        """
        grid_x = int((point.x - self.origin.x) / self.cell_size)
        grid_y = int((point.y - self.origin.y) / self.cell_size)
        return (grid_y, grid_x)  # Note: row = y, col = x
    
    def grid_to_world(self, row: int, col: int) -> Point2D:
        """
        Convert grid cell indices to world coordinates (center of cell).
        
        Args:
            row: Grid row index
            col: Grid column index
            
        Returns:
            Point2D: Center of grid cell in world coordinates
        """
        world_x = self.origin.x + (col + 0.5) * self.cell_size
        world_y = self.origin.y + (row + 0.5) * self.cell_size
        return Point2D(world_x, world_y)
    
    def world_to_grid_continuous(self, point: Point2D) -> Point2D:
        """
        Convert world coordinates to continuous grid coordinates.
        
        Args:
            point: Point in world coordinates
            
        Returns:
            Point2D: Point in continuous grid coordinates
        """
        grid_x = (point.x - self.origin.x) / self.cell_size
        grid_y = (point.y - self.origin.y) / self.cell_size
        return Point2D(grid_x, grid_y)
    
    def grid_continuous_to_world(self, point: Point2D) -> Point2D:
        """
        Convert continuous grid coordinates to world coordinates.
        
        Args:
            point: Point in continuous grid coordinates
            
        Returns:
            Point2D: Point in world coordinates
        """
        world_x = self.origin.x + point.x * self.cell_size
        world_y = self.origin.y + point.y * self.cell_size
        return Point2D(world_x, world_y)
    
    def get_cell_bounds(self, row: int, col: int) -> Tuple[Point2D, Point2D]:
        """
        Get the minimum and maximum bounds of a grid cell.
        
        Args:
            row: Grid row index
            col: Grid column index
            
        Returns:
            Tuple[Point2D, Point2D]: (min_point, max_point) of cell bounds
        """
        min_x = self.origin.x + col * self.cell_size
        min_y = self.origin.y + row * self.cell_size
        max_x = min_x + self.cell_size
        max_y = min_y + self.cell_size
        
        return (Point2D(min_x, min_y), Point2D(max_x, max_y))