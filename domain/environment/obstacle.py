"""
Obstacle definitions for the mining dispatch system.

Defines various types of obstacles that can exist in the environment.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Set, Union
import math

from utils.geo.coordinates import Point2D, BoundingBox
from utils.io.serialization import Serializable


class Obstacle(ABC, Serializable):
    """
    Abstract base class for all obstacle types.
    
    Obstacles are physical barriers that vehicles cannot traverse.
    """
    
    @abstractmethod
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if the obstacle contains a point.
        
        Args:
            point: Position to check
            
        Returns:
            bool: True if point is inside or on the obstacle
        """
        pass
    
    @abstractmethod
    def distance_to(self, point: Point2D) -> float:
        """
        Calculate the minimum distance from a point to the obstacle.
        
        Args:
            point: Position to measure from
            
        Returns:
            float: Minimum distance to obstacle (0 if point is inside)
        """
        pass
    
    @abstractmethod
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of the obstacle.
        
        Returns:
            BoundingBox: Smallest rectangle containing the obstacle
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert obstacle to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Obstacle':
        """
        Create an obstacle from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Obstacle: New obstacle instance
        """
        pass


class PointObstacle(Obstacle):
    """
    Simple point obstacle with a radius.
    
    Represents a circular obstacle centered at a point.
    """
    
    def __init__(self, position: Point2D, radius: float = 1.0):
        """
        Initialize a point obstacle.
        
        Args:
            position: Center of the obstacle
            radius: Radius of the obstacle
        """
        self.position = position
        self.radius = radius
    
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if the obstacle contains a point.
        
        Args:
            point: Position to check
            
        Returns:
            bool: True if point is inside or on the obstacle
        """
        return self.position.distance_to(point) <= self.radius
    
    def distance_to(self, point: Point2D) -> float:
        """
        Calculate the minimum distance from a point to the obstacle.
        
        Args:
            point: Position to measure from
            
        Returns:
            float: Minimum distance to obstacle (0 if point is inside)
        """
        distance = self.position.distance_to(point)
        return max(0.0, distance - self.radius)
    
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of the obstacle.
        
        Returns:
            BoundingBox: Smallest rectangle containing the obstacle
        """
        min_point = Point2D(self.position.x - self.radius, self.position.y - self.radius)
        max_point = Point2D(self.position.x + self.radius, self.position.y + self.radius)
        return BoundingBox(min_point, max_point)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert obstacle to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'type': 'point',
            'position': {
                'x': self.position.x,
                'y': self.position.y
            },
            'radius': self.radius
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PointObstacle':
        """
        Create a point obstacle from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            PointObstacle: New point obstacle instance
        """
        if 'position' not in data:
            raise ValueError("Missing required position in data")
            
        position_data = data['position']
        position = Point2D(position_data['x'], position_data['y'])
        radius = data.get('radius', 1.0)
        
        return cls(position, radius)


class RectangleObstacle(Obstacle):
    """
    Rectangular obstacle defined by two corner points.
    
    Represents a rectangle aligned with the coordinate axes.
    """
    
    def __init__(self, min_point: Point2D, max_point: Point2D):
        """
        Initialize a rectangle obstacle.
        
        Args:
            min_point: Minimum corner (lower-left)
            max_point: Maximum corner (upper-right)
        """
        self.min_point = min_point
        self.max_point = max_point
    
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if the obstacle contains a point.
        
        Args:
            point: Position to check
            
        Returns:
            bool: True if point is inside or on the obstacle
        """
        return (self.min_point.x <= point.x <= self.max_point.x and
                self.min_point.y <= point.y <= self.max_point.y)
    
    def distance_to(self, point: Point2D) -> float:
        """
        Calculate the minimum distance from a point to the obstacle.
        
        Args:
            point: Position to measure from
            
        Returns:
            float: Minimum distance to obstacle (0 if point is inside)
        """
        if self.contains_point(point):
            return 0.0
            
        # Calculate distance to each edge and take minimum
        dx = max(0, max(self.min_point.x - point.x, point.x - self.max_point.x))
        dy = max(0, max(self.min_point.y - point.y, point.y - self.max_point.y))
        
        return math.sqrt(dx*dx + dy*dy)
    
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of the obstacle.
        
        Returns:
            BoundingBox: Smallest rectangle containing the obstacle
        """
        return BoundingBox(self.min_point, self.max_point)
    
    @property
    def width(self) -> float:
        """Get the width of the rectangle."""
        return self.max_point.x - self.min_point.x
    
    @property
    def height(self) -> float:
        """Get the height of the rectangle."""
        return self.max_point.y - self.min_point.y
    
    @property
    def area(self) -> float:
        """Get the area of the rectangle."""
        return self.width * self.height
    
    @property
    def center(self) -> Point2D:
        """Get the center point of the rectangle."""
        return Point2D(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert obstacle to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'type': 'rectangle',
            'min_point': {
                'x': self.min_point.x,
                'y': self.min_point.y
            },
            'max_point': {
                'x': self.max_point.x,
                'y': self.max_point.y
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RectangleObstacle':
        """
        Create a rectangle obstacle from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            RectangleObstacle: New rectangle obstacle instance
        """
        if 'min_point' not in data or 'max_point' not in data:
            raise ValueError("Missing required min_point or max_point in data")
            
        min_data = data['min_point']
        max_data = data['max_point']
        
        min_point = Point2D(min_data['x'], min_data['y'])
        max_point = Point2D(max_data['x'], max_data['y'])
        
        return cls(min_point, max_point)


class PolygonObstacle(Obstacle):
    """
    Polygon obstacle defined by a list of vertices.
    
    Represents a closed polygon of arbitrary shape.
    """
    
    def __init__(self, vertices: List[Point2D]):
        """
        Initialize a polygon obstacle.
        
        Args:
            vertices: List of polygon vertices in order
        """
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
            
        self.vertices = vertices
    
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if the obstacle contains a point using ray casting algorithm.
        
        Args:
            point: Position to check
            
        Returns:
            bool: True if point is inside or on the obstacle
        """
        # Check if point is on any edge
        if self._point_on_edge(point):
            return True
            
        # Ray casting algorithm for point-in-polygon test
        inside = False
        n = len(self.vertices)
        
        for i in range(n):
            j = (i + 1) % n
            
            if ((self.vertices[i].y > point.y) != (self.vertices[j].y > point.y) and
                (point.x < (self.vertices[j].x - self.vertices[i].x) * 
                 (point.y - self.vertices[i].y) / 
                 (self.vertices[j].y - self.vertices[i].y) + self.vertices[i].x)):
                inside = not inside
                
        return inside
    
    def _point_on_edge(self, point: Point2D) -> bool:
        """
        Check if a point lies on any edge of the polygon.
        
        Args:
            point: Position to check
            
        Returns:
            bool: True if point is on any edge
        """
        n = len(self.vertices)
        
        for i in range(n):
            j = (i + 1) % n
            
            # Check if point is on line segment
            if self._point_on_line_segment(point, self.vertices[i], self.vertices[j]):
                return True
                
        return False
    
    def _point_on_line_segment(self, p: Point2D, a: Point2D, b: Point2D) -> bool:
        """
        Check if point p is on line segment ab.
        
        Args:
            p: Point to check
            a: Segment start
            b: Segment end
            
        Returns:
            bool: True if point is on segment
        """
        # Cross product near zero means collinear
        cross_product = (p.y - a.y) * (b.x - a.x) - (p.x - a.x) * (b.y - a.y)
        if abs(cross_product) > 1e-9:
            return False
            
        # Check if point is within segment bounds
        return (min(a.x, b.x) <= p.x <= max(a.x, b.x) and
                min(a.y, b.y) <= p.y <= max(a.y, b.y))
    
    def distance_to(self, point: Point2D) -> float:
        """
        Calculate the minimum distance from a point to the obstacle.
        
        Args:
            point: Position to measure from
            
        Returns:
            float: Minimum distance to obstacle (0 if point is inside)
        """
        if self.contains_point(point):
            return 0.0
            
        # Find minimum distance to any edge
        min_distance = float('inf')
        n = len(self.vertices)
        
        for i in range(n):
            j = (i + 1) % n
            
            # Distance from point to line segment
            edge_distance = self._point_to_line_segment_distance(
                point, self.vertices[i], self.vertices[j]
            )
            
            min_distance = min(min_distance, edge_distance)
            
        return min_distance
    
    def _point_to_line_segment_distance(self, p: Point2D, a: Point2D, b: Point2D) -> float:
        """
        Calculate distance from point p to line segment ab.
        
        Args:
            p: Point
            a: Segment start
            b: Segment end
            
        Returns:
            float: Minimum distance
        """
        # Vector from a to b
        ab_x = b.x - a.x
        ab_y = b.y - a.y
        
        # Vector from a to p
        ap_x = p.x - a.x
        ap_y = p.y - a.y
        
        # Squared length of ab
        ab_len_sq = ab_x * ab_x + ab_y * ab_y
        
        if ab_len_sq < 1e-9:
            # a and b are the same point
            return math.sqrt(ap_x * ap_x + ap_y * ap_y)
            
        # Calculate projection proportion (dot product / squared length)
        t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq
        
        if t < 0:
            # Beyond a
            return math.sqrt(ap_x * ap_x + ap_y * ap_y)
        elif t > 1:
            # Beyond b
            bp_x = p.x - b.x
            bp_y = p.y - b.y
            return math.sqrt(bp_x * bp_x + bp_y * bp_y)
        else:
            # Projected point is on the segment
            proj_x = a.x + t * ab_x
            proj_y = a.y + t * ab_y
            
            dx = p.x - proj_x
            dy = p.y - proj_y
            
            return math.sqrt(dx * dx + dy * dy)
    
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of the obstacle.
        
        Returns:
            BoundingBox: Smallest rectangle containing the obstacle
        """
        min_x = min(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        max_y = max(v.y for v in self.vertices)
        
        return BoundingBox(Point2D(min_x, min_y), Point2D(max_x, max_y))
    
    @property
    def perimeter(self) -> float:
        """Calculate the perimeter of the polygon."""
        n = len(self.vertices)
        perimeter = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            perimeter += self.vertices[i].distance_to(self.vertices[j])
            
        return perimeter
    
    @property
    def area(self) -> float:
        """Calculate the area of the polygon using the shoelace formula."""
        n = len(self.vertices)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[i].y * self.vertices[j].x
            
        return abs(area) / 2.0
    
    @property
    def centroid(self) -> Point2D:
        """Calculate the centroid of the polygon."""
        n = len(self.vertices)
        area = 0.0
        cx = 0.0
        cy = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            
            # Shoelace formula terms
            cross = (self.vertices[i].x * self.vertices[j].y - 
                    self.vertices[j].x * self.vertices[i].y)
            
            area += cross
            cx += (self.vertices[i].x + self.vertices[j].x) * cross
            cy += (self.vertices[i].y + self.vertices[j].y) * cross
        
        area /= 2.0
        area = abs(area)
        
        if area < 1e-9:
            # Degenerate polygon, return average of vertices
            cx = sum(v.x for v in self.vertices) / n
            cy = sum(v.y for v in self.vertices) / n
            return Point2D(cx, cy)
            
        cx /= 6.0 * area
        cy /= 6.0 * area
        
        return Point2D(cx, cy)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert obstacle to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'type': 'polygon',
            'vertices': [
                {'x': v.x, 'y': v.y} for v in self.vertices
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolygonObstacle':
        """
        Create a polygon obstacle from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            PolygonObstacle: New polygon obstacle instance
        """
        if 'vertices' not in data:
            raise ValueError("Missing required vertices in data")
            
        vertices_data = data['vertices']
        vertices = [Point2D(v['x'], v['y']) for v in vertices_data]
        
        return cls(vertices)


class GridObstacle:
    """
    Grid cell obstacle for grid-based environments.
    
    Represents a single cell in a grid that is an obstacle.
    """
    
    def __init__(self, grid_x: int, grid_y: int):
        """
        Initialize a grid obstacle.
        
        Args:
            grid_x: Grid x-coordinate
            grid_y: Grid y-coordinate
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
    
    def __eq__(self, other):
        """Check if two grid obstacles are equal."""
        if not isinstance(other, GridObstacle):
            return False
        return self.grid_x == other.grid_x and self.grid_y == other.grid_y
    
    def __hash__(self):
        """Hash function for grid obstacles."""
        return hash((self.grid_x, self.grid_y))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert obstacle to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'type': 'grid',
            'grid_x': self.grid_x,
            'grid_y': self.grid_y
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridObstacle':
        """
        Create a grid obstacle from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            GridObstacle: New grid obstacle instance
        """
        if 'grid_x' not in data or 'grid_y' not in data:
            raise ValueError("Missing required grid_x or grid_y in data")
            
        return cls(data['grid_x'], data['grid_y'])


class CompositeObstacle(Obstacle):
    """
    Composite obstacle consisting of multiple other obstacles.
    
    Allows for creating complex obstacle shapes from simpler components.
    """
    
    def __init__(self, obstacles: List[Obstacle]):
        """
        Initialize a composite obstacle.
        
        Args:
            obstacles: List of component obstacles
        """
        self.obstacles = obstacles
    
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if any component obstacle contains the point.
        
        Args:
            point: Position to check
            
        Returns:
            bool: True if point is inside or on any component
        """
        return any(obstacle.contains_point(point) for obstacle in self.obstacles)
    
    def distance_to(self, point: Point2D) -> float:
        """
        Calculate the minimum distance from a point to any component.
        
        Args:
            point: Position to measure from
            
        Returns:
            float: Minimum distance to any component (0 if point is inside)
        """
        if self.contains_point(point):
            return 0.0
            
        return min(obstacle.distance_to(point) for obstacle in self.obstacles)
    
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box containing all components.
        
        Returns:
            BoundingBox: Smallest rectangle containing all components
        """
        if not self.obstacles:
            # Empty composite, return a point at origin
            return BoundingBox(Point2D(0, 0), Point2D(0, 0))
            
        # Get individual bounding boxes
        boxes = [obstacle.get_bounding_box() for obstacle in self.obstacles]
        
        # Find min/max coordinates
        min_x = min(box.min_point.x for box in boxes)
        min_y = min(box.min_point.y for box in boxes)
        max_x = max(box.max_point.x for box in boxes)
        max_y = max(box.max_point.y for box in boxes)
        
        return BoundingBox(Point2D(min_x, min_y), Point2D(max_x, max_y))
    
    def add_obstacle(self, obstacle: Obstacle) -> None:
        """
        Add a component obstacle.
        
        Args:
            obstacle: Obstacle to add
        """
        self.obstacles.append(obstacle)
    
    def remove_obstacle(self, obstacle: Obstacle) -> bool:
        """
        Remove a component obstacle.
        
        Args:
            obstacle: Obstacle to remove
            
        Returns:
            bool: True if obstacle was removed, False if not found
        """
        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert obstacle to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'type': 'composite',
            'obstacles': [obstacle.to_dict() for obstacle in self.obstacles]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompositeObstacle':
        """
        Create a composite obstacle from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            CompositeObstacle: New composite obstacle instance
        """
        if 'obstacles' not in data:
            raise ValueError("Missing required obstacles in data")
            
        obstacles_data = data['obstacles']
        obstacles = []
        
        for obstacle_data in obstacles_data:
            obstacle_type = obstacle_data.get('type')
            
            if obstacle_type == 'point':
                obstacles.append(PointObstacle.from_dict(obstacle_data))
            elif obstacle_type == 'rectangle':
                obstacles.append(RectangleObstacle.from_dict(obstacle_data))
            elif obstacle_type == 'polygon':
                obstacles.append(PolygonObstacle.from_dict(obstacle_data))
            elif obstacle_type == 'grid':
                obstacles.append(GridObstacle.from_dict(obstacle_data))
                
        return cls(obstacles)