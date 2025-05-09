"""
Base environment definitions for the mining dispatch system.

Defines the abstract environment interface and common environment behavior.
All specific environment types should inherit from these base classes.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from abc import ABC, abstractmethod
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable
from datetime import datetime
import math

from utils.geo.coordinates import Point2D, BoundingBox
from utils.io.serialization import Serializable
from utils.logger import get_logger

# Initialize logger
logger = get_logger("environment")


class EnvironmentError(Exception):
    """Base exception for environment-related errors."""
    pass


class Environment(ABC, Serializable):
    """
    Abstract base class for all environment types.
    
    Defines the common interface and behavior that all environments must implement.
    An environment represents the physical space in which vehicles operate.
    """
    
    def __init__(self, name: str, 
                bounds: Union[BoundingBox, Tuple[float, float, float, float]],
                resolution: float = 1.0):
        """
        Initialize an environment.
        
        Args:
            name: Name of the environment
            bounds: Bounding box of the environment (BoundingBox or (min_x, min_y, max_x, max_y))
            resolution: Resolution of the environment grid in meters
        """
        self.name = name
        
        # Convert bounds to BoundingBox if needed
        if isinstance(bounds, tuple) and len(bounds) == 4:
            min_x, min_y, max_x, max_y = bounds
            self.bounds = BoundingBox(Point2D(min_x, min_y), Point2D(max_x, max_y))
        elif isinstance(bounds, BoundingBox):
            self.bounds = bounds
        else:
            raise ValueError(f"Invalid bounds format: {bounds}")
        
        self.resolution = resolution
        self.creation_time = datetime.now()
        self.last_update_time = self.creation_time
        
        # Core components
        self._obstacles = set()  # Set of obstacle points
        self._key_locations = {}  # Dict of key locations (name -> Point2D)
    
    @property
    def width(self) -> float:
        """Get the width of the environment."""
        return self.bounds.width
    
    @property
    def height(self) -> float:
        """Get the height of the environment."""
        return self.bounds.height
    
    @property
    def center(self) -> Point2D:
        """Get the center point of the environment."""
        return self.bounds.center
    
    @property
    def obstacles(self) -> Set[Point2D]:
        """Get the set of obstacle points."""
        return self._obstacles.copy()
    
    @property
    def key_locations(self) -> Dict[str, Point2D]:
        """Get the dictionary of key locations."""
        return self._key_locations.copy()
    
    def add_obstacle(self, point: Union[Point2D, Tuple[float, float]]) -> None:
        """
        Add an obstacle to the environment.
        
        Args:
            point: Obstacle point
        """
        if isinstance(point, tuple) and len(point) >= 2:
            self._obstacles.add(Point2D(point[0], point[1]))
        elif isinstance(point, Point2D):
            self._obstacles.add(point)
        else:
            raise ValueError(f"Invalid point format: {point}")
        
        self.last_update_time = datetime.now()
    
    def add_obstacles(self, points: List[Union[Point2D, Tuple[float, float]]]) -> None:
        """
        Add multiple obstacles to the environment.
        
        Args:
            points: List of obstacle points
        """
        for point in points:
            self.add_obstacle(point)
    
    def remove_obstacle(self, point: Union[Point2D, Tuple[float, float]]) -> bool:
        """
        Remove an obstacle from the environment.
        
        Args:
            point: Obstacle point
            
        Returns:
            bool: True if obstacle was removed, False if not found
        """
        target = None
        if isinstance(point, tuple) and len(point) >= 2:
            target = Point2D(point[0], point[1])
        elif isinstance(point, Point2D):
            target = point
        else:
            raise ValueError(f"Invalid point format: {point}")
        
        # Find exact match or closest point
        if target in self._obstacles:
            self._obstacles.remove(target)
            self.last_update_time = datetime.now()
            return True
        
        # Try to find closest match within resolution
        for obs in self._obstacles:
            if obs.distance_to(target) < self.resolution:
                self._obstacles.remove(obs)
                self.last_update_time = datetime.now()
                return True
        
        return False
    
    def clear_obstacles(self) -> None:
        """Clear all obstacles from the environment."""
        self._obstacles.clear()
        self.last_update_time = datetime.now()
    
    def add_key_location(self, name: str, point: Union[Point2D, Tuple[float, float]]) -> None:
        """
        Add a key location to the environment.
        
        Args:
            name: Name of the location
            point: Location point
        """
        if isinstance(point, tuple) and len(point) >= 2:
            self._key_locations[name] = Point2D(point[0], point[1])
        elif isinstance(point, Point2D):
            self._key_locations[name] = point
        else:
            raise ValueError(f"Invalid point format: {point}")
        
        self.last_update_time = datetime.now()
    
    def remove_key_location(self, name: str) -> bool:
        """
        Remove a key location from the environment.
        
        Args:
            name: Name of the location
            
        Returns:
            bool: True if location was removed, False if not found
        """
        if name in self._key_locations:
            del self._key_locations[name]
            self.last_update_time = datetime.now()
            return True
        return False
    
    def get_key_location(self, name: str) -> Optional[Point2D]:
        """
        Get a key location by name.
        
        Args:
            name: Name of the location
            
        Returns:
            Optional[Point2D]: Location point, or None if not found
        """
        return self._key_locations.get(name)
    
    @abstractmethod
    def is_obstacle(self, point: Union[Point2D, Tuple[float, float]]) -> bool:
        """
        Check if a point is an obstacle.
        
        Args:
            point: Point to check
            
        Returns:
            bool: True if point is an obstacle, False otherwise
        """
        pass
    
    @abstractmethod
    def is_traversable(self, point: Union[Point2D, Tuple[float, float]], 
                     vehicle=None) -> bool:
        """
        Check if a point is traversable by a vehicle.
        
        Args:
            point: Point to check
            vehicle: Vehicle to check traversability for
            
        Returns:
            bool: True if point is traversable, False otherwise
        """
        pass
    
    @abstractmethod
    def find_path(self, start: Union[Point2D, Tuple[float, float]],
                end: Union[Point2D, Tuple[float, float]],
                vehicle=None) -> List[Point2D]:
        """
        Find a path from start to end.
        
        Args:
            start: Start point
            end: End point
            vehicle: Vehicle to find path for
            
        Returns:
            List[Point2D]: Path from start to end
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert environment to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'name': self.name,
            'bounds': {
                'min_x': self.bounds.min_point.x,
                'min_y': self.bounds.min_point.y,
                'max_x': self.bounds.max_point.x,
                'max_y': self.bounds.max_point.y
            },
            'resolution': self.resolution,
            'creation_time': self.creation_time.isoformat(),
            'last_update_time': self.last_update_time.isoformat(),
            'obstacles': [{'x': p.x, 'y': p.y} for p in self._obstacles],
            'key_locations': {name: {'x': p.x, 'y': p.y} for name, p in self._key_locations.items()}
        }


class GridEnvironment(Environment):
    """
    Environment represented as a grid of cells.
    
    Each cell can be traversable or non-traversable.
    """
    
    def __init__(self, name: str, 
                bounds: Union[BoundingBox, Tuple[float, float, float, float]],
                resolution: float = 1.0):
        """
        Initialize a grid environment.
        
        Args:
            name: Name of the environment
            bounds: Bounding box of the environment
            resolution: Resolution of the grid in meters
        """
        super().__init__(name, bounds, resolution)
        
        # Calculate grid dimensions
        self.grid_width = math.ceil(self.width / self.resolution)
        self.grid_height = math.ceil(self.height / self.resolution)
        
        # Initialize grid (all cells traversable)
        self.grid = [[True for _ in range(self.grid_height)] for _ in range(self.grid_width)]
        
        # Initialize terrain properties grid (default hardness)
        self.terrain_grid = [[1.0 for _ in range(self.grid_height)] for _ in range(self.grid_width)]
    
    def _point_to_grid(self, point: Union[Point2D, Tuple[float, float]]) -> Tuple[int, int]:
        """
        Convert a point to grid coordinates.
        
        Args:
            point: Point in world coordinates
            
        Returns:
            Tuple[int, int]: Grid coordinates (x, y)
            
        Raises:
            ValueError: If point is outside grid bounds
        """
        if isinstance(point, tuple) and len(point) >= 2:
            x, y = point[0], point[1]
        elif isinstance(point, Point2D):
            x, y = point.x, point.y
        else:
            raise ValueError(f"Invalid point format: {point}")
        
        # Convert to grid coordinates
        grid_x = int((x - self.bounds.min_point.x) / self.resolution)
        grid_y = int((y - self.bounds.min_point.y) / self.resolution)
        
        # Check if within bounds
        if (grid_x < 0 or grid_x >= self.grid_width or
            grid_y < 0 or grid_y >= self.grid_height):
            raise ValueError(f"Point ({x}, {y}) is outside grid bounds")
        
        return grid_x, grid_y
    
    def _grid_to_point(self, grid_x: int, grid_y: int) -> Point2D:
        """
        Convert grid coordinates to a point.
        
        Args:
            grid_x: Grid x-coordinate
            grid_y: Grid y-coordinate
            
        Returns:
            Point2D: Point in world coordinates
            
        Raises:
            ValueError: If grid coordinates are outside grid bounds
        """
        # Check if within bounds
        if (grid_x < 0 or grid_x >= self.grid_width or
            grid_y < 0 or grid_y >= self.grid_height):
            raise ValueError(f"Grid coordinates ({grid_x}, {grid_y}) are outside grid bounds")
        
        # Convert to world coordinates (center of cell)
        x = self.bounds.min_point.x + (grid_x + 0.5) * self.resolution
        y = self.bounds.min_point.y + (grid_y + 0.5) * self.resolution
        
        return Point2D(x, y)
    
    def add_obstacle(self, point: Union[Point2D, Tuple[float, float]]) -> None:
        """
        Add an obstacle to the grid.
        
        Args:
            point: Obstacle point
            
        Raises:
            ValueError: If point is outside grid bounds
        """
        # Add to obstacles set
        super().add_obstacle(point)
        
        # Update grid
        grid_x, grid_y = self._point_to_grid(point)
        self.grid[grid_x][grid_y] = False
    
    def remove_obstacle(self, point: Union[Point2D, Tuple[float, float]]) -> bool:
        """
        Remove an obstacle from the grid.
        
        Args:
            point: Obstacle point
            
        Returns:
            bool: True if obstacle was removed, False if not found
            
        Raises:
            ValueError: If point is outside grid bounds
        """
        # Try to convert to grid coordinates
        try:
            grid_x, grid_y = self._point_to_grid(point)
        except ValueError:
            return False
        
        # Check if cell is an obstacle
        if not self.grid[grid_x][grid_y]:
            # Update grid
            self.grid[grid_x][grid_y] = True
            
            # Remove from obstacles set
            result = super().remove_obstacle(point)
            
            return result
        
        return False
    
    def clear_obstacles(self) -> None:
        """Clear all obstacles from the grid."""
        super().clear_obstacles()
        
        # Reset grid (all cells traversable)
        self.grid = [[True for _ in range(self.grid_height)] for _ in range(self.grid_width)]
    
    def set_terrain_property(self, point: Union[Point2D, Tuple[float, float]], 
                          hardness: float) -> None:
        """
        Set terrain property at a point.
        
        Args:
            point: Point to set property for
            hardness: Terrain hardness (0-1, where 1 is solid)
            
        Raises:
            ValueError: If point is outside grid bounds
        """
        grid_x, grid_y = self._point_to_grid(point)
        self.terrain_grid[grid_x][grid_y] = max(0.0, min(1.0, hardness))
    
    def get_terrain_property(self, point: Union[Point2D, Tuple[float, float]]) -> float:
        """
        Get terrain property at a point.
        
        Args:
            point: Point to get property for
            
        Returns:
            float: Terrain hardness
            
        Raises:
            ValueError: If point is outside grid bounds
        """
        try:
            grid_x, grid_y = self._point_to_grid(point)
            return self.terrain_grid[grid_x][grid_y]
        except ValueError:
            # For points outside bounds, return maximum hardness
            return 1.0
    
    def is_obstacle(self, point: Union[Point2D, Tuple[float, float]]) -> bool:
        """
        Check if a point is an obstacle.
        
        Args:
            point: Point to check
            
        Returns:
            bool: True if point is an obstacle, False otherwise
        """
        try:
            grid_x, grid_y = self._point_to_grid(point)
            return not self.grid[grid_x][grid_y]
        except ValueError:
            # Points outside bounds are considered obstacles
            return True
    
    def is_traversable(self, point: Union[Point2D, Tuple[float, float]], 
                     vehicle=None) -> bool:
        """
        Check if a point is traversable by a vehicle.
        
        Args:
            point: Point to check
            vehicle: Vehicle to check traversability for
            
        Returns:
            bool: True if point is traversable, False otherwise
        """
        # Check if point is an obstacle
        if self.is_obstacle(point):
            return False
        
        # Check vehicle-specific traversability
        if vehicle is not None:
            try:
                grid_x, grid_y = self._point_to_grid(point)
                terrain_hardness = self.terrain_grid[grid_x][grid_y]
                
                # If vehicle has terrain capability attribute
                if hasattr(vehicle, 'terrain_capability'):
                    # Vehicle can traverse terrain if its capability exceeds terrain hardness
                    return vehicle.terrain_capability >= terrain_hardness
            except ValueError:
                return False
        
        # Default: traversable if not an obstacle
        return True
    
    def find_path(self, start: Union[Point2D, Tuple[float, float]],
                end: Union[Point2D, Tuple[float, float]],
                vehicle=None) -> List[Point2D]:
        """
        Find a path from start to end using A* algorithm.
        
        Args:
            start: Start point
            end: End point
            vehicle: Vehicle to find path for
            
        Returns:
            List[Point2D]: Path from start to end
            
        Raises:
            EnvironmentError: If no path is found
        """
        # Convert to grid coordinates
        try:
            start_grid = self._point_to_grid(start)
            end_grid = self._point_to_grid(end)
        except ValueError as e:
            raise EnvironmentError(f"Path points outside grid bounds: {str(e)}")
        
        # Check if start or end is obstacle
        if self.is_obstacle(start):
            raise EnvironmentError(f"Start point is an obstacle")
        if self.is_obstacle(end):
            raise EnvironmentError(f"End point is an obstacle")
        
        # Implement A* algorithm
        import heapq
        
        # Heuristic function (Manhattan distance)
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # Initialize open and closed sets
        open_set = []
        closed_set = set()
        
        # Start node: (f_score, grid_x, grid_y, path)
        start_node = (0, start_grid[0], start_grid[1], [])
        heapq.heappush(open_set, start_node)
        
        while open_set:
            # Get node with lowest f_score
            f, x, y, path = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if (x, y) == end_grid:
                # Add end point to path
                full_path = path + [self._grid_to_point(x, y)]
                return full_path
            
            # Skip if already processed
            if (x, y) in closed_set:
                continue
            
            # Mark as processed
            closed_set.add((x, y))
            
            # Add current point to path
            new_path = path + [self._grid_to_point(x, y)]
            
            # Check all neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), 
                          (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                
                # Skip if outside grid
                if (nx < 0 or nx >= self.grid_width or
                    ny < 0 or ny >= self.grid_height):
                    continue
                
                # Skip if already processed
                if (nx, ny) in closed_set:
                    continue
                
                # Skip if obstacle or not traversable
                neighbor_point = self._grid_to_point(nx, ny)
                if not self.is_traversable(neighbor_point, vehicle):
                    continue
                
                # Calculate costs
                # Movement cost (diagonal is more expensive)
                movement_cost = 1.4 if dx != 0 and dy != 0 else 1.0
                
                # Terrain cost based on hardness
                terrain_cost = 1.0
                if vehicle and hasattr(vehicle, 'terrain_capability'):
                    terrain_hardness = self.terrain_grid[nx][ny]
                    terrain_cost = 1.0 + terrain_hardness  # Harder terrain is more costly
                
                # Total cost for this neighbor
                g_score = len(new_path) + movement_cost * terrain_cost
                h_score = heuristic((nx, ny), end_grid)
                f_score = g_score + h_score
                
                # Add to open set
                heapq.heappush(open_set, (f_score, nx, ny, new_path))
        
        # No path found
        raise EnvironmentError(f"No path found from {start} to {end}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert grid environment to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        base_dict = super().to_dict()
        
        # Convert grid to more compact representation
        grid_repr = []
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if not self.grid[x][y]:  # If cell is obstacle
                    grid_repr.append((x, y))
        
        grid_dict = {
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'obstacles_grid': grid_repr
        }
        
        return {**base_dict, **grid_dict}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridEnvironment':
        """
        Create a grid environment from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            GridEnvironment: New grid environment instance
        """
        # Extract bounds
        bounds_data = data['bounds']
        bounds = (
            bounds_data['min_x'],
            bounds_data['min_y'],
            bounds_data['max_x'],
            bounds_data['max_y']
        )
        
        # Create environment
        env = cls(
            name=data['name'],
            bounds=bounds,
            resolution=data['resolution']
        )
        
        # Set grid dimensions
        if 'grid_width' in data and 'grid_height' in data:
            env.grid_width = data['grid_width']
            env.grid_height = data['grid_height']
            
            # Initialize grid (all cells traversable)
            env.grid = [[True for _ in range(env.grid_height)] for _ in range(env.grid_width)]
        
        # Add obstacles from grid representation
        if 'obstacles_grid' in data:
            for x, y in data['obstacles_grid']:
                if 0 <= x < env.grid_width and 0 <= y < env.grid_height:
                    env.grid[x][y] = False
        
        # Add obstacles from world coordinates
        if 'obstacles' in data:
            for obs in data['obstacles']:
                try:
                    point = Point2D(obs['x'], obs['y'])
                    env._obstacles.add(point)
                except (KeyError, ValueError):
                    continue
        
        # Add key locations
        if 'key_locations' in data:
            for name, loc in data['key_locations'].items():
                try:
                    point = Point2D(loc['x'], loc['y'])
                    env._key_locations[name] = point
                except (KeyError, ValueError):
                    continue
        
        # Set timestamps
        if 'creation_time' in data:
            try:
                env.creation_time = datetime.fromisoformat(data['creation_time'])
            except (ValueError, TypeError):
                pass
                
        if 'last_update_time' in data:
            try:
                env.last_update_time = datetime.fromisoformat(data['last_update_time'])
            except (ValueError, TypeError):
                pass
        
        return env