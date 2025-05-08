"""
Terrain feature definitions for the mining dispatch system.

Defines various terrain features such as slopes, hills, excavation areas,
and other physical characteristics of the environment.
"""

from enum import Enum, auto
import math
from typing import Dict, List, Tuple, Optional, Union, Any
import uuid

from utils.geo.coordinates import Point2D, Point3D, BoundingBox


class TerrainType(Enum):
    """Enumeration of terrain types."""
    FLAT = auto()
    SLOPE = auto()
    HILL = auto()
    VALLEY = auto()
    EXCAVATION = auto()
    ROAD = auto()
    WATER = auto()
    RESTRICTED = auto()


class SlopeDirection(Enum):
    """Enumeration of slope directions."""
    NORTH = auto()
    NORTHEAST = auto()
    EAST = auto()
    SOUTHEAST = auto()
    SOUTH = auto()
    SOUTHWEST = auto()
    WEST = auto()
    NORTHWEST = auto()


class MaterialType(Enum):
    """Enumeration of material types."""
    DIRT = auto()
    CLAY = auto()
    SAND = auto()
    GRAVEL = auto()
    LOOSE_ROCK = auto()
    SOLID_ROCK = auto()
    ORE = auto()
    WASTE = auto()
    WATER = auto()


class TerrainFeature:
    """
    Base class for terrain features.
    
    Terrain features are physical characteristics of the environment
    that affect vehicle movement, operations, and planning.
    """
    
    def __init__(self, feature_id: str = None, name: str = None, 
                 bounds: BoundingBox = None, properties: Dict[str, Any] = None):
        """
        Initialize a terrain feature.
        
        Args:
            feature_id: Unique identifier for the feature
            name: Name of the feature
            bounds: Bounding box of the feature
            properties: Additional properties
        """
        self.feature_id = feature_id or str(uuid.uuid4())
        self.name = name or f"Feature_{self.feature_id[:8]}"
        self.bounds = bounds or BoundingBox(Point2D(0, 0), Point2D(0, 0))
        self.properties = properties or {}
    
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if the feature contains a point.
        
        Args:
            point: Point to check
            
        Returns:
            bool: True if the feature contains the point
        """
        # Base implementation just checks bounding box
        return self.bounds.contains_point(point)
    
    def get_elevation(self, point: Point2D) -> float:
        """
        Get the elevation at a point within the feature.
        
        Args:
            point: Point to get elevation for
            
        Returns:
            float: Elevation in meters
        """
        # Base implementation returns zero elevation
        return 0.0
    
    def get_slope(self, point: Point2D) -> Tuple[float, SlopeDirection]:
        """
        Get the slope at a point within the feature.
        
        Args:
            point: Point to get slope for
            
        Returns:
            Tuple[float, SlopeDirection]: Slope angle in degrees and direction
        """
        # Base implementation returns flat terrain
        return 0.0, SlopeDirection.NORTH
    
    def get_hardness(self, point: Point2D) -> float:
        """
        Get the terrain hardness at a point within the feature.
        
        Args:
            point: Point to get hardness for
            
        Returns:
            float: Hardness value (0.0 to 10.0)
        """
        # Base implementation returns medium hardness
        return 5.0
    
    def get_material(self, point: Point2D) -> MaterialType:
        """
        Get the material type at a point within the feature.
        
        Args:
            point: Point to get material for
            
        Returns:
            MaterialType: Material type
        """
        # Base implementation returns dirt
        return MaterialType.DIRT
    
    def is_traversable(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> bool:
        """
        Check if a point is traversable by a vehicle.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for checking capability
            
        Returns:
            bool: True if traversable
        """
        # Base implementation assumes traversable
        return True
    
    def get_traversal_cost(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> float:
        """
        Get the cost of traversing a point.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for calculating cost
            
        Returns:
            float: Traversal cost (higher means more difficult)
        """
        # Base implementation returns unit cost
        return 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert feature to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'feature_id': self.feature_id,
            'name': self.name,
            'type': self.__class__.__name__,
            'bounds': {
                'min_x': self.bounds.min_point.x,
                'min_y': self.bounds.min_point.y,
                'max_x': self.bounds.max_point.x,
                'max_y': self.bounds.max_point.y
            },
            'properties': self.properties
        }


class SlopeFeature(TerrainFeature):
    """
    Slope terrain feature.
    
    Represents a sloped area with constant gradient.
    """
    
    def __init__(self, feature_id: str = None, name: str = None, 
                 bounds: BoundingBox = None, properties: Dict[str, Any] = None,
                 direction: SlopeDirection = SlopeDirection.NORTH,
                 angle: float = 10.0, base_elevation: float = 0.0):
        """
        Initialize a slope feature.
        
        Args:
            feature_id: Unique identifier for the feature
            name: Name of the feature
            bounds: Bounding box of the feature
            properties: Additional properties
            direction: Direction of the slope
            angle: Angle of the slope in degrees
            base_elevation: Elevation at the lowest point
        """
        super().__init__(feature_id, name, bounds, properties)
        
        self.direction = direction
        self.angle = angle
        self.base_elevation = base_elevation
        
        # Calculate height difference
        self.height = self._calculate_height()
    
    def _calculate_height(self) -> float:
        """
        Calculate the height difference from base to top.
        
        Returns:
            float: Height difference
        """
        # Calculate based on slope length and angle
        slope_length = self._get_slope_length()
        return slope_length * math.tan(math.radians(self.angle))
    
    def _get_slope_length(self) -> float:
        """
        Get the length of the slope.
        
        Returns:
            float: Slope length
        """
        # Determine length based on direction
        if self.direction in [SlopeDirection.NORTH, SlopeDirection.SOUTH]:
            return self.bounds.height
        elif self.direction in [SlopeDirection.EAST, SlopeDirection.WEST]:
            return self.bounds.width
        else:
            # Diagonal direction, use average of width and height
            return (self.bounds.width + self.bounds.height) / 2.0
    
    def _get_directional_vector(self) -> Tuple[float, float]:
        """
        Get directional vector for the slope.
        
        Returns:
            Tuple[float, float]: (dx, dy) vector
        """
        # Map direction to vector components
        vectors = {
            SlopeDirection.NORTH: (0, -1),
            SlopeDirection.NORTHEAST: (1, -1),
            SlopeDirection.EAST: (1, 0),
            SlopeDirection.SOUTHEAST: (1, 1),
            SlopeDirection.SOUTH: (0, 1),
            SlopeDirection.SOUTHWEST: (-1, 1),
            SlopeDirection.WEST: (-1, 0),
            SlopeDirection.NORTHWEST: (-1, -1)
        }
        
        return vectors.get(self.direction, (0, 0))
    
    def _get_position_along_slope(self, point: Point2D) -> float:
        """
        Get normalized position along the slope direction.
        
        Args:
            point: Point to check
            
        Returns:
            float: Normalized position (0.0 to 1.0)
        """
        # Convert to local coordinates
        local_x = (point.x - self.bounds.min_point.x) / self.bounds.width
        local_y = (point.y - self.bounds.min_point.y) / self.bounds.height
        
        # Map direction to position calculation
        if self.direction == SlopeDirection.NORTH:
            return 1.0 - local_y
        elif self.direction == SlopeDirection.NORTHEAST:
            return (local_x + (1.0 - local_y)) / 2.0
        elif self.direction == SlopeDirection.EAST:
            return local_x
        elif self.direction == SlopeDirection.SOUTHEAST:
            return (local_x + local_y) / 2.0
        elif self.direction == SlopeDirection.SOUTH:
            return local_y
        elif self.direction == SlopeDirection.SOUTHWEST:
            return ((1.0 - local_x) + local_y) / 2.0
        elif self.direction == SlopeDirection.WEST:
            return 1.0 - local_x
        elif self.direction == SlopeDirection.NORTHWEST:
            return ((1.0 - local_x) + (1.0 - local_y)) / 2.0
        else:
            return 0.0
    
    def get_elevation(self, point: Point2D) -> float:
        """
        Get the elevation at a point on the slope.
        
        Args:
            point: Point to get elevation for
            
        Returns:
            float: Elevation in meters
        """
        if not self.contains_point(point):
            return self.base_elevation
        
        # Get position along slope
        position = self._get_position_along_slope(point)
        
        # Calculate elevation based on position
        return self.base_elevation + (position * self.height)
    
    def get_slope(self, point: Point2D) -> Tuple[float, SlopeDirection]:
        """
        Get the slope at a point on the feature.
        
        Args:
            point: Point to get slope for
            
        Returns:
            Tuple[float, SlopeDirection]: Slope angle in degrees and direction
        """
        if not self.contains_point(point):
            return 0.0, SlopeDirection.NORTH
        
        return self.angle, self.direction
    
    def get_traversal_cost(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> float:
        """
        Get the cost of traversing a point on the slope.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for calculating cost
            
        Returns:
            float: Traversal cost
        """
        if not self.contains_point(point):
            return 1.0
        
        # Base cost based on slope angle
        # Steeper slopes are more costly
        base_cost = 1.0 + (self.angle / 10.0)
        
        # Adjust based on vehicle properties
        if vehicle_properties:
            # Check if vehicle is going uphill or downhill
            if 'direction' in vehicle_properties:
                vehicle_dir = vehicle_properties['direction']
                slope_dir = self._get_directional_vector()
                
                # Calculate dot product to determine if going uphill or downhill
                dot_product = vehicle_dir[0] * slope_dir[0] + vehicle_dir[1] * slope_dir[1]
                
                # Going uphill is more costly than downhill
                if dot_product > 0:  # Uphill
                    base_cost *= 1.5
                elif dot_product < 0:  # Downhill
                    base_cost *= 0.8
            
            # Adjust based on vehicle slope capability
            if 'slope_capability' in vehicle_properties:
                slope_capability = vehicle_properties['slope_capability']
                if slope_capability > 0:
                    # Reduce cost if vehicle has good slope capability
                    capability_factor = 1.0 - (slope_capability / 30.0)  # 30 degrees is full capability
                    base_cost *= max(0.5, capability_factor)
        
        return base_cost
    
    def is_traversable(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> bool:
        """
        Check if a point on the slope is traversable.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for checking capability
            
        Returns:
            bool: True if traversable
        """
        if not self.contains_point(point):
            return True
        
        # Check if slope is too steep for the vehicle
        if vehicle_properties and 'max_slope' in vehicle_properties:
            max_slope = vehicle_properties['max_slope']
            return self.angle <= max_slope
        
        # Default: assume traversable if less than 30 degrees
        return self.angle <= 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert slope feature to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        base_dict = super().to_dict()
        
        slope_dict = {
            'direction': self.direction.name,
            'angle': self.angle,
            'base_elevation': self.base_elevation,
            'height': self.height
        }
        
        return {**base_dict, **slope_dict}


class HillFeature(TerrainFeature):
    """
    Hill terrain feature.
    
    Represents a hill or mound with peak at the center.
    """
    
    def __init__(self, feature_id: str = None, name: str = None, 
                 bounds: BoundingBox = None, properties: Dict[str, Any] = None,
                 peak_elevation: float = 10.0, base_elevation: float = 0.0,
                 peak_offset: Tuple[float, float] = (0.5, 0.5)):
        """
        Initialize a hill feature.
        
        Args:
            feature_id: Unique identifier for the feature
            name: Name of the feature
            bounds: Bounding box of the feature
            properties: Additional properties
            peak_elevation: Elevation at the peak
            base_elevation: Elevation at the base
            peak_offset: Normalized offset of peak from center (0-1, 0-1)
        """
        super().__init__(feature_id, name, bounds, properties)
        
        self.peak_elevation = peak_elevation
        self.base_elevation = base_elevation
        self.peak_offset = peak_offset
        
        # Calculate peak position
        self.peak_position = self._calculate_peak_position()
    
    def _calculate_peak_position(self) -> Point2D:
        """
        Calculate the absolute position of the peak.
        
        Returns:
            Point2D: Peak position
        """
        x = self.bounds.min_point.x + (self.bounds.width * self.peak_offset[0])
        y = self.bounds.min_point.y + (self.bounds.height * self.peak_offset[1])
        return Point2D(x, y)
    
    def get_elevation(self, point: Point2D) -> float:
        """
        Get the elevation at a point on the hill.
        
        Args:
            point: Point to get elevation for
            
        Returns:
            float: Elevation in meters
        """
        if not self.contains_point(point):
            return self.base_elevation
        
        # Calculate distance from peak as a percentage of max possible distance
        dx = point.x - self.peak_position.x
        dy = point.y - self.peak_position.y
        
        # Calculate normalized distance from peak (0 at peak, 1 at max distance)
        max_distance = math.sqrt((self.bounds.width / 2)**2 + (self.bounds.height / 2)**2)
        distance = math.sqrt(dx**2 + dy**2)
        normalized_distance = min(1.0, distance / max_distance)
        
        # Use cosine function for smooth hill profile
        # cos(0) = 1.0 at peak, cos(π/2) = 0.0 at base
        height_factor = math.cos(normalized_distance * (math.pi / 2))
        
        # Calculate elevation
        elevation_diff = self.peak_elevation - self.base_elevation
        return self.base_elevation + (height_factor * elevation_diff)
    
    def get_slope(self, point: Point2D) -> Tuple[float, SlopeDirection]:
        """
        Get the slope at a point on the hill.
        
        Args:
            point: Point to get slope for
            
        Returns:
            Tuple[float, SlopeDirection]: Slope angle in degrees and direction
        """
        if not self.contains_point(point):
            return 0.0, SlopeDirection.NORTH
        
        # Calculate vector from point to peak
        dx = self.peak_position.x - point.x
        dy = self.peak_position.y - point.y
        
        # Determine slope direction (away from peak)
        direction = SlopeDirection.NORTH  # Default
        
        if abs(dx) > abs(dy):
            # Primarily horizontal
            if dx > 0:
                direction = SlopeDirection.EAST
            else:
                direction = SlopeDirection.WEST
        else:
            # Primarily vertical
            if dy > 0:
                direction = SlopeDirection.SOUTH
            else:
                direction = SlopeDirection.NORTH
        
        # Adjust for diagonals
        if abs(dx) > 0.4 * abs(dy) and abs(dy) > 0.4 * abs(dx):
            if dx > 0 and dy > 0:
                direction = SlopeDirection.SOUTHEAST
            elif dx > 0 and dy < 0:
                direction = SlopeDirection.NORTHEAST
            elif dx < 0 and dy > 0:
                direction = SlopeDirection.SOUTHWEST
            else:
                direction = SlopeDirection.NORTHWEST
        
        # Calculate slope angle
        # Higher slope near peak, gentler slope near base
        normalized_distance = point.distance_to(self.peak_position) / (
            max(self.bounds.width, self.bounds.height) / 2)
        
        height_diff = self.peak_elevation - self.base_elevation
        max_slope_angle = math.degrees(math.atan2(height_diff, 
                                                 min(self.bounds.width, self.bounds.height) / 2))
        
        # Adjust slope based on distance from peak
        angle = max_slope_angle * (1.0 - normalized_distance)
        
        return angle, direction
    
    def get_traversal_cost(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> float:
        """
        Get the cost of traversing a point on the hill.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for calculating cost
            
        Returns:
            float: Traversal cost
        """
        if not self.contains_point(point):
            return 1.0
        
        # Get slope at point
        angle, _ = self.get_slope(point)
        
        # Base cost based on slope angle
        base_cost = 1.0 + (angle / 15.0)
        
        # Adjust based on vehicle properties
        if vehicle_properties:
            # Adjust based on vehicle slope capability
            if 'slope_capability' in vehicle_properties:
                slope_capability = vehicle_properties['slope_capability']
                if slope_capability > 0:
                    # Reduce cost if vehicle has good slope capability
                    capability_factor = 1.0 - (slope_capability / 30.0)
                    base_cost *= max(0.5, capability_factor)
        
        return base_cost
    
    def is_traversable(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> bool:
        """
        Check if a point on the hill is traversable.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for checking capability
            
        Returns:
            bool: True if traversable
        """
        if not self.contains_point(point):
            return True
        
        # Get slope at point
        angle, _ = self.get_slope(point)
        
        # Check if slope is too steep for the vehicle
        if vehicle_properties and 'max_slope' in vehicle_properties:
            max_slope = vehicle_properties['max_slope']
            return angle <= max_slope
        
        # Default: assume traversable if less than 30 degrees
        return angle <= 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert hill feature to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        base_dict = super().to_dict()
        
        hill_dict = {
            'peak_elevation': self.peak_elevation,
            'base_elevation': self.base_elevation,
            'peak_offset': self.peak_offset,
            'peak_position': {
                'x': self.peak_position.x,
                'y': self.peak_position.y
            }
        }
        
        return {**base_dict, **hill_dict}


class ExcavationFeature(TerrainFeature):
    """
    Excavation terrain feature.
    
    Represents an excavated area with lower elevation than surroundings.
    """
    
    def __init__(self, feature_id: str = None, name: str = None, 
                 bounds: BoundingBox = None, properties: Dict[str, Any] = None,
                 depth: float = 10