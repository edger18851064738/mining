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
                 depth: float = 10.0, base_elevation: float = 0.0,
                 rim_elevation: float = None, profile: str = "flat"):
        """
        Initialize an excavation feature.
        
        Args:
            feature_id: Unique identifier for the feature
            name: Name of the feature
            bounds: Bounding box of the feature
            properties: Additional properties
            depth: Depth of the excavation in meters
            base_elevation: Elevation at the base of the excavation
            rim_elevation: Elevation at the rim (default: base_elevation + depth)
            profile: Profile shape of the excavation ("flat", "bowl", "stepped")
        """
        super().__init__(feature_id, name, bounds, properties)
        
        self.depth = depth
        self.base_elevation = base_elevation
        self.rim_elevation = rim_elevation if rim_elevation is not None else base_elevation + depth
        self.profile = profile
        
        # Set default material type for excavation
        self.material = MaterialType.LOOSE_ROCK
        
        # Pre-calculate some properties
        self.center = Point2D(
            (self.bounds.min_point.x + self.bounds.max_point.x) / 2,
            (self.bounds.min_point.y + self.bounds.max_point.y) / 2
        )
        self.max_radius = min(self.bounds.width, self.bounds.height) / 2
    
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if a point is within the excavation area.
        
        Args:
            point: Point to check
            
        Returns:
            bool: True if point is inside the excavation
        """
        # Simplified implementation using bounding box and distance from center
        if not super().contains_point(point):
            return False
            
        # For simplicity, assume circular excavation within the bounding box
        distance_from_center = point.distance_to(self.center)
        return distance_from_center <= self.max_radius
    
    def get_elevation(self, point: Point2D) -> float:
        """
        Get the elevation at a point within the excavation.
        
        Args:
            point: Point to get elevation for
            
        Returns:
            float: Elevation in meters
        """
        if not self.contains_point(point):
            return self.rim_elevation
        
        # Calculate normalized distance from center (0 at center, 1 at edge)
        distance_from_center = point.distance_to(self.center)
        normalized_distance = min(1.0, distance_from_center / self.max_radius)
        
        # Calculate elevation based on profile
        if self.profile == "flat":
            # Flat base with steep walls
            if normalized_distance > 0.9:
                # Transition zone at the edge
                t = (normalized_distance - 0.9) / 0.1
                return self.base_elevation + (t * (self.rim_elevation - self.base_elevation))
            else:
                return self.base_elevation
        
        elif self.profile == "bowl":
            # Smooth bowl-shaped excavation
            # Use quadratic function: elevation = base + (rim-base)*dist²
            return self.base_elevation + (self.rim_elevation - self.base_elevation) * normalized_distance**2
        
        elif self.profile == "stepped":
            # Stepped excavation with 3 levels
            if normalized_distance < 0.33:
                return self.base_elevation
            elif normalized_distance < 0.67:
                return self.base_elevation + self.depth / 3
            else:
                return self.base_elevation + 2 * self.depth / 3
        
        # Default fallback
        return self.base_elevation
    
    def get_slope(self, point: Point2D) -> Tuple[float, SlopeDirection]:
        """
        Get the slope at a point within the excavation.
        
        Args:
            point: Point to get slope for
            
        Returns:
            Tuple[float, SlopeDirection]: Slope angle in degrees and direction
        """
        if not self.contains_point(point):
            return 0.0, SlopeDirection.NORTH
        
        # Vector from point to center
        dx = self.center.x - point.x
        dy = self.center.y - point.y
        
        # Determine slope direction (toward center)
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
        
        # Calculate normalized distance from center (0 at center, 1 at edge)
        distance_from_center = point.distance_to(self.center)
        normalized_distance = min(1.0, distance_from_center / self.max_radius)
        
        # Calculate slope angle based on profile
        if self.profile == "flat":
            # Flat base with steep walls
            if normalized_distance > 0.9:
                # Steep walls at edges
                return 45.0, direction
            else:
                return 0.0, direction
                
        elif self.profile == "bowl":
            # For bowl shape, slope increases linearly with distance from center
            # At edge (normalized_distance=1), slope is atan(depth/radius)
            max_angle = math.degrees(math.atan2(self.depth, self.max_radius))
            return max_angle * normalized_distance, direction
            
        elif self.profile == "stepped":
            # Stepped excavation with steep transitions between levels
            if abs(normalized_distance - 0.33) < 0.02 or abs(normalized_distance - 0.67) < 0.02:
                return 45.0, direction
            else:
                return 0.0, direction
        
        # Default fallback
        return 0.0, direction
    
    def get_hardness(self, point: Point2D) -> float:
        """
        Get the terrain hardness at a point within the excavation.
        
        Args:
            point: Point to get hardness for
            
        Returns:
            float: Hardness value (0.0 to 10.0)
        """
        if not self.contains_point(point):
            return 5.0  # Default medium hardness
        
        # Excavated areas typically have lower hardness
        # Deeper parts might have different rock types or be partially filled with debris
        distance_from_center = point.distance_to(self.center)
        normalized_distance = min(1.0, distance_from_center / self.max_radius)
        
        # Base hardness for excavated material
        base_hardness = self.properties.get('base_hardness', 3.0)
        
        # Adjust hardness based on depth profile
        if self.profile == "flat":
            return base_hardness
        elif self.profile == "bowl":
            # Harder at deeper parts
            return base_hardness + (1.0 - normalized_distance) * 2.0
        elif self.profile == "stepped":
            # Different hardness at different levels
            if normalized_distance < 0.33:
                return base_hardness + 2.0  # Hardest at bottom level
            elif normalized_distance < 0.67:
                return base_hardness + 1.0  # Medium at middle level
            else:
                return base_hardness  # Softest at top level
        
        return base_hardness
    
    def get_material(self, point: Point2D) -> MaterialType:
        """
        Get the material type at a point within the excavation.
        
        Args:
            point: Point to get material for
            
        Returns:
            MaterialType: Material type
        """
        if not self.contains_point(point):
            return MaterialType.DIRT
        
        # Get configured material type
        return self.properties.get('material_type', self.material)
    
    def is_traversable(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> bool:
        """
        Check if a point within the excavation is traversable.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for checking capability
            
        Returns:
            bool: True if traversable
        """
        if not self.contains_point(point):
            return True
        
        # Check slope at point
        angle, _ = self.get_slope(point)
        
        # Check if slope is too steep for the vehicle
        if vehicle_properties and 'max_slope' in vehicle_properties:
            max_slope = vehicle_properties['max_slope']
            if angle > max_slope:
                return False
        
        # Check if vehicle can handle this material
        material = self.get_material(point)
        if vehicle_properties and 'traversable_materials' in vehicle_properties:
            if material not in vehicle_properties['traversable_materials']:
                return False
        
        # By default, assume excavations are traversable except at very steep angles
        return angle <= 30.0
    
    def get_traversal_cost(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> float:
        """
        Get the cost of traversing a point within the excavation.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for calculating cost
            
        Returns:
            float: Traversal cost
        """
        if not self.contains_point(point):
            return 1.0
        
        # Get slope and hardness
        angle, _ = self.get_slope(point)
        hardness = self.get_hardness(point)
        
        # Base cost factors
        slope_factor = 1.0 + (angle / 10.0)  # Steeper slopes are more costly
        hardness_factor = 1.0 + (hardness / 10.0)  # Harder terrain is more costly
        
        # Combined base cost
        base_cost = slope_factor * hardness_factor
        
        # Adjust for vehicle properties
        if vehicle_properties:
            # Adjust based on vehicle terrain capability
            if 'terrain_capability' in vehicle_properties:
                terrain_capability = vehicle_properties['terrain_capability']
                base_cost *= max(0.5, 1.0 - terrain_capability / 2.0)
        
        return base_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert excavation feature to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        base_dict = super().to_dict()
        
        excavation_dict = {
            'depth': self.depth,
            'base_elevation': self.base_elevation,
            'rim_elevation': self.rim_elevation,
            'profile': self.profile,
            'material': self.material.name
        }
        
        return {**base_dict, **excavation_dict}


class RoadFeature(TerrainFeature):
    """
    Road terrain feature.
    
    Represents a constructed road or path for vehicle travel.
    """
    
    def __init__(self, feature_id: str = None, name: str = None, 
                 path_points: List[Point2D] = None, width: float = 5.0,
                 properties: Dict[str, Any] = None, elevation: float = 0.0,
                 road_type: str = "main"):
        """
        Initialize a road feature.
        
        Args:
            feature_id: Unique identifier for the feature
            name: Name of the feature
            path_points: List of points defining the road centerline
            width: Width of the road in meters
            properties: Additional properties
            elevation: Base elevation of the road
            road_type: Type of road ("main", "secondary", "access")
        """
        # Create a bounding box from the path points
        if path_points and len(path_points) >= 2:
            min_x = min(p.x for p in path_points)
            min_y = min(p.y for p in path_points)
            max_x = max(p.x for p in path_points)
            max_y = max(p.y for p in path_points)
            
            # Expand by road width
            min_x -= width / 2
            min_y -= width / 2
            max_x += width / 2
            max_y += width / 2
            
            bounds = BoundingBox(Point2D(min_x, min_y), Point2D(max_x, max_y))
        else:
            bounds = None
        
        super().__init__(feature_id, name, bounds, properties)
        
        self.path_points = path_points or []
        self.width = width
        self.elevation = elevation
        self.road_type = road_type
        
        # Set road properties based on type
        if road_type == "main":
            self.speed_limit = properties.get('speed_limit', 10.0)  # m/s
            self.hardness = properties.get('hardness', 9.0)  # Very hard surface
        elif road_type == "secondary":
            self.speed_limit = properties.get('speed_limit', 7.0)  # m/s
            self.hardness = properties.get('hardness', 8.0)  # Hard surface
        else:  # access
            self.speed_limit = properties.get('speed_limit', 5.0)  # m/s
            self.hardness = properties.get('hardness', 7.0)  # Moderately hard surface
    
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if a point is on the road.
        
        Args:
            point: Point to check
            
        Returns:
            bool: True if point is on the road
        """
        if not super().contains_point(point):
            return False
        
        # Check distance to any road segment
        half_width = self.width / 2
        
        for i in range(len(self.path_points) - 1):
            segment_start = self.path_points[i]
            segment_end = self.path_points[i + 1]
            
            # Calculate distance from point to this segment
            # Vector from segment_start to segment_end
            segment_vec_x = segment_end.x - segment_start.x
            segment_vec_y = segment_end.y - segment_start.y
            
            # Vector from segment_start to point
            point_vec_x = point.x - segment_start.x
            point_vec_y = point.y - segment_start.y
            
            # Squared length of the segment
            segment_length_squared = segment_vec_x**2 + segment_vec_y**2
            
            # Handle degenerate case where segment is a point
            if segment_length_squared < 1e-10:
                # Distance to segment_start
                distance = math.sqrt(point_vec_x**2 + point_vec_y**2)
                if distance <= half_width:
                    return True
                continue
            
            # Calculate projection parameter
            t = (point_vec_x * segment_vec_x + point_vec_y * segment_vec_y) / segment_length_squared
            
            if t < 0.0:
                # Closest to segment_start
                distance = math.sqrt(point_vec_x**2 + point_vec_y**2)
            elif t > 1.0:
                # Closest to segment_end
                end_vec_x = point.x - segment_end.x
                end_vec_y = point.y - segment_end.y
                distance = math.sqrt(end_vec_x**2 + end_vec_y**2)
            else:
                # Closest to point on segment
                closest_x = segment_start.x + t * segment_vec_x
                closest_y = segment_start.y + t * segment_vec_y
                distance = math.sqrt((point.x - closest_x)**2 + (point.y - closest_y)**2)
            
            if distance <= half_width:
                return True
        
        return False
    
    def get_elevation(self, point: Point2D) -> float:
        """
        Get the elevation of the road at a point.
        
        Args:
            point: Point to get elevation for
            
        Returns:
            float: Elevation in meters
        """
        if not self.contains_point(point):
            return self.elevation
        
        # Roads are typically flat or with slight grade
        # For now, just return base elevation
        return self.elevation
    
    def get_slope(self, point: Point2D) -> Tuple[float, SlopeDirection]:
        """
        Get the slope of the road at a point.
        
        Args:
            point: Point to get slope for
            
        Returns:
            Tuple[float, SlopeDirection]: Slope angle in degrees and direction
        """
        if not self.contains_point(point):
            return 0.0, SlopeDirection.NORTH
        
        # Find the road segment this point is closest to
        min_distance = float('inf')
        closest_segment = None
        
        for i in range(len(self.path_points) - 1):
            segment_start = self.path_points[i]
            segment_end = self.path_points[i + 1]
            
            # Calculate distance from point to this segment
            distance = self._point_to_segment_distance(point, segment_start, segment_end)
            
            if distance < min_distance:
                min_distance = distance
                closest_segment = (segment_start, segment_end)
        
        if closest_segment is None:
            return 0.0, SlopeDirection.NORTH
        
        # Calculate slope along road direction
        start, end = closest_segment
        dx = end.x - start.x
        dy = end.y - start.y
        
        # Determine direction
        if abs(dx) > abs(dy):
            if dx > 0:
                direction = SlopeDirection.EAST
            else:
                direction = SlopeDirection.WEST
        else:
            if dy > 0:
                direction = SlopeDirection.SOUTH
            else:
                direction = SlopeDirection.NORTH
        
        # Roads typically have gentle slopes (less than 10 degrees)
        # For now, assume flat or gentle slope
        return 3.0, direction
    
    def _point_to_segment_distance(self, point: Point2D, segment_start: Point2D, segment_end: Point2D) -> float:
        """Calculate the distance from a point to a line segment."""
        # Vector from segment_start to segment_end
        segment_vec_x = segment_end.x - segment_start.x
        segment_vec_y = segment_end.y - segment_start.y
        
        # Vector from segment_start to point
        point_vec_x = point.x - segment_start.x
        point_vec_y = point.y - segment_start.y
        
        # Squared length of the segment
        segment_length_squared = segment_vec_x**2 + segment_vec_y**2
        
        # Handle degenerate case where segment is a point
        if segment_length_squared < 1e-10:
            return math.sqrt(point_vec_x**2 + point_vec_y**2)
        
        # Calculate projection parameter
        t = (point_vec_x * segment_vec_x + point_vec_y * segment_vec_y) / segment_length_squared
        
        if t < 0.0:
            # Closest to segment_start
            return math.sqrt(point_vec_x**2 + point_vec_y**2)
        elif t > 1.0:
            # Closest to segment_end
            end_vec_x = point.x - segment_end.x
            end_vec_y = point.y - segment_end.y
            return math.sqrt(end_vec_x**2 + end_vec_y**2)
        
        # Closest to point on segment
        closest_x = segment_start.x + t * segment_vec_x
        closest_y = segment_start.y + t * segment_vec_y
        return math.sqrt((point.x - closest_x)**2 + (point.y - closest_y)**2)
    
    def get_hardness(self, point: Point2D) -> float:
        """
        Get the road surface hardness at a point.
        
        Args:
            point: Point to get hardness for
            
        Returns:
            float: Hardness value (0.0 to 10.0)
        """
        if not self.contains_point(point):
            return 5.0  # Default medium hardness
        
        # Roads have high hardness
        return self.hardness
    
    def get_material(self, point: Point2D) -> MaterialType:
        """
        Get the road material at a point.
        
        Args:
            point: Point to get material for
            
        Returns:
            MaterialType: Material type
        """
        if not self.contains_point(point):
            return MaterialType.DIRT
        
        # Roads are typically made of gravel or compacted materials
        if self.road_type == "main":
            return MaterialType.SOLID_ROCK  # Asphalt or concrete
        else:
            return MaterialType.GRAVEL  # Gravel for secondary and access roads
    
    def is_traversable(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> bool:
        """
        Check if a point on the road is traversable.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for checking capability
            
        Returns:
            bool: True if traversable
        """
        # Roads are designed for vehicle travel
        if self.contains_point(point):
            return True
        
        return False
    
    def get_traversal_cost(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> float:
        """
        Get the cost of traversing a point on the road.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for calculating cost
            
        Returns:
            float: Traversal cost
        """
        if not self.contains_point(point):
            return 1.0
        
        # Roads have very low traversal cost
        # Base cost depends on road type
        if self.road_type == "main":
            base_cost = 0.5  # Main roads are fastest
        elif self.road_type == "secondary":
            base_cost = 0.7  # Secondary roads are a bit slower
        else:
            base_cost = 0.9  # Access roads are the slowest
        
        # Adjust based on vehicle properties
        if vehicle_properties and 'max_speed' in vehicle_properties:
            # If vehicle is slower than the speed limit, adjust cost
            speed_ratio = self.speed_limit / vehicle_properties['max_speed']
            base_cost *= min(1.0, speed_ratio)
        
        return base_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert road feature to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        base_dict = super().to_dict()
        
        road_dict = {
            'path_points': [{'x': p.x, 'y': p.y} for p in self.path_points],
            'width': self.width,
            'elevation': self.elevation,
            'road_type': self.road_type,
            'speed_limit': self.speed_limit,
            'hardness': self.hardness
        }
        
        return {**base_dict, **road_dict}


class WaterFeature(TerrainFeature):
    """
    Water terrain feature.
    
    Represents a body of water such as a lake, pond, or puddle.
    """
    
    def __init__(self, feature_id: str = None, name: str = None, 
                 bounds: BoundingBox = None, properties: Dict[str, Any] = None,
                 depth: float = 2.0, base_elevation: float = 0.0,
                 shape_points: List[Point2D] = None):
        """
        Initialize a water feature.
        
        Args:
            feature_id: Unique identifier for the feature
            name: Name of the feature
            bounds: Bounding box of the feature
            properties: Additional properties
            depth: Maximum depth of the water
            base_elevation: Elevation at the bottom
            shape_points: Points defining the shape of the water body
        """
        super().__init__(feature_id, name, bounds, properties)
        
        self.depth = depth
        self.base_elevation = base_elevation
        self.water_elevation = base_elevation + depth
        
        # If shape points provided, use them for custom shape
        self.shape_points = shape_points
        
        # Set water-specific properties
        self.is_fordable = properties.get('is_fordable', False)
        self.max_ford_depth = properties.get('max_ford_depth', 0.5)  # Maximum fordable depth
        self.current_direction = properties.get('current_direction', 0.0)  # Direction in radians
        self.current_speed = properties.get('current_speed', 0.0)  # Speed in m/s
    
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if a point is within the water body.
        
        Args:
            point: Point to check
            
        Returns:
            bool: True if point is inside the water body
        """
        # If custom shape defined, use point-in-polygon test
        if self.shape_points and len(self.shape_points) >= 3:
            return self._point_in_polygon(point, self.shape_points)
        
        # Otherwise use bounding box
        return super().contains_point(point)
    
    def _point_in_polygon(self, point: Point2D, polygon: List[Point2D]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        inside = False
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            
            if ((polygon[i].y > point.y) != (polygon[j].y > point.y)) and \
               (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / 
                (polygon[j].y - polygon[i].y) + polygon[i].x):
                inside = not inside
                
        return inside
    
    def get_elevation(self, point: Point2D) -> float:
        """
        Get the elevation at a point.
        
        For water bodies, this returns the water surface elevation.
        
        Args:
            point: Point to get elevation for
            
        Returns:
            float: Elevation in meters
        """
        if not self.contains_point(point):
            return self.water_elevation
        
        # Return water surface elevation
        return self.water_elevation
    
    def get_water_depth(self, point: Point2D) -> float:
        """
        Get the water depth at a point.
        
        Args:
            point: Point to get depth for
            
        Returns:
            float: Water depth in meters (0 if outside water body)
        """
        if not self.contains_point(point):
            return 0.0
        
        # For simplicity, assume uniform depth
        # In a more complex model, depth could vary based on distance from shore
        return self.depth
    
    def get_slope(self, point: Point2D) -> Tuple[float, SlopeDirection]:
        """
        Get the slope at a point.
        
        Water surfaces are flat, so slope is always zero.
        
        Args:
            point: Point to get slope for
            
        Returns:
            Tuple[float, SlopeDirection]: Slope angle in degrees and direction
        """
        # Water surfaces are flat
        return 0.0, SlopeDirection.NORTH
    
    def get_hardness(self, point: Point2D) -> float:
        """
        Get the hardness at a point.
        
        Water has zero hardness.
        
        Args:
            point: Point to get hardness for
            
        Returns:
            float: Hardness value (0.0 to 10.0)
        """
        if not self.contains_point(point):
            return 5.0  # Default medium hardness
        
        # Water has zero hardness
        return 0.0
    
    def get_material(self, point: Point2D) -> MaterialType:
        """
        Get the material at a point.
        
        Args:
            point: Point to get material for
            
        Returns:
            MaterialType: Material type
        """
        if not self.contains_point(point):
            return MaterialType.DIRT
        
        return MaterialType.WATER
    
    def is_traversable(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> bool:
        """
        Check if a point is traversable.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for checking capability
            
        Returns:
            bool: True if traversable
        """
        if not self.contains_point(point):
            return True
        
        # Check if water is fordable
        if not self.is_fordable:
            return False
        
        # Check water depth against vehicle's wading depth
        depth = self.get_water_depth(point)
        
        if vehicle_properties and 'wading_depth' in vehicle_properties:
            return depth <= vehicle_properties['wading_depth']
        
        # Default: water is traversable if depth is less than max ford depth
        return depth <= self.max_ford_depth
    
    def get_traversal_cost(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> float:
        """
        Get the cost of traversing a point.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for calculating cost
            
        Returns:
            float: Traversal cost
        """
        if not self.contains_point(point):
            return 1.0
        
        # If not traversable, return infinite cost
        if not self.is_traversable(point, vehicle_properties):
            return float('inf')
        
        # Water traversal is slow and costly
        # Base cost depends on depth relative to maximum fordable depth
        depth = self.get_water_depth(point)
        max_depth = self.max_ford_depth
        
        if vehicle_properties and 'wading_depth' in vehicle_properties:
            max_depth = vehicle_properties['wading_depth']
        
        # Cost increases with depth
        depth_factor = depth / max_depth
        base_cost = 2.0 + 3.0 * depth_factor  # Range from 2.0 to 5.0
        
        # Current affects cost
        if self.current_speed > 0.0:
            # Calculate angle between vehicle direction and current
            if vehicle_properties and 'direction' in vehicle_properties:
                vehicle_dir = vehicle_properties['direction']
                current_dir = (math.cos(self.current_direction), math.sin(self.current_direction))
                
                # Dot product to determine if going against current
                dot_product = vehicle_dir[0] * current_dir[0] + vehicle_dir[1] * current_dir[1]
                
                # Going against current is more costly
                current_factor = 1.0 - (dot_product * 0.5)  # Range from 0.5 to 1.5
                base_cost *= current_factor
        
        return base_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert water feature to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        base_dict = super().to_dict()
        
        water_dict = {
            'depth': self.depth,
            'base_elevation': self.base_elevation,
            'water_elevation': self.water_elevation,
            'is_fordable': self.is_fordable,
            'max_ford_depth': self.max_ford_depth,
            'current_direction': self.current_direction,
            'current_speed': self.current_speed
        }
        
        if self.shape_points:
            water_dict['shape_points'] = [{'x': p.x, 'y': p.y} for p in self.shape_points]
        
        return {**base_dict, **water_dict}


class RestrictedAreaFeature(TerrainFeature):
    """
    Restricted area terrain feature.
    
    Represents an area with restricted access, such as a blasting zone,
    no-go area, or private property.
    """
    
    def __init__(self, feature_id: str = None, name: str = None, 
                 bounds: BoundingBox = None, properties: Dict[str, Any] = None,
                 shape_points: List[Point2D] = None,
                 restriction_level: int = 10,
                 access_codes: List[str] = None):
        """
        Initialize a restricted area feature.
        
        Args:
            feature_id: Unique identifier for the feature
            name: Name of the feature
            bounds: Bounding box of the feature
            properties: Additional properties
            shape_points: Points defining the shape of the restricted area
            restriction_level: Restriction level (0-10, 10 being completely restricted)
            access_codes: List of access codes that allow entry
        """
        super().__init__(feature_id, name, bounds, properties)
        
        self.shape_points = shape_points
        self.restriction_level = restriction_level
        self.access_codes = access_codes or []
        
        # Restriction reason
        self.reason = properties.get('reason', 'Restricted Area')
        
        # Effective time period
        self.start_time = properties.get('start_time', None)
        self.end_time = properties.get('end_time', None)
    
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if a point is within the restricted area.
        
        Args:
            point: Point to check
            
        Returns:
            bool: True if point is inside the restricted area
        """
        # If custom shape defined, use point-in-polygon test
        if self.shape_points and len(self.shape_points) >= 3:
            return self._point_in_polygon(point, self.shape_points)
        
        # Otherwise use bounding box
        return super().contains_point(point)
    
    def _point_in_polygon(self, point: Point2D, polygon: List[Point2D]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        inside = False
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            
            if ((polygon[i].y > point.y) != (polygon[j].y > point.y)) and \
               (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / 
                (polygon[j].y - polygon[i].y) + polygon[i].x):
                inside = not inside
                
        return inside
    
    def is_active(self, current_time: datetime = None) -> bool:
        """
        Check if restriction is currently active.
        
        Args:
            current_time: Current time, or None for now
            
        Returns:
            bool: True if restriction is active
        """
        if current_time is None:
            current_time = datetime.now()
        
        # If no time restrictions, always active
        if self.start_time is None and self.end_time is None:
            return True
        
        # Check if within time window
        if self.start_time and current_time < self.start_time:
            return False
        
        if self.end_time and current_time > self.end_time:
            return False
        
        return True
    
    def has_access(self, vehicle_properties: Dict[str, Any] = None) -> bool:
        """
        Check if a vehicle has access to the restricted area.
        
        Args:
            vehicle_properties: Vehicle properties for checking access
            
        Returns:
            bool: True if vehicle has access
        """
        if not vehicle_properties:
            return False
        
        # Check access codes
        vehicle_access_codes = vehicle_properties.get('access_codes', [])
        if set(vehicle_access_codes).intersection(set(self.access_codes)):
            return True
        
        # Check access level
        vehicle_access_level = vehicle_properties.get('access_level', 0)
        if vehicle_access_level >= self.restriction_level:
            return True
        
        return False
    
    def is_traversable(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> bool:
        """
        Check if a point in the restricted area is traversable.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for checking capability
            
        Returns:
            bool: True if traversable
        """
        if not self.contains_point(point):
            return True
        
        # Check if restriction is active
        if not self.is_active():
            return True
        
        # Check if vehicle has access
        if vehicle_properties and self.has_access(vehicle_properties):
            return True
        
        # If full restriction (level 10), not traversable
        if self.restriction_level >= 10:
            return False
        
        # Partial restrictions may be traversable but at high cost
        return self.restriction_level < 8
    
    def get_traversal_cost(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> float:
        """
        Get the cost of traversing a point in the restricted area.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for calculating cost
            
        Returns:
            float: Traversal cost
        """
        if not self.contains_point(point):
            return 1.0
        
        # Check if restriction is active
        if not self.is_active():
            return 1.0
        
        # Check if vehicle has access
        if vehicle_properties and self.has_access(vehicle_properties):
            return 1.0
        
        # Cost increases with restriction level
        if self.restriction_level >= 10:
            return float('inf')  # Not traversable
        
        return 1.0 + (self.restriction_level / 2.0)  # Range from 1.0 to 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert restricted area feature to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        base_dict = super().to_dict()
        
        restricted_dict = {
            'restriction_level': self.restriction_level,
            'access_codes': self.access_codes,
            'reason': self.reason
        }
        
        if self.shape_points:
            restricted_dict['shape_points'] = [{'x': p.x, 'y': p.y} for p in self.shape_points]
        
        if self.start_time:
            restricted_dict['start_time'] = self.start_time.isoformat()
        
        if self.end_time:
            restricted_dict['end_time'] = self.end_time.isoformat()
        
        return {**base_dict, **restricted_dict}