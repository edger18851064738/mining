"""
Mining environment implementation for the mining dispatch system.

Provides a specialized environment representation for open-pit mining operations,
including terrain hardness, grades, and mining-specific operational zones.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from enum import Enum, auto
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable

from domain.environment.base import GridEnvironment, EnvironmentError
from utils.geo.coordinates import Point2D, BoundingBox
from utils.geo.distances import point_to_polygon_distance
from utils.logger import get_logger
from utils.config import get_config

# Initialize logger
logger = get_logger("mining_environment")


class ZoneType(Enum):
    """Types of operational zones in a mining environment."""
    LOADING = auto()      # Area where materials are loaded onto vehicles
    UNLOADING = auto()    # Area where materials are unloaded
    PARKING = auto()      # Area for vehicle parking/storage
    CHARGING = auto()     # Area for charging electric vehicles
    MAINTENANCE = auto()  # Area for vehicle maintenance
    EXCAVATION = auto()   # Active excavation area
    DRILLING = auto()     # Drilling operation area
    BLASTING = auto()     # Area prepared for blasting
    RESTRICTED = auto()   # Restricted access area
    TRANSIT = auto()      # High-traffic transit corridor


class TerrainType(Enum):
    """Types of terrain in a mining environment."""
    SOLID_ROCK = auto()    # Solid, unbroken rock
    LOOSE_ROCK = auto()    # Broken, loose rock
    GRAVEL = auto()        # Gravel and small rocks
    DIRT = auto()          # Dirt and soil
    CLAY = auto()          # Clay or mud
    SAND = auto()          # Sandy terrain
    WATER = auto()         # Water-covered area (puddles, ponds)
    ORE_BODY = auto()      # Mineral ore deposit
    WASTE = auto()         # Mine waste material
    ROAD = auto()          # Constructed road surface


class MiningZone:
    """
    Representation of an operational zone in a mining environment.
    
    Zones are polygonal areas with specific operational purposes.
    """
    
    def __init__(self, zone_id: str, zone_type: ZoneType, 
                vertices: List[Point2D], properties: Dict[str, Any] = None):
        """
        Initialize a mining zone.
        
        Args:
            zone_id: Unique identifier for the zone
            zone_type: Type of zone
            vertices: Vertices of the zone polygon
            properties: Additional zone properties
        """
        self.zone_id = zone_id
        self.zone_type = zone_type
        self.vertices = vertices
        self.properties = properties or {}
        
        # Calculate bounding box for quick checks
        min_x = min(v.x for v in vertices)
        min_y = min(v.y for v in vertices)
        max_x = max(v.x for v in vertices)
        max_y = max(v.y for v in vertices)
        self.bounds = BoundingBox(Point2D(min_x, min_y), Point2D(max_x, max_y))
    
    def contains_point(self, point: Point2D) -> bool:
        """
        Check if a point is inside the zone.
        
        Args:
            point: Point to check
            
        Returns:
            bool: True if point is inside zone
        """
        # Quick check using bounding box
        if not (self.bounds.min_point.x <= point.x <= self.bounds.max_point.x and
                self.bounds.min_point.y <= point.y <= self.bounds.max_point.y):
            return False
        
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
    
    def distance_to_point(self, point: Point2D) -> float:
        """
        Calculate the distance from a point to the zone boundary.
        
        Args:
            point: Point to measure from
            
        Returns:
            float: Distance to zone (0 if inside)
        """
        if self.contains_point(point):
            return 0.0
        
        return point_to_polygon_distance(point, self.vertices)
    
    def __repr__(self) -> str:
        """String representation of the zone."""
        return f"MiningZone({self.zone_id}, {self.zone_type.name}, {len(self.vertices)} vertices)"


class OreMaterial:
    """
    Representation of a mineral ore material.
    
    Contains properties like grade, density, and hardness.
    """
    
    def __init__(self, material_id: str, name: str, 
                grade: float = 0.0, density: float = 0.0, hardness: float = 0.0):
        """
        Initialize an ore material.
        
        Args:
            material_id: Unique identifier for the material
            name: Descriptive name
            grade: Ore grade (percentage)
            density: Material density (metric tons/m³)
            hardness: Material hardness (0-10 scale)
        """
        self.material_id = material_id
        self.name = name
        self.grade = grade
        self.density = density
        self.hardness = hardness
    
    def __repr__(self) -> str:
        """String representation of the material."""
        return f"OreMaterial({self.name}, grade={self.grade:.2f}%, hardness={self.hardness:.1f})"


class MiningEnvironment(GridEnvironment):
    """
    Specialized environment for open-pit mining operations.
    
    Features:
    - Mining operational zones (loading, unloading, etc.)
    - Terrain with varying properties (hardness, grade)
    - Ore deposits and material properties
    - Mining-specific path planning considerations
    """
    
    def __init__(self, name: str, 
                bounds: Union[BoundingBox, Tuple[float, float, float, float]],
                resolution: float = 1.0,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize a mining environment.
        
        Args:
            name: Name of the environment
            bounds: Bounding box of the environment
            resolution: Resolution of the grid in meters
            config: Optional configuration dictionary
        """
        super().__init__(name, bounds, resolution)
        
        # Get system configuration
        self.config = config or get_config().map
        
        # Mining-specific grids
        self.hardness_grid = []    # Terrain hardness (0-10 scale)
        self.grade_grid = []       # Ore grade percentage
        self.elevation_grid = []   # Elevation in meters
        self.moisture_grid = []    # Ground moisture (0-1 scale)
        
        # Initialize all grids with default values
        self._init_terrain_grids()
        
        # Mining operational zones
        self.zones = {}
        
        # Materials dictionary
        self.materials = {}
        
        # Create default zones from config
        self._create_default_zones()
        
        # Initialize default materials
        self._init_default_materials()
        
        logger.info(f"Mining environment '{name}' initialized with {self.grid_width}x{self.grid_height} grid")
    
    def _init_terrain_grids(self) -> None:
        """Initialize terrain property grids with default values."""
        # Initialize hardness grid (medium hardness)
        self.hardness_grid = [
            [5.0 for _ in range(self.grid_height)] 
            for _ in range(self.grid_width)
        ]
        
        # Initialize grade grid (zero grade)
        self.grade_grid = [
            [0.0 for _ in range(self.grid_height)] 
            for _ in range(self.grid_width)
        ]
        
        # Initialize elevation grid (flat terrain)
        self.elevation_grid = [
            [0.0 for _ in range(self.grid_height)] 
            for _ in range(self.grid_width)
        ]
        
        # Initialize moisture grid (dry)
        self.moisture_grid = [
            [0.0 for _ in range(self.grid_height)] 
            for _ in range(self.grid_width)
        ]
    
    def _create_default_zones(self) -> None:
        """Create default operational zones based on configuration."""
        # Create zones from key locations
        if hasattr(self.config, 'key_locations'):
            key_locations = self.config.key_locations
            grid_size = self.grid_width * self.resolution
            zone_radius = self.config.safe_radius if hasattr(self.config, 'safe_radius') else 30
            
            # Create parking zone
            if 'parking' in key_locations:
                parking_center = Point2D(
                    key_locations['parking'][0],
                    key_locations['parking'][1]
                )
                self._create_circular_zone(
                    'parking_zone',
                    ZoneType.PARKING,
                    parking_center,
                    zone_radius,
                    {'capacity': 10}
                )
                
                # Also create a charging zone near parking
                charging_center = Point2D(
                    parking_center.x + zone_radius * 0.7,
                    parking_center.y
                )
                self._create_circular_zone(
                    'charging_zone',
                    ZoneType.CHARGING,
                    charging_center,
                    zone_radius * 0.6,
                    {'charging_rate': 50.0}
                )
            
            # Create unloading zone
            if 'unload' in key_locations:
                unload_center = Point2D(
                    key_locations['unload'][0],
                    key_locations['unload'][1]
                )
                self._create_circular_zone(
                    'unload_zone',
                    ZoneType.UNLOADING,
                    unload_center,
                    zone_radius,
                    {'capacity': 5, 'unload_rate': 5000.0}
                )
            
            # Create loading zones
            for i in range(1, 4):
                load_key = f'load{i}'
                if load_key in key_locations:
                    load_center = Point2D(
                        key_locations[load_key][0],
                        key_locations[load_key][1]
                    )
                    self._create_circular_zone(
                        f'load_zone_{i}',
                        ZoneType.LOADING,
                        load_center,
                        zone_radius,
                        {'capacity': 3, 'load_rate': 3000.0, 'material_id': f'ore_{i}'}
                    )
                    
                    # Create surrounding excavation zones
                    excavation_radius = zone_radius * 1.5
                    excavation_center = Point2D(
                        load_center.x + excavation_radius * 0.5 * math.cos(i * 2 * math.pi / 3),
                        load_center.y + excavation_radius * 0.5 * math.sin(i * 2 * math.pi / 3)
                    )
                    self._create_circular_zone(
                        f'excavation_zone_{i}',
                        ZoneType.EXCAVATION,
                        excavation_center,
                        excavation_radius,
                        {'material_id': f'ore_{i}', 'productivity': 2000.0}
                    )
            
            # Create a maintenance zone
            maintenance_center = Point2D(
                grid_size * 0.35,
                grid_size * 0.35
            )
            self._create_circular_zone(
                'maintenance_zone',
                ZoneType.MAINTENANCE,
                maintenance_center,
                zone_radius * 0.8,
                {'capacity': 3, 'repair_rate': 1.0}
            )
            
            # Create a restricted area (e.g., blasting zone)
            restricted_center = Point2D(
                grid_size * 0.65,
                grid_size * 0.35
            )
            self._create_circular_zone(
                'restricted_zone',
                ZoneType.RESTRICTED,
                restricted_center,
                zone_radius * 1.2,
                {'reason': 'Blasting area', 'access_level': 0}
            )
            
            # Create a few transit corridors
            self._create_transit_corridors()
    
    def _create_circular_zone(self, zone_id: str, zone_type: ZoneType, 
                             center: Point2D, radius: float, 
                             properties: Dict[str, Any] = None) -> None:
        """
        Create a circular zone by approximating with a polygon.
        
        Args:
            zone_id: Zone identifier
            zone_type: Type of zone
            center: Center point
            radius: Radius of circle
            properties: Zone properties
        """
        # Create polygon approximation of circle
        num_points = 16  # Number of vertices in the polygon
        vertices = []
        
        for i in range(num_points):
            angle = i * 2 * math.pi / num_points
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            vertices.append(Point2D(x, y))
        
        # Create and add the zone
        zone = MiningZone(zone_id, zone_type, vertices, properties)
        self.zones[zone_id] = zone
        
        # Update terrain properties inside zone
        self._apply_zone_terrain_effects(zone)
    
    def _create_transit_corridors(self) -> None:
        """Create transit corridors connecting key zones."""
        # Find all loading, unloading, and parking zones
        loading_zones = [z for z in self.zones.values() if z.zone_type == ZoneType.LOADING]
        unloading_zones = [z for z in self.zones.values() if z.zone_type == ZoneType.UNLOADING]
        parking_zones = [z for z in self.zones.values() if z.zone_type == ZoneType.PARKING]
        
        # Get centroids of zones
        zone_centroids = {}
        for zone in loading_zones + unloading_zones + parking_zones:
            # Calculate zone centroid
            x_sum = sum(v.x for v in zone.vertices)
            y_sum = sum(v.y for v in zone.vertices)
            centroid = Point2D(x_sum / len(zone.vertices), y_sum / len(zone.vertices))
            zone_centroids[zone.zone_id] = centroid
        
        # Create corridors between zones
        corridor_width = self.config.safe_radius * 0.5 if hasattr(self.config, 'safe_radius') else 15
        
        # Connect all loading zones to unloading zones
        for i, loading_zone in enumerate(loading_zones):
            for j, unloading_zone in enumerate(unloading_zones):
                self._create_corridor(
                    f"corridor_load{i+1}_unload{j+1}",
                    zone_centroids[loading_zone.zone_id],
                    zone_centroids[unloading_zone.zone_id],
                    corridor_width
                )
        
        # Connect parking to all other zones
        if parking_zones:
            parking_centroid = zone_centroids[parking_zones[0].zone_id]
            
            for i, loading_zone in enumerate(loading_zones):
                self._create_corridor(
                    f"corridor_parking_load{i+1}",
                    parking_centroid,
                    zone_centroids[loading_zone.zone_id],
                    corridor_width
                )
                
            for j, unloading_zone in enumerate(unloading_zones):
                self._create_corridor(
                    f"corridor_parking_unload{j+1}",
                    parking_centroid,
                    zone_centroids[unloading_zone.zone_id],
                    corridor_width
                )
    
    def _create_corridor(self, corridor_id: str, start: Point2D, end: Point2D, 
                        width: float) -> None:
        """
        Create a transit corridor between two points.
        
        Args:
            corridor_id: Corridor identifier
            start: Start point
            end: End point
            width: Corridor width
        """
        # Vector from start to end
        dx = end.x - start.x
        dy = end.y - start.y
        
        # Normalized perpendicular vector
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1e-6:
            return  # Points too close
            
        nx = -dy / length
        ny = dx / length
        
        # Create corridor polygon
        half_width = width / 2
        vertices = [
            Point2D(start.x + nx * half_width, start.y + ny * half_width),
            Point2D(start.x - nx * half_width, start.y - ny * half_width),
            Point2D(end.x - nx * half_width, end.y - ny * half_width),
            Point2D(end.x + nx * half_width, end.y + ny * half_width)
        ]
        
        # Create and add the zone
        zone = MiningZone(corridor_id, ZoneType.TRANSIT, vertices, {
            'speed_limit': 8.0,
            'two_way': True
        })
        self.zones[corridor_id] = zone
        
        # Improve terrain in corridor (make it road-like)
        self._apply_zone_terrain_effects(zone, terrain_type=TerrainType.ROAD)
    
    def _init_default_materials(self) -> None:
        """Initialize default material types."""
        # Add basic materials
        self.materials['waste'] = OreMaterial(
            'waste', 'Waste Rock',
            grade=0.0, density=2.8, hardness=6.0
        )
        
        # Add ore types
        for i in range(1, 4):
            self.materials[f'ore_{i}'] = OreMaterial(
                f'ore_{i}', f'Ore Type {i}',
                grade=1.0 + i,  # Different grades
                density=3.0 + i * 0.2,
                hardness=5.0 + i * 0.5
            )
    
    def _apply_zone_terrain_effects(self, zone: MiningZone, 
                                   terrain_type: Optional[TerrainType] = None) -> None:
        """
        Apply terrain modifications within a zone.
        
        Args:
            zone: Zone to modify terrain in
            terrain_type: Optional terrain type to apply
        """
        # Get zone bounds in grid coordinates
        min_x_grid = max(0, int((zone.bounds.min_point.x - self.origin.x) / self.resolution))
        min_y_grid = max(0, int((zone.bounds.min_point.y - self.origin.y) / self.resolution))
        max_x_grid = min(self.grid_width - 1, int((zone.bounds.max_point.x - self.origin.x) / self.resolution))
        max_y_grid = min(self.grid_height - 1, int((zone.bounds.max_point.y - self.origin.y) / self.resolution))
        
        # Iterate over grid cells in bounding box
        for x in range(min_x_grid, max_x_grid + 1):
            for y in range(min_y_grid, max_y_grid + 1):
                # Get world coordinates of cell center
                world_x = self.origin.x + (x + 0.5) * self.resolution
                world_y = self.origin.y + (y + 0.5) * self.resolution
                point = Point2D(world_x, world_y)
                
                # Check if point is in zone
                if zone.contains_point(point):
                    # Apply terrain modifications based on zone type
                    self._modify_terrain_at_point(x, y, zone.zone_type, terrain_type)
    
    def _modify_terrain_at_point(self, grid_x: int, grid_y: int, 
                                zone_type: ZoneType, 
                                terrain_type: Optional[TerrainType] = None) -> None:
        """
        Modify terrain properties at a grid point based on zone type.
        
        Args:
            grid_x: Grid x-coordinate
            grid_y: Grid y-coordinate
            zone_type: Type of zone
            terrain_type: Optional specific terrain type
        """
        # Apply terrain type specific changes
        if terrain_type == TerrainType.ROAD:
            # Roads are hard, flat, and dry
            self.hardness_grid[grid_x][grid_y] = 8.0
            self.elevation_grid[grid_x][grid_y] = 0.0
            self.moisture_grid[grid_x][grid_y] = 0.0
            self.grade_grid[grid_x][grid_y] = 0.0
            # Make sure roads are traversable
            self.grid[grid_x][grid_y] = True
            return
            
        # Apply zone-specific changes
        if zone_type == ZoneType.LOADING:
            # Loading zones have medium hardness, slightly raised
            self.hardness_grid[grid_x][grid_y] = 6.0
            self.elevation_grid[grid_x][grid_y] = 1.0
            self.moisture_grid[grid_x][grid_y] = 0.1
            # Higher grade near loading zones
            self.grade_grid[grid_x][grid_y] = 3.0 + random.uniform(-0.5, 0.5)
            
        elif zone_type == ZoneType.UNLOADING:
            # Unloading zones have high hardness, raised platform
            self.hardness_grid[grid_x][grid_y] = 8.0
            self.elevation_grid[grid_x][grid_y] = 2.0
            self.moisture_grid[grid_x][grid_y] = 0.0
            
        elif zone_type == ZoneType.PARKING or zone_type == ZoneType.CHARGING:
            # Parking areas are flat, hard
            self.hardness_grid[grid_x][grid_y] = 9.0
            self.elevation_grid[grid_x][grid_y] = 0.0
            self.moisture_grid[grid_x][grid_y] = 0.0
            
        elif zone_type == ZoneType.EXCAVATION:
            # Excavation areas have variable hardness, uneven
            self.hardness_grid[grid_x][grid_y] = 4.0 + random.uniform(-1.0, 1.0)
            self.elevation_grid[grid_x][grid_y] = -3.0 + random.uniform(-2.0, 2.0)
            self.moisture_grid[grid_x][grid_y] = 0.2 + random.uniform(0.0, 0.3)
            # Higher grade in excavation areas
            self.grade_grid[grid_x][grid_y] = 5.0 + random.uniform(-1.0, 1.0)
            
        elif zone_type == ZoneType.RESTRICTED:
            # Restricted areas often have obstacles
            self.hardness_grid[grid_x][grid_y] = 7.0
            self.elevation_grid[grid_x][grid_y] = random.uniform(-5.0, 5.0)
            self.moisture_grid[grid_x][grid_y] = random.uniform(0.0, 0.8)
            
            # Add obstacles with 40% probability
            if random.random() < 0.4:
                self.grid[grid_x][grid_y] = False
            
        elif zone_type == ZoneType.TRANSIT:
            # Transit corridors are smooth and hard
            self.hardness_grid[grid_x][grid_y] = 8.5
            self.elevation_grid[grid_x][grid_y] = 0.0
            self.moisture_grid[grid_x][grid_y] = 0.0
            # Make sure transit corridors are traversable
            self.grid[grid_x][grid_y] = True
    
    def create_terrain_features(self) -> None:
        """Generate realistic terrain features with varying properties."""
        # Generate some hills and valleys
        self._generate_elevation_features()
        
        # Generate ore deposits
        self._generate_ore_deposits()
        
        # Add some random moisture
        self._generate_moisture_variation()
        
        # Generate hardness variation
        self._apply_hardness_variation()
        
        # Add paths between key locations
        self._create_additional_paths()
    
    def _generate_elevation_features(self) -> None:
        """Generate elevation features like hills and valleys."""
        # Use simplex noise or similar to generate realistic terrain
        try:
            from noise import snoise2
            
            # Generate base elevation noise
            scale = 0.01  # Scale factor for noise (smaller = larger features)
            octaves = 6   # Number of octaves (more = more detail)
            persistence = 0.6  # How much each octave contributes
            
            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    # Generate noise value
                    nx = x / self.grid_width - 0.5
                    ny = y / self.grid_height - 0.5
                    elevation = snoise2(nx * scale, ny * scale, octaves, persistence)
                    
                    # Scale to reasonable elevation range (-10 to 10 meters)
                    elevation *= 10.0
                    
                    # Apply elevation, but only if not already modified
                    if self.elevation_grid[x][y] == 0.0:
                        self.elevation_grid[x][y] = elevation
        except ImportError:
            # Fallback if noise module not available
            logger.warning("Noise module not available. Using simplified terrain generation.")
            
            # Create a few random hills and valleys
            num_features = int(min(self.grid_width, self.grid_height) / 10)
            
            for _ in range(num_features):
                # Random center
                center_x = random.randint(0, self.grid_width - 1)
                center_y = random.randint(0, self.grid_height - 1)
                
                # Random properties
                radius = random.randint(5, min(self.grid_width, self.grid_height) // 4)
                height = random.uniform(-10.0, 10.0)
                
                # Apply elevation feature
                for x in range(max(0, center_x - radius), min(self.grid_width, center_x + radius)):
                    for y in range(max(0, center_y - radius), min(self.grid_height, center_y + radius)):
                        # Distance from center
                        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                        
                        if dist < radius:
                            # Apply smooth falloff from center
                            factor = (1 - dist/radius)**2
                            self.elevation_grid[x][y] += height * factor
    
    def _generate_ore_deposits(self) -> None:
        """Generate ore deposits with varying grade."""
        # Generate a few ore deposits
        num_deposits = random.randint(3, 6)
        
        for i in range(num_deposits):
            # Random center
            center_x = random.randint(0, self.grid_width - 1)
            center_y = random.randint(0, self.grid_height - 1)
            
            # Random properties
            radius = random.randint(3, min(self.grid_width, self.grid_height) // 6)
            max_grade = random.uniform(2.0, 8.0)
            
            # Select material type
            material_id = random.choice(list(self.materials.keys()))
            
            # Apply ore deposit
            for x in range(max(0, center_x - radius), min(self.grid_width, center_x + radius)):
                for y in range(max(0, center_y - radius), min(self.grid_height, center_y + radius)):
                    # Distance from center
                    dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    if dist < radius:
                        # Apply smooth falloff from center
                        factor = (1 - dist/radius)**2
                        self.grade_grid[x][y] = max(
                            self.grade_grid[x][y],
                            max_grade * factor
                        )
    
    def _generate_moisture_variation(self) -> None:
        """Generate moisture variation across the terrain."""
        # Add some random moisture patches
        num_moisture_areas = random.randint(2, 5)
        
        for _ in range(num_moisture_areas):
            # Random center
            center_x = random.randint(0, self.grid_width - 1)
            center_y = random.randint(0, self.grid_height - 1)
            
            # Random properties
            radius = random.randint(5, min(self.grid_width, self.grid_height) // 5)
            max_moisture = random.uniform(0.4, 0.9)
            
            # Apply moisture
            for x in range(max(0, center_x - radius), min(self.grid_width, center_x + radius)):
                for y in range(max(0, center_y - radius), min(self.grid_height, center_y + radius)):
                    # Distance from center
                    dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    if dist < radius:
                        # Apply smooth falloff from center
                        factor = (1 - dist/radius)**2
                        self.moisture_grid[x][y] = max(
                            self.moisture_grid[x][y],
                            max_moisture * factor
                        )
                        
                        # High moisture reduces hardness
                        moisture = self.moisture_grid[x][y]
                        if moisture > 0.3:
                            self.hardness_grid[x][y] *= (1.0 - (moisture - 0.3))
    
    def _apply_hardness_variation(self) -> None:
        """Apply hardness variation based on other terrain properties."""
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                # Base hardness
                base_hardness = self.hardness_grid[x][y]
                
                # Adjust hardness based on grade (higher grade = harder material)
                grade_factor = 1.0 + (self.grade_grid[x][y] / 20.0)
                
                # Adjust hardness based on moisture (more moisture = softer)
                moisture_factor = 1.0 - (self.moisture_grid[x][y] * 0.5)
                
                # Adjust hardness based on elevation (higher = slightly harder)
                elevation = self.elevation_grid[x][y]
                elevation_factor = 1.0 + (elevation / 100.0) if elevation > 0 else 1.0
                
                # Apply combined factors
                self.hardness_grid[x][y] = base_hardness * grade_factor * moisture_factor * elevation_factor
                
                # Clamp to valid range
                self.hardness_grid[x][y] = max(1.0, min(10.0, self.hardness_grid[x][y]))
    
    def _create_additional_paths(self) -> None:
        """Create additional paths between key locations."""
        # This would create additional minor paths that aren't main transit corridors
        # Implementation depends on specific requirements
        pass
    
    def get_terrain_property(self, point: Union[Point2D, Tuple[float, float]], 
                           property_name: str) -> float:
        """
        Get a terrain property at a specific point.
        
        Args:
            point: Point to query
            property_name: Name of property ('hardness', 'grade', 'elevation', 'moisture')
            
        Returns:
            float: Property value
            
        Raises:
            ValueError: If property name is invalid
        """
        # Convert point to Point2D if necessary
        if not isinstance(point, Point2D):
            point = Point2D(*point)
        
        try:
            # Convert to grid coordinates
            grid_x, grid_y = self._point_to_grid(point)
            
            # Return requested property
            if property_name == 'hardness':
                return self.hardness_grid[grid_x][grid_y]
            elif property_name == 'grade':
                return self.grade_grid[grid_x][grid_y]
            elif property_name == 'elevation':
                return self.elevation_grid[grid_x][grid_y]
            elif property_name == 'moisture':
                return self.moisture_grid[grid_x][grid_y]
            else:
                raise ValueError(f"Unknown terrain property: {property_name}")
        except (IndexError, ValueError):
            # Default values for out-of-bounds or invalid points
            if property_name == 'hardness':
                return 5.0
            elif property_name == 'grade':
                return 0.0
            elif property_name == 'elevation':
                return 0.0
            elif property_name == 'moisture':
                return 0.0
            else:
                raise ValueError(f"Unknown terrain property: {property_name}")
    
    def set_terrain_property(self, point: Union[Point2D, Tuple[float, float]], 
                           property_name: str, value: float) -> None:
        """
        Set a terrain property at a specific point.
        
        Args:
            point: Point to modify
            property_name: Name of property ('hardness', 'grade', 'elevation', 'moisture')
            value: New property value
            
        Raises:
            ValueError: If property name is invalid
        """
        # Convert point to Point2D if necessary
        if not isinstance(point, Point2D):
            point = Point2D(*point)
        
        try:
            # Convert to grid coordinates
            grid_x, grid_y = self._point_to_grid(point)
            
            # Set requested property
            if property_name == 'hardness':
                self.hardness_grid[grid_x][grid_y] = value
            elif property_name == 'grade':
                self.grade_grid[grid_x][grid_y] = value
            elif property_name == 'elevation':
                self.elevation_grid[grid_x][grid_y] = value
            elif property_name == 'moisture':
                self.moisture_grid[grid_x][grid_y] = value
            else:
                raise ValueError(f"Unknown terrain property: {property_name}")
        except (IndexError, ValueError):
            # Ignore out-of-bounds points
            pass
    
    def add_zone(self, zone: MiningZone) -> None:
        """
        Add a new operational zone to the environment.
        
        Args:
            zone: Zone to add
        """
        if zone.zone_id in self.zones:
            logger.warning(f"Replacing existing zone with ID {zone.zone_id}")
            
        self.zones[zone.zone_id] = zone
        self._apply_zone_terrain_effects(zone)
    
    def get_zones_by_type(self, zone_type: ZoneType) -> List[MiningZone]:
        """
        Get all zones of a specific type.
        
        Args:
            zone_type: Type of zones to retrieve
            
        Returns:
            List[MiningZone]: List of matching zones
        """
        return [zone for zone in self.zones.values() if zone.zone_type == zone_type]
    
    def get_zones_containing_point(self, point: Point2D) -> List[MiningZone]:
        """
        Get all zones containing a point.
        
        Args:
            point: Point to check
            
        Returns:
            List[MiningZone]: List of zones containing the point
        """
        return [zone for zone in self.zones.values() if zone.contains_point(point)]
    
    def find_nearest_zone(self, point: Point2D, zone_type: Optional[ZoneType] = None) -> Optional[MiningZone]:
        """
        Find the nearest zone to a point.
        
        Args:
            point: Reference point
            zone_type: Optional type filter
            
        Returns:
            Optional[MiningZone]: Nearest zone or None if none found
        """
        # Filter zones by type if specified
        candidate_zones = self.zones.values()
        if zone_type is not None:
            candidate_zones = [z for z in candidate_zones if z.zone_type == zone_type]
        
        if not candidate_zones:
            return None
        
        # Find zone with minimum distance
        return min(candidate_zones, key=lambda z: z.distance_to_point(point))
    
    def is_traversable(self, point: Union[Point2D, Tuple[float, float]], 
                     vehicle=None) -> bool:
        """
        Check if a point is traversable by a vehicle.
        
        Takes into account obstacles, terrain hardness, and vehicle capabilities.
        
        Args:
            point: Point to check
            vehicle: Vehicle to check traversability for
            
        Returns:
            bool: True if point is traversable, False otherwise
        """
        # First check the base traversability (obstacle grid)
        if not super().is_traversable(point, vehicle):
            return False
        
        # If no vehicle specified, just use base traversability
        if vehicle is None:
            return True
        
        # Check vehicle-specific constraints
        try:
            # Check terrain hardness
            hardness = self.get_terrain_property(point, 'hardness')
            
            # Check if vehicle can handle this terrain hardness
            if hasattr(vehicle, 'min_hardness'):
                if hardness < vehicle.min_hardness:
                    return False
                    
            # Check restricted zones
            point_obj = Point2D(*point) if not isinstance(point, Point2D) else point
            restricted_zones = [z for z in self.get_zones_containing_point(point_obj) 
                              if z.zone_type == ZoneType.RESTRICTED]
            
            if restricted_zones:
                # Vehicle can't traverse restricted zones
                return False
            
            # Additional checks can be added here for other terrain properties
            # or vehicle capabilities
            
            return True
            
        except (ValueError, AttributeError):
            # If any error occurs, use base traversability as fallback
            return True
    
    def calculate_traversal_cost(self, point: Union[Point2D, Tuple[float, float]], 
                              vehicle=None) -> float:
        """
        Calculate the cost of traversing a point for a vehicle.
        
        Args:
            point: Point to check
            vehicle: Vehicle to calculate cost for
            
        Returns:
            float: Traversal cost (higher = more difficult)
        """
        # Base cost
        base_cost = 1.0
        
        # If not traversable, return infinite cost
        if not self.is_traversable(point, vehicle):
            return float('inf')
        
        try:
            # Get terrain properties
            hardness = self.get_terrain_property(point, 'hardness')
            elevation = self.get_terrain_property(point, 'elevation')
            moisture = self.get_terrain_property(point, 'moisture')
            
            # Hardness cost (harder terrain is better to a point, then worse)
            optimal_hardness = 7.0  # Optimal hardness for roads
            hardness_diff = abs(hardness - optimal_hardness)
            hardness_cost = 1.0 + (hardness_diff / 10.0)
            
            # Moisture cost (wetter terrain is worse)
            moisture_cost = 1.0 + (moisture * 1.5)
            
            # Elevation change cost (steep terrain is worse)
            elevation_cost = 1.0
            # We'd need to calculate slope based on neighboring cells
            # This is simplified for now
            
            # Calculate combined cost
            combined_cost = base_cost * hardness_cost * moisture_cost * elevation_cost
            
            # Check for zones with special costs
            point_obj = Point2D(*point) if not isinstance(point, Point2D) else point
            zones = self.get_zones_containing_point(point_obj)
            
            for zone in zones:
                if zone.zone_type == ZoneType.TRANSIT:
                    # Transit corridors have reduced cost
                    combined_cost *= 0.7
                elif zone.zone_type in (ZoneType.LOADING, ZoneType.UNLOADING, ZoneType.PARKING):
                    # Operational zones have slightly reduced cost
                    combined_cost *= 0.85
                elif zone.zone_type == ZoneType.EXCAVATION:
                    # Excavation zones have increased cost
                    combined_cost *= 1.3
            
            return combined_cost
            
        except (ValueError, AttributeError):
            # If any error occurs, return base cost
            return base_cost
    
    def find_path(self, start: Union[Point2D, Tuple[float, float]],
                end: Union[Point2D, Tuple[float, float]],
                vehicle=None) -> List[Point2D]:
        """
        Find a path from start to end, considering mining-specific constraints.
        
        Override of base find_path with mining-specific cost functions.
        
        Args:
            start: Start point
            end: End point
            vehicle: Vehicle to find path for
            
        Returns:
            List[Point2D]: Path from start to end
            
        Raises:
            EnvironmentError: If no path is found
        """
        # We'll use the A* implementation from GridEnvironment
        # but with our custom traversability and cost functions
        
        # First, check if HybridPathPlanner is being used externally
        if hasattr(self, 'dispatch') and self.dispatch is not None:
            if hasattr(self.dispatch, 'planner'):
                # Try using the dispatch system's planner
                try:
                    planner = self.dispatch.planner
                    if hasattr(planner, 'plan_path'):
                        return planner.plan_path(start, end, vehicle)
                except Exception as e:
                    logger.warning(f"Error using dispatch planner: {str(e)}")
        
        # If the above fails, use the base implementation
        # with mining-specific costs
        try:
            return super().find_path(start, end, vehicle)
        except Exception as e:
            logger.warning(f"Base A* path finding failed: {str(e)}")
            
            # Fallback to simpler direct path if A* fails
            if isinstance(start, tuple):
                start = Point2D(*start)
            if isinstance(end, tuple):
                end = Point2D(*end)
                
            return [start, end]
    
    def generate_random_environment(self) -> None:
        """Generate a randomized mining environment."""
        # Clear any existing data
        self.grid = [[True for _ in range(self.grid_height)] for _ in range(self.grid_width)]
        self._init_terrain_grids()
        self.zones = {}
        
        # Generate terrain features
        self.create_terrain_features()
        
        # Generate random obstacles
        self._generate_random_obstacles()
        
        # Create operational zones
        self._create_default_zones()
    
    def _generate_random_obstacles(self) -> None:
        """Generate random obstacles throughout the environment."""
        # Determine obstacle density
        density = self.config.obstacle_density if hasattr(self.config, 'obstacle_density') else 0.15
        
        # Get safe radius around key locations
        safe_radius = self.config.safe_radius if hasattr(self.config, 'safe_radius') else 30
        
        # Get key locations
        key_points = []
        if hasattr(self.config, 'key_locations'):
            for location in self.config.key_locations.values():
                key_points.append(Point2D(location[0], location[1]))
        
        # Generate obstacles
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                # Check if we're near a key location
                grid_point = self._grid_to_point(x, y)
                
                # Skip if point is near a key location
                if any(grid_point.distance_to(key_point) <= safe_radius for key_point in key_points):
                    continue
                
                # Random obstacle with given density
                if random.random() < density:
                    self.grid[x][y] = False  # Mark as obstacle
                    
                    # Make ground harder at obstacle locations
                    self.hardness_grid[x][y] = min(10.0, self.hardness_grid[x][y] + 2.0)
                    
                    # Apply random elevation change
                    self.elevation_grid[x][y] += random.uniform(-3.0, 3.0)
    
    def visualize_environment(self) -> dict:
        """
        Generate visualization data for the environment.
        
        Returns:
            dict: Dictionary of visualization data
        """
        # This would return data suitable for visualization in a UI
        # Implementation depends on the visualization requirements
        
        # Basic visualization data
        visualization_data = {
            'grid_size': (self.grid_width, self.grid_height),
            'resolution': self.resolution,
            'bounds': {
                'min_x': self.bounds.min_point.x,
                'min_y': self.bounds.min_point.y,
                'max_x': self.bounds.max_point.x,
                'max_y': self.bounds.max_point.y
            },
            'obstacles': [],
            'zones': {},
            'terrain': {
                'hardness': [],
                'elevation': [],
                'grade': [],
                'moisture': []
            }
        }
        
        # Extract obstacle positions
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if not self.grid[x][y]:  # If cell is obstacle
                    world_point = self._grid_to_point(x, y)
                    visualization_data['obstacles'].append({
                        'x': world_point.x,
                        'y': world_point.y
                    })
        
        # Extract zone data
        for zone_id, zone in self.zones.items():
            visualization_data['zones'][zone_id] = {
                'type': zone.zone_type.name,
                'vertices': [{'x': v.x, 'y': v.y} for v in zone.vertices],
                'properties': zone.properties
            }
        
        # Extract terrain samples (downsampled for visualization)
        sample_rate = max(1, min(self.grid_width, self.grid_height) // 100)
        for x in range(0, self.grid_width, sample_rate):
            for y in range(0, self.grid_height, sample_rate):
                world_point = self._grid_to_point(x, y)
                
                visualization_data['terrain']['hardness'].append({
                    'x': world_point.x,
                    'y': world_point.y,
                    'value': self.hardness_grid[x][y]
                })
                
                visualization_data['terrain']['elevation'].append({
                    'x': world_point.x,
                    'y': world_point.y,
                    'value': self.elevation_grid[x][y]
                })
                
                visualization_data['terrain']['grade'].append({
                    'x': world_point.x,
                    'y': world_point.y,
                    'value': self.grade_grid[x][y]
                })
                
                visualization_data['terrain']['moisture'].append({
                    'x': world_point.x,
                    'y': world_point.y,
                    'value': self.moisture_grid[x][y]
                })
        
        return visualization_data