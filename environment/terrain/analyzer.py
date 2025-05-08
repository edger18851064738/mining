"""
Terrain analysis utilities for the mining dispatch system.

Provides tools for analyzing terrain features, identifying traversable paths,
calculating optimal routes, and evaluating operational impacts.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set, Callable, Any
from enum import Enum, auto

from utils.geo.coordinates import Point2D, BoundingBox
from utils.geo.distances import point_to_polygon_distance
from utils.math.vectors import Vector2D
from utils.logger import get_logger

from environment.terrain.features import (
    TerrainFeature, SlopeFeature, HillFeature, ExcavationFeature, 
    RoadFeature, WaterFeature, RestrictedAreaFeature,
    TerrainType, MaterialType, SlopeDirection
)

# Initialize logger
logger = get_logger("terrain_analyzer")


class AnalysisResolution(Enum):
    """Resolution levels for terrain analysis."""
    LOW = auto()      # Fast but less accurate
    MEDIUM = auto()   # Balanced performance
    HIGH = auto()     # Detailed but slower
    ULTRA = auto()    # Highest detail for critical areas


class AnalysisMode(Enum):
    """Analysis modes for different operational focus."""
    TRAVERSABILITY = auto()   # Focus on vehicle movement
    EXCAVATION = auto()       # Focus on mining operations
    SAFETY = auto()           # Focus on hazard identification
    EFFICIENCY = auto()       # Focus on operational efficiency
    ENVIRONMENTAL = auto()    # Focus on environmental impact


class TerrainAnalysisError(Exception):
    """Base exception for terrain analysis errors."""
    pass


class TerrainAnalyzer:
    """
    Analyzer for terrain characteristics and traversability.
    
    Provides tools for analyzing terrain features, computing traversability,
    and supporting path planning decisions based on terrain characteristics.
    """
    
    def __init__(self, features: Optional[List[TerrainFeature]] = None, 
                 resolution: AnalysisResolution = AnalysisResolution.MEDIUM):
        """
        Initialize terrain analyzer.
        
        Args:
            features: List of terrain features to analyze
            resolution: Analysis resolution
        """
        self.features = features or []
        self.resolution = resolution
        self.analysis_cache = {}  # Cache analysis results
        self.bounds = self._calculate_bounds()
    
    def add_feature(self, feature: TerrainFeature) -> None:
        """
        Add a terrain feature to the analyzer.
        
        Args:
            feature: Terrain feature to add
        """
        self.features.append(feature)
        self._update_bounds(feature.bounds)
        
        # Clear cache as features have changed
        self.analysis_cache.clear()
    
    def remove_feature(self, feature_id: str) -> bool:
        """
        Remove a terrain feature by ID.
        
        Args:
            feature_id: ID of the feature to remove
            
        Returns:
            bool: True if feature was removed, False if not found
        """
        for i, feature in enumerate(self.features):
            if feature.feature_id == feature_id:
                self.features.pop(i)
                # Clear cache and recalculate bounds
                self.analysis_cache.clear()
                self.bounds = self._calculate_bounds()
                return True
        return False
    
    def _calculate_bounds(self) -> BoundingBox:
        """Calculate the bounding box containing all features."""
        if not self.features:
            return BoundingBox(Point2D(0, 0), Point2D(0, 0))
            
        min_x = min(feature.bounds.min_point.x for feature in self.features)
        min_y = min(feature.bounds.min_point.y for feature in self.features)
        max_x = max(feature.bounds.max_point.x for feature in self.features)
        max_y = max(feature.bounds.max_point.y for feature in self.features)
        
        return BoundingBox(Point2D(min_x, min_y), Point2D(max_x, max_y))
    
    def _update_bounds(self, feature_bounds: BoundingBox) -> None:
        """Update bounds when adding a new feature."""
        if self.bounds.min_point.x == 0 and self.bounds.max_point.x == 0:
            # First feature, just use its bounds
            self.bounds = feature_bounds
            return
            
        min_x = min(self.bounds.min_point.x, feature_bounds.min_point.x)
        min_y = min(self.bounds.min_point.y, feature_bounds.min_point.y)
        max_x = max(self.bounds.max_point.x, feature_bounds.max_point.x)
        max_y = max(self.bounds.max_point.y, feature_bounds.max_point.y)
        
        self.bounds = BoundingBox(Point2D(min_x, min_y), Point2D(max_x, max_y))
    
    def get_elevation(self, point: Point2D) -> float:
        """
        Get the elevation at a specific point.
        
        Args:
            point: Point to query
            
        Returns:
            float: Elevation in meters
        """
        # Check relevant features at this point
        relevant_features = self.get_features_at_point(point)
        
        if not relevant_features:
            return 0.0  # Default elevation
        
        # Find the feature with highest priority (assumed to be last in the list)
        # This simple approach could be improved with explicit priority handling
        elevations = [feature.get_elevation(point) for feature in relevant_features]
        return max(elevations)  # Return highest elevation
    
    def get_features_at_point(self, point: Point2D) -> List[TerrainFeature]:
        """
        Get all terrain features at a specific point.
        
        Args:
            point: Point to query
            
        Returns:
            List[TerrainFeature]: List of features at this point
        """
        return [feature for feature in self.features if feature.contains_point(point)]
    
    def get_terrain_property(self, point: Point2D, property_name: str) -> Union[float, MaterialType, TerrainType]:
        """
        Get a specific terrain property at a point.
        
        Args:
            point: Point to query
            property_name: Name of the property ('hardness', 'slope', 'material', etc.)
            
        Returns:
            Union[float, MaterialType, TerrainType]: Property value
            
        Raises:
            TerrainAnalysisError: If property name is invalid
        """
        features = self.get_features_at_point(point)
        
        if not features:
            # Default values for various properties
            if property_name == 'hardness':
                return 5.0  # Medium hardness
            elif property_name == 'slope':
                return 0.0  # Flat
            elif property_name == 'material':
                return MaterialType.DIRT
            elif property_name == 'elevation':
                return 0.0
            else:
                raise TerrainAnalysisError(f"Unknown property: {property_name}")
        
        # Get property from all relevant features and determine the result
        if property_name == 'hardness':
            # Use maximum hardness
            return max(feature.get_hardness(point) for feature in features)
        elif property_name == 'slope':
            # Use maximum slope
            slopes = [feature.get_slope(point)[0] for feature in features 
                     if hasattr(feature, 'get_slope')]
            return max(slopes) if slopes else 0.0
        elif property_name == 'material':
            # Use the material from the most specific feature (last in list)
            for feature in reversed(features):
                if hasattr(feature, 'get_material'):
                    return feature.get_material(point)
            return MaterialType.DIRT
        elif property_name == 'elevation':
            return self.get_elevation(point)
        else:
            raise TerrainAnalysisError(f"Unknown property: {property_name}")
    
    def is_traversable(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> bool:
        """
        Check if a point is traversable by a vehicle.
        
        Args:
            point: Point to check
            vehicle_properties: Optional vehicle properties for checking capability
            
        Returns:
            bool: True if traversable
        """
        features = self.get_features_at_point(point)
        
        if not features:
            return True  # Default: traversable if no specific features
        
        # If any feature says it's not traversable, the point is not traversable
        for feature in features:
            if not feature.is_traversable(point, vehicle_properties):
                return False
                
        return True
    
    def get_traversal_cost(self, point: Point2D, vehicle_properties: Dict[str, Any] = None) -> float:
        """
        Get the cost of traversing a specific point.
        
        Args:
            point: Point to query
            vehicle_properties: Optional vehicle properties
            
        Returns:
            float: Traversal cost (higher values mean more difficult)
        """
        features = self.get_features_at_point(point)
        
        if not features:
            return 1.0  # Default cost
        
        # Combine costs from all features
        # Simple approach: use maximum cost
        costs = [feature.get_traversal_cost(point, vehicle_properties) for feature in features]
        return max(costs)
    
    def analyze_region(self, region: BoundingBox, grid_size: float, 
                     analysis_type: str) -> Dict[str, Any]:
        """
        Analyze a region for a specific property on a grid.
        
        Args:
            region: Area to analyze
            grid_size: Grid cell size for analysis
            analysis_type: Type of analysis ('traversability', 'grade', 'slope', etc.)
            
        Returns:
            Dict[str, Any]: Analysis results
            
        Raises:
            TerrainAnalysisError: If analysis type is invalid
        """
        # Check if we have cached results
        cache_key = (region.min_point.x, region.min_point.y, 
                    region.max_point.x, region.max_point.y, 
                    grid_size, analysis_type)
                    
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Calculate grid dimensions
        width = int((region.max_point.x - region.min_point.x) / grid_size) + 1
        height = int((region.max_point.y - region.min_point.y) / grid_size) + 1
        
        # Create grid for analysis
        grid = []
        points = []
        
        for y in range(height):
            row = []
            for x in range(width):
                # Calculate world coordinates
                world_x = region.min_point.x + x * grid_size
                world_y = region.min_point.y + y * grid_size
                point = Point2D(world_x, world_y)
                points.append(point)
                
                # Analyze this point
                if analysis_type == 'traversability':
                    value = 1.0 if self.is_traversable(point) else 0.0
                elif analysis_type == 'elevation':
                    value = self.get_elevation(point)
                elif analysis_type == 'slope':
                    value = self.get_terrain_property(point, 'slope')
                elif analysis_type == 'hardness':
                    value = self.get_terrain_property(point, 'hardness')
                elif analysis_type == 'cost':
                    value = self.get_traversal_cost(point)
                else:
                    raise TerrainAnalysisError(f"Unknown analysis type: {analysis_type}")
                    
                row.append(value)
            grid.append(row)
        
        # Create result
        result = {
            'grid': grid,
            'width': width,
            'height': height,
            'grid_size': grid_size,
            'min_value': min(min(row) for row in grid) if grid else 0.0,
            'max_value': max(max(row) for row in grid) if grid else 0.0,
            'analysis_type': analysis_type,
            'region': {
                'min_x': region.min_point.x,
                'min_y': region.min_point.y,
                'max_x': region.max_point.x,
                'max_y': region.max_point.y
            }
        }
        
        # Cache result
        self.analysis_cache[cache_key] = result
        
        return result
    
    def find_obstacle_free_path(self, start: Point2D, end: Point2D, 
                               vehicle_properties: Dict[str, Any] = None,
                               grid_size: float = 5.0) -> List[Point2D]:
        """
        Find an obstacle-free path from start to end considering terrain.
        
        Uses A* algorithm with terrain-aware cost function.
        
        Args:
            start: Start point
            end: End point
            vehicle_properties: Optional vehicle properties
            grid_size: Grid size for discretization
            
        Returns:
            List[Point2D]: Path from start to end
            
        Raises:
            TerrainAnalysisError: If no path is found
        """
        # Check if points are traversable
        if not self.is_traversable(start, vehicle_properties):
            raise TerrainAnalysisError(f"Start point is not traversable")
        if not self.is_traversable(end, vehicle_properties):
            raise TerrainAnalysisError(f"End point is not traversable")
        
        # Discretize points to grid
        def to_grid(point):
            x = int(point.x / grid_size)
            y = int(point.y / grid_size)
            return (x, y)
            
        def to_world(grid_point):
            x = grid_point[0] * grid_size
            y = grid_point[1] * grid_size
            return Point2D(x, y)
        
        start_grid = to_grid(start)
        end_grid = to_grid(end)
        
        # A* algorithm
        import heapq
        
        # Heuristic function (Euclidean distance)
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        # Get neighbors
        def get_neighbors(point):
            x, y = point
            # 8-connected grid
            neighbors = [
                (x+1, y), (x-1, y), (x, y+1), (x, y-1),
                (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)
            ]
            return [n for n in neighbors if self.is_traversable(to_world(n), vehicle_properties)]
        
        # Get movement cost
        def movement_cost(from_point, to_point):
            # Base cost (1.0 for adjacent, 1.414 for diagonal)
            if abs(from_point[0] - to_point[0]) + abs(from_point[1] - to_point[1]) == 1:
                base_cost = 1.0
            else:
                base_cost = 1.414
                
            # Terrain cost
            world_point = to_world(to_point)
            terrain_cost = self.get_traversal_cost(world_point, vehicle_properties)
            
            return base_cost * terrain_cost
        
        # A* search
        closed_set = set()
        open_set = [(0, start_grid)]  # priority queue: (f_score, point)
        heapq.heapify(open_set)
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: heuristic(start_grid, end_grid)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == end_grid:
                # Reconstruct path
                path = [end]
                while current in came_from:
                    current = came_from[current]
                    if current != start_grid:  # Don't add start yet
                        path.append(to_world(current))
                path.append(start)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            for neighbor in get_neighbors(current):
                if neighbor in closed_set:
                    continue
                    
                tentative_g_score = g_score[current] + movement_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end_grid)
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        raise TerrainAnalysisError(f"No path found from {start} to {end}")
    
    def find_slope_features(self, min_angle: float = 15.0) -> List[SlopeFeature]:
        """
        Find all slope features with angle greater than min_angle.
        
        Args:
            min_angle: Minimum slope angle in degrees
            
        Returns:
            List[SlopeFeature]: List of steep slope features
        """
        return [feature for feature in self.features 
                if isinstance(feature, SlopeFeature) and feature.angle >= min_angle]
    
    def find_excavation_features(self) -> List[ExcavationFeature]:
        """
        Find all excavation features.
        
        Returns:
            List[ExcavationFeature]: List of excavation features
        """
        return [feature for feature in self.features 
                if isinstance(feature, ExcavationFeature)]
    
    def find_road_features(self) -> List[RoadFeature]:
        """
        Find all road features.
        
        Returns:
            List[RoadFeature]: List of road features
        """
        return [feature for feature in self.features 
                if isinstance(feature, RoadFeature)]
    
    def find_feature_by_id(self, feature_id: str) -> Optional[TerrainFeature]:
        """
        Find a feature by its ID.
        
        Args:
            feature_id: Feature ID
            
        Returns:
            Optional[TerrainFeature]: Found feature or None
        """
        for feature in self.features:
            if feature.feature_id == feature_id:
                return feature
        return None
    
    def find_nearest_feature(self, point: Point2D, 
                           feature_type: Optional[type] = None) -> Optional[TerrainFeature]:
        """
        Find the nearest feature to a point.
        
        Args:
            point: Reference point
            feature_type: Optional feature type to filter
            
        Returns:
            Optional[TerrainFeature]: Nearest feature or None
        """
        features = self.features
        if feature_type:
            features = [f for f in features if isinstance(f, feature_type)]
            
        if not features:
            return None
            
        # Find feature with minimum distance
        def distance_to_feature(feature):
            if feature.contains_point(point):
                return 0.0
            
            # Use bounding box first for efficiency
            if (point.x < feature.bounds.min_point.x or 
                point.x > feature.bounds.max_point.x or
                point.y < feature.bounds.min_point.y or
                point.y > feature.bounds.max_point.y):
                
                # Point is outside bounding box, find closest point on box
                closest_x = max(feature.bounds.min_point.x, 
                               min(point.x, feature.bounds.max_point.x))
                closest_y = max(feature.bounds.min_point.y, 
                               min(point.y, feature.bounds.max_point.y))
                return point.distance_to(Point2D(closest_x, closest_y))
            
            # For more complex shapes, use shape-specific distance calculation
            if hasattr(feature, 'distance_to_point'):
                return feature.distance_to_point(point)
            
            # Default: use bounding box center
            return point.distance_to(feature.bounds.center)
        
        return min(features, key=distance_to_feature)
    
    def identify_hazard_areas(self, vehicle_properties: Dict[str, Any] = None) -> List[BoundingBox]:
        """
        Identify areas that might be hazardous for the vehicle.
        
        Args:
            vehicle_properties: Optional vehicle properties
            
        Returns:
            List[BoundingBox]: List of hazardous areas
        """
        hazard_areas = []
        
        # Check for steep slopes
        steep_slopes = self.find_slope_features(20.0)  # Slopes over 20 degrees
        for slope in steep_slopes:
            if vehicle_properties and 'max_slope' in vehicle_properties:
                max_slope = vehicle_properties['max_slope']
                if slope.angle > max_slope:
                    hazard_areas.append(slope.bounds)
            else:
                hazard_areas.append(slope.bounds)
        
        # Check for water features
        water_features = [f for f in self.features if isinstance(f, WaterFeature)]
        for water in water_features:
            if not water.is_fordable:
                hazard_areas.append(water.bounds)
        
        # Check for restricted areas
        restricted_features = [f for f in self.features if isinstance(f, RestrictedAreaFeature)]
        for restricted in restricted_features:
            if restricted.restriction_level >= 8:  # High restriction
                hazard_areas.append(restricted.bounds)
        
        return hazard_areas
    
    def generate_traversability_map(self, region: BoundingBox, grid_size: float,
                                  vehicle_properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a traversability map for the given region.
        
        Args:
            region: Area to analyze
            grid_size: Grid cell size
            vehicle_properties: Optional vehicle properties
            
        Returns:
            Dict[str, Any]: Traversability map
        """
        # Calculate grid dimensions
        width = int((region.max_point.x - region.min_point.x) / grid_size) + 1
        height = int((region.max_point.y - region.min_point.y) / grid_size) + 1
        
        # Create traversability grid
        traversable = [[False for _ in range(width)] for _ in range(height)]
        cost = [[float('inf') for _ in range(width)] for _ in range(height)]
        
        # Check each cell
        for y in range(height):
            for x in range(width):
                # Calculate world coordinates
                world_x = region.min_point.x + x * grid_size
                world_y = region.min_point.y + y * grid_size
                point = Point2D(world_x, world_y)
                
                # Check traversability
                is_traversable = self.is_traversable(point, vehicle_properties)
                traversable[y][x] = is_traversable
                
                # Calculate cost if traversable
                if is_traversable:
                    cost[y][x] = self.get_traversal_cost(point, vehicle_properties)
        
        return {
            'region': {
                'min_x': region.min_point.x,
                'min_y': region.min_point.y,
                'max_x': region.max_point.x,
                'max_y': region.max_point.y
            },
            'grid_size': grid_size,
            'width': width,
            'height': height,
            'traversable': traversable,
            'cost': cost
        }
    
    def get_elevation_profile(self, path: List[Point2D]) -> List[float]:
        """
        Calculate elevation profile along a path.
        
        Args:
            path: List of points in the path
            
        Returns:
            List[float]: Elevation at each point
        """
        return [self.get_elevation(point) for point in path]
    
    def find_optimal_path(self, start: Point2D, end: Point2D, 
                        vehicle_properties: Dict[str, Any],
                        cost_factors: Dict[str, float] = None) -> List[Point2D]:
        """
        Find an optimal path considering multiple cost factors.
        
        Args:
            start: Start point
            end: End point
            vehicle_properties: Vehicle properties
            cost_factors: Weights for different cost factors (distance, slope, etc.)
            
        Returns:
            List[Point2D]: Optimal path
            
        Raises:
            TerrainAnalysisError: If no path is found
        """
        # Default cost factors
        factors = {
            'distance': 1.0,
            'slope': 1.0,
            'hardness': 1.0,
            'elevation_change': 1.0
        }
        
        # Update with provided factors
        if cost_factors:
            factors.update(cost_factors)
        
        # Use A* with custom cost function
        # Implementation similar to find_obstacle_free_path but with weighted costs
        # This would be a more complex implementation that combines multiple cost factors
        
        # For now, use the simpler path finding method
        return self.find_obstacle_free_path(start, end, vehicle_properties)


class TerrainDifferenceAnalyzer:
    """
    Analyzer for comparing terrain features between two snapshots.
    
    Useful for tracking changes over time or identifying recent excavations.
    """
    
    def __init__(self, baseline_analyzer: TerrainAnalyzer, current_analyzer: TerrainAnalyzer):
        """
        Initialize with two terrain analyzers to compare.
        
        Args:
            baseline_analyzer: Baseline terrain analyzer
            current_analyzer: Current terrain analyzer
        """
        self.baseline = baseline_analyzer
        self.current = current_analyzer
    
    def find_elevation_changes(self, region: BoundingBox, grid_size: float, 
                              threshold: float = 0.5) -> Dict[str, Any]:
        """
        Find areas where elevation has changed.
        
        Args:
            region: Area to analyze
            grid_size: Grid cell size
            threshold: Minimum elevation change to consider
            
        Returns:
            Dict[str, Any]: Elevation change analysis
        """
        # Analyze both terrains
        baseline_elevation = self.baseline.analyze_region(region, grid_size, 'elevation')
        current_elevation = self.current.analyze_region(region, grid_size, 'elevation')
        
        # Calculate differences
        width = baseline_elevation['width']
        height = baseline_elevation['height']
        
        changes = []
        total_excavated = 0.0
        total_filled = 0.0
        
        for y in range(height):
            for x in range(width):
                baseline_value = baseline_elevation['grid'][y][x]
                current_value = current_elevation['grid'][y][x]
                
                diff = current_value - baseline_value
                
                if abs(diff) >= threshold:
                    # Calculate world coordinates
                    world_x = region.min_point.x + x * grid_size
                    world_y = region.min_point.y + y * grid_size
                    
                    # Calculate volume (area × height difference)
                    cell_area = grid_size * grid_size
                    volume_change = cell_area * diff
                    
                    changes.append({
                        'x': world_x,
                        'y': world_y,
                        'baseline_elevation': baseline_value,
                        'current_elevation': current_value,
                        'difference': diff,
                        'volume_change': volume_change
                    })
                    
                    # Update totals
                    if diff < 0:
                        total_excavated += abs(volume_change)
                    else:
                        total_filled += volume_change
        
        return {
            'region': region,
            'grid_size': grid_size,
            'changes': changes,
            'total_excavated': total_excavated,
            'total_filled': total_filled,
            'net_volume_change': total_filled - total_excavated,
            'threshold': threshold
        }
    
    def identify_new_features(self) -> List[TerrainFeature]:
        """
        Identify features that exist in current but not in baseline.
        
        Returns:
            List[TerrainFeature]: List of new features
        """
        # Simple approach: check feature IDs
        baseline_ids = {f.feature_id for f in self.baseline.features}
        
        return [f for f in self.current.features if f.feature_id not in baseline_ids]
    
    def identify_modified_features(self) -> List[Tuple[TerrainFeature, TerrainFeature]]:
        """
        Identify features that exist in both but have been modified.
        
        Returns:
            List[Tuple[TerrainFeature, TerrainFeature]]: List of (baseline, current) feature pairs
        """
        # Map features by ID
        baseline_features = {f.feature_id: f for f in self.baseline.features}
        current_features = {f.feature_id: f for f in self.current.features}
        
        # Find common IDs
        common_ids = set(baseline_features.keys()) & set(current_features.keys())
        
        # Find modified features (simple check)
        modified = []
        
        for feature_id in common_ids:
            baseline_feature = baseline_features[feature_id]
            current_feature = current_features[feature_id]
            
            # Check if bounds have changed
            if (baseline_feature.bounds.min_point != current_feature.bounds.min_point or
                baseline_feature.bounds.max_point != current_feature.bounds.max_point):
                modified.append((baseline_feature, current_feature))
                continue
            
            # Check specific feature properties
            if isinstance(baseline_feature, SlopeFeature) and isinstance(current_feature, SlopeFeature):
                if baseline_feature.angle != current_feature.angle:
                    modified.append((baseline_feature, current_feature))
            
            elif isinstance(baseline_feature, ExcavationFeature) and isinstance(current_feature, ExcavationFeature):
                if baseline_feature.depth != current_feature.depth:
                    modified.append((baseline_feature, current_feature))
        
        return modified


class HeatMapGenerator:
    """
    Generate heat maps for various terrain properties.
    
    Creates visualizable heat maps for properties like slope, hardness,
    traversability, and other factors that affect operations.
    """
    
    def __init__(self, terrain_analyzer: TerrainAnalyzer):
        """
        Initialize with a terrain analyzer.
        
        Args:
            terrain_analyzer: Terrain analyzer to use
        """
        self.analyzer = terrain_analyzer
    
    def generate_traversability_heatmap(self, region: BoundingBox, grid_size: float,
                                       vehicle_properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a traversability heat map.
        
        Args:
            region: Area to analyze
            grid_size: Grid cell size
            vehicle_properties: Optional vehicle properties
            
        Returns:
            Dict[str, Any]: Heat map data
        """
        # Get traversability map
        trav_map = self.analyzer.generate_traversability_map(region, grid_size, vehicle_properties)
        
        # Convert to heat map format (0-1 values)
        width = trav_map['width']
        height = trav_map['height']
        
        heatmap = []
        for y in range(height):
            row = []
            for x in range(width):
                if trav_map['traversable'][y][x]:
                    # Scale cost to 0-1 (lower cost = higher traversability)
                    cost = trav_map['cost'][y][x]
                    if cost == float('inf'):
                        value = 0.0  # Not traversable
                    else:
                        # Normalize cost (assumes costs typically range from 1-10)
                        value = max(0.0, min(1.0, 1.0 - (cost - 1.0) / 9.0))
                else:
                    value = 0.0  # Not traversable
                
                row.append(value)
            heatmap.append(row)
        
        return {
            'region': trav_map['region'],
            'grid_size': grid_size,
            'width': width,
            'height': height,
            'heatmap': heatmap,
            'min_value': 0.0,
            'max_value': 1.0,
            'type': 'traversability'
        }
    
    def generate_property_heatmap(self, region: BoundingBox, grid_size: float,
                                property_name: str) -> Dict[str, Any]:
        """
        Generate a heat map for a specific property.
        
        Args:
            region: Area to analyze
            grid_size: Grid cell size
            property_name: Property to visualize ('elevation', 'slope', 'hardness')
            
        Returns:
            Dict[str, Any]: Heat map data
            
        Raises:
            TerrainAnalysisError: If property name is invalid
        """
        try:
            # Use analyzer's region analysis
            analysis = self.analyzer.analyze_region(region, grid_size, property_name)
            
            # Convert to heat map format
            min_value = analysis['min_value']
            max_value = analysis['max_value']
            
            # If min and max are the same, avoid division by zero
            if math.isclose(min_value, max_value):
                range_value = 1.0
            else:
                range_value = max_value - min_value
            
            heatmap = []
            for row in analysis['grid']:
                heatmap_row = [(value - min_value) / range_value for value in row]
                heatmap.append(heatmap_row)
            
            return {
                'region': analysis['region'],
                'grid_size': grid_size,
                'width': analysis['width'],
                'height': analysis['height'],
                'heatmap': heatmap,
                'min_value': min_value,
                'max_value': max_value,
                'type': property_name
            }
            
        except Exception as e:
            raise TerrainAnalysisError(f"Failed to generate heatmap: {str(e)}")
    
    def generate_path_safety_heatmap(self, region: BoundingBox, grid_size: float,
                                   hazard_areas: List[BoundingBox]) -> Dict[str, Any]:
        """
        Generate a safety heat map highlighting high-risk areas.
        
        Args:
            region: Area to analyze
            grid_size: Grid cell size
            hazard_areas: List of hazardous areas
            
        Returns:
            Dict[str, Any]: Heat map data
        """
        # Calculate grid dimensions
        width = int((region.max_point.x - region.min_point.x) / grid_size) + 1
        height = int((region.max_point.y - region.min_point.y) / grid_size) + 1
        
        # Initialize heatmap with all safe
        heatmap = [[1.0 for _ in range(width)] for _ in range(height)]
        
        # Mark hazard areas
        for hazard in hazard_areas:
            # Calculate grid coordinates for hazard
            min_x = int((hazard.min_point.x - region.min_point.x) / grid_size)
            min_y = int((hazard.min_point.y - region.min_point.y) / grid_size)
            max_x = int((hazard.max_point.x - region.min_point.x) / grid_size) + 1
            max_y = int((hazard.max_point.y - region.min_point.y) / grid_size) + 1
            
            # Clamp to grid bounds
            min_x = max(0, min(min_x, width))
            min_y = max(0, min(min_y, height))
            max_x = max(0, min(max_x, width))
            max_y = max(0, min(max_y, height))
            
            # Mark as hazardous (0.0)
            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    if 0 <= y < height and 0 <= x < width:
                        heatmap[y][x] = 0.0
        
        # Add safety gradient around hazards
        safety_heatmap = [row.copy() for row in heatmap]
        gradient_radius = int(20.0 / grid_size)  # 20 meters radius
        
        for y in range(height):
            for x in range(width):
                if heatmap[y][x] < 0.5:  # Hazardous
                    continue
                    
                # Check distance to nearest hazard
                min_distance = float('inf')
                
                for hy in range(max(0, y - gradient_radius), min(height, y + gradient_radius + 1)):
                    for hx in range(max(0, x - gradient_radius), min(width, x + gradient_radius + 1)):
                        if heatmap[hy][hx] < 0.5:  # Hazardous
                            # Manhattan distance
                            distance = abs(x - hx) + abs(y - hy)
                            min_distance = min(min_distance, distance)
                
                if min_distance < gradient_radius:
                    # Apply gradient (linear)
                    safety = min_distance / gradient_radius
                    safety_heatmap[y][x] = safety
        
        return {
            'region': {
                'min_x': region.min_point.x,
                'min_y': region.min_point.y,
                'max_x': region.max_point.x,
                'max_y': region.max_point.y
            },
            'grid_size': grid_size,
            'width': width,
            'height': height,
            'heatmap': safety_heatmap,
            'min_value': 0.0,
            'max_value': 1.0,
            'type': 'safety'
        }