"""
Coordinate transformation utilities for the mining dispatch system.

Provides specialized transformations for the environment module, building
on the base coordinate transformations.
"""

import math
from typing import Tuple, List, Union, Optional, Dict, Any
import numpy as np

from utils.geo.coordinates import Point2D, Point3D
from utils.geo.transforms import CoordinateTransformer

from environment.coordinates.point import EnvironmentPoint2D, EnvironmentPoint3D


class EnvironmentTransformer:
    """
    Environment-specific coordinate transformer.
    
    Extends the base CoordinateTransformer with functionality specific
    to the environment module.
    """
    
    def __init__(self, grid_size: float = 1.0, origin: Point2D = None, 
                 elevation_base: float = 0.0):
        """
        Initialize an environment transformer.
        
        Args:
            grid_size: Size of a grid cell in meters
            origin: Origin point in world coordinates
            elevation_base: Base elevation in meters
        """
        self.base_transformer = CoordinateTransformer(grid_size, origin)
        self.elevation_base = elevation_base
        self.elevation_scale = 1.0
        self.properties_transformers = {}  # Property-specific transformers
    
    def world_to_grid(self, point: Union[Point2D, EnvironmentPoint2D]) -> EnvironmentPoint2D:
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            point: Point in world coordinates
            
        Returns:
            EnvironmentPoint2D: Point in grid coordinates
        """
        base_result = self.base_transformer.meters_to_grid(point)
        
        # Transfer properties if present
        properties = getattr(point, 'properties', None)
        
        # Transform properties if needed
        if properties:
            transformed_properties = self._transform_properties(properties, 'world_to_grid')
            return EnvironmentPoint2D(base_result.x, base_result.y, transformed_properties)
        else:
            return EnvironmentPoint2D(base_result.x, base_result.y)
    
    def grid_to_world(self, point: Union[Point2D, EnvironmentPoint2D]) -> EnvironmentPoint2D:
        """
        Convert grid coordinates to world coordinates.
        
        Args:
            point: Point in grid coordinates
            
        Returns:
            EnvironmentPoint2D: Point in world coordinates
        """
        base_result = self.base_transformer.grid_to_meters(point)
        
        # Transfer properties if present
        properties = getattr(point, 'properties', None)
        
        # Transform properties if needed
        if properties:
            transformed_properties = self._transform_properties(properties, 'grid_to_world')
            return EnvironmentPoint2D(base_result.x, base_result.y, transformed_properties)
        else:
            return EnvironmentPoint2D(base_result.x, base_result.y)
    
    def elevation_to_world(self, elevation: float) -> float:
        """
        Convert elevation value to world coordinates.
        
        Args:
            elevation: Elevation value in local units
            
        Returns:
            float: Elevation in world units
        """
        return self.elevation_base + (elevation * self.elevation_scale)
    
    def world_to_elevation(self, world_elevation: float) -> float:
        """
        Convert world elevation to local elevation units.
        
        Args:
            world_elevation: Elevation in world units
            
        Returns:
            float: Elevation in local units
        """
        if self.elevation_scale == 0:
            return 0.0
        return (world_elevation - self.elevation_base) / self.elevation_scale
    
    def world_to_grid_3d(self, point: Union[Point3D, EnvironmentPoint3D]) -> EnvironmentPoint3D:
        """
        Convert 3D world coordinates to grid coordinates with elevation.
        
        Args:
            point: 3D point in world coordinates
            
        Returns:
            EnvironmentPoint3D: Point in grid coordinates with transformed elevation
        """
        # Transform 2D components
        base_result = self.base_transformer.meters_to_grid(Point2D(point.x, point.y))
        
        # Transform elevation
        elevation = self.world_to_elevation(point.z)
        
        # Transfer properties if present
        properties = getattr(point, 'properties', None)
        
        # Transform properties if needed
        if properties:
            transformed_properties = self._transform_properties(properties, 'world_to_grid')
            return EnvironmentPoint3D(base_result.x, base_result.y, elevation, transformed_properties)
        else:
            return EnvironmentPoint3D(base_result.x, base_result.y, elevation)
    
    def grid_to_world_3d(self, point: Union[Point3D, EnvironmentPoint3D]) -> EnvironmentPoint3D:
        """
        Convert 3D grid coordinates to world coordinates.
        
        Args:
            point: 3D point in grid coordinates
            
        Returns:
            EnvironmentPoint3D: Point in world coordinates with transformed elevation
        """
        # Transform 2D components
        base_result = self.base_transformer.grid_to_meters(Point2D(point.x, point.y))
        
        # Transform elevation
        world_elevation = self.elevation_to_world(point.z)
        
        # Transfer properties if present
        properties = getattr(point, 'properties', None)
        
        # Transform properties if needed
        if properties:
            transformed_properties = self._transform_properties(properties, 'grid_to_world')
            return EnvironmentPoint3D(base_result.x, base_result.y, world_elevation, transformed_properties)
        else:
            return EnvironmentPoint3D(base_result.x, base_result.y, world_elevation)
    
    def register_property_transformer(self, property_name: str, 
                                     world_to_grid_func: callable, 
                                     grid_to_world_func: callable) -> None:
        """
        Register a transformer for a specific property.
        
        Args:
            property_name: Name of the property to transform
            world_to_grid_func: Function to transform property from world to grid
            grid_to_world_func: Function to transform property from grid to world
        """
        self.properties_transformers[property_name] = (world_to_grid_func, grid_to_world_func)
    
    def _transform_properties(self, properties: Dict[str, Any], 
                             direction: str) -> Dict[str, Any]:
        """
        Transform properties based on registered transformers.
        
        Args:
            properties: Properties dictionary
            direction: 'world_to_grid' or 'grid_to_world'
            
        Returns:
            Dict[str, Any]: Transformed properties
        """
        result = properties.copy()
        
        for prop_name, (w2g, g2w) in self.properties_transformers.items():
            if prop_name in result:
                if direction == 'world_to_grid':
                    result[prop_name] = w2g(result[prop_name])
                elif direction == 'grid_to_world':
                    result[prop_name] = g2w(result[prop_name])
        
        return result
    
    def transform_path(self, path: List[Union[Point2D, EnvironmentPoint2D]], 
                      direction: str) -> List[EnvironmentPoint2D]:
        """
        Transform a path between coordinate systems.
        
        Args:
            path: List of points
            direction: 'world_to_grid' or 'grid_to_world'
            
        Returns:
            List[EnvironmentPoint2D]: Transformed path
        """
        if direction == 'world_to_grid':
            return [self.world_to_grid(p) for p in path]
        elif direction == 'grid_to_world':
            return [self.grid_to_world(p) for p in path]
        else:
            raise ValueError(f"Unknown transform direction: {direction}")