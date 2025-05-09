"""
Point class implementations for the mining dispatch system.

These are wrapper classes around the utils.geo.coordinates Point classes,
providing additional functionality specific to the environment module.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import math
from typing import Tuple, List, Union, Optional, Any

from utils.geo.coordinates import Point2D as BasePoint2D
from utils.geo.coordinates import Point3D as BasePoint3D


class EnvironmentPoint2D(BasePoint2D):
    """
    Extended 2D point for environment-specific operations.
    
    This class extends the base Point2D with additional functionality
    needed by the environment module.
    """
    
    def __init__(self, x: float, y: float, properties: dict = None):
        """
        Initialize an environment point.
        
        Args:
            x: x-coordinate
            y: y-coordinate
            properties: Optional properties dictionary
        """
        super().__init__(x, y)
        self.properties = properties or {}
    
    def with_property(self, key: str, value: Any) -> 'EnvironmentPoint2D':
        """
        Create a new point with an added property.
        
        Args:
            key: Property key
            value: Property value
            
        Returns:
            EnvironmentPoint2D: New point with property added
        """
        new_point = EnvironmentPoint2D(self.x, self.y, self.properties.copy())
        new_point.properties[key] = value
        return new_point
    
    def with_properties(self, properties: dict) -> 'EnvironmentPoint2D':
        """
        Create a new point with added properties.
        
        Args:
            properties: Properties to add
            
        Returns:
            EnvironmentPoint2D: New point with properties added
        """
        new_properties = self.properties.copy()
        new_properties.update(properties)
        return EnvironmentPoint2D(self.x, self.y, new_properties)
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a property value.
        
        Args:
            key: Property key
            default: Default value if property doesn't exist
            
        Returns:
            Any: Property value or default
        """
        return self.properties.get(key, default)
    
    def has_property(self, key: str) -> bool:
        """
        Check if point has a property.
        
        Args:
            key: Property key
            
        Returns:
            bool: True if property exists
        """
        return key in self.properties
    
    @classmethod
    def from_base(cls, point: BasePoint2D, properties: dict = None) -> 'EnvironmentPoint2D':
        """
        Create from a base Point2D.
        
        Args:
            point: Base Point2D object
            properties: Optional properties dictionary
            
        Returns:
            EnvironmentPoint2D: New environment point
        """
        return cls(point.x, point.y, properties)
    
    def to_base(self) -> BasePoint2D:
        """
        Convert to a base Point2D.
        
        Returns:
            BasePoint2D: Base point
        """
        return BasePoint2D(self.x, self.y)


class EnvironmentPoint3D(BasePoint3D):
    """
    Extended 3D point for environment-specific operations.
    
    This class extends the base Point3D with additional functionality
    needed by the environment module.
    """
    
    def __init__(self, x: float, y: float, z: float, properties: dict = None):
        """
        Initialize an environment point.
        
        Args:
            x: x-coordinate
            y: y-coordinate
            z: z-coordinate
            properties: Optional properties dictionary
        """
        super().__init__(x, y, z)
        self.properties = properties or {}
    
    def with_property(self, key: str, value: Any) -> 'EnvironmentPoint3D':
        """
        Create a new point with an added property.
        
        Args:
            key: Property key
            value: Property value
            
        Returns:
            EnvironmentPoint3D: New point with property added
        """
        new_point = EnvironmentPoint3D(self.x, self.y, self.z, self.properties.copy())
        new_point.properties[key] = value
        return new_point
    
    def with_properties(self, properties: dict) -> 'EnvironmentPoint3D':
        """
        Create a new point with added properties.
        
        Args:
            properties: Properties to add
            
        Returns:
            EnvironmentPoint3D: New point with properties added
        """
        new_properties = self.properties.copy()
        new_properties.update(properties)
        return EnvironmentPoint3D(self.x, self.y, self.z, new_properties)
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a property value.
        
        Args:
            key: Property key
            default: Default value if property doesn't exist
            
        Returns:
            Any: Property value or default
        """
        return self.properties.get(key, default)
    
    def has_property(self, key: str) -> bool:
        """
        Check if point has a property.
        
        Args:
            key: Property key
            
        Returns:
            bool: True if property exists
        """
        return key in self.properties
    
    @classmethod
    def from_base(cls, point: BasePoint3D, properties: dict = None) -> 'EnvironmentPoint3D':
        """
        Create from a base Point3D.
        
        Args:
            point: Base Point3D object
            properties: Optional properties dictionary
            
        Returns:
            EnvironmentPoint3D: New environment point
        """
        return cls(point.x, point.y, point.z, properties)
    
    def to_base(self) -> BasePoint3D:
        """
        Convert to a base Point3D.
        
        Returns:
            BasePoint3D: Base point
        """
        return BasePoint3D(self.x, self.y, self.z)
    
    def to_2d(self) -> EnvironmentPoint2D:
        """
        Convert to a 2D environment point.
        
        Returns:
            EnvironmentPoint2D: 2D point with same x and y
        """
        return EnvironmentPoint2D(self.x, self.y, self.properties.copy())