"""
Map interfaces for the mining dispatch system.

Defines the abstract interfaces that all map implementations must follow,
including grid maps and road networks.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import uuid

from utils.geo.coordinates import Point2D, BoundingBox
from utils.io.serialization import Serializable


class MapError(Exception):
    """Base exception for map-related errors."""
    pass


class MapNode:
    """
    Representation of a node in a map.
    
    Nodes are points of interest or connection points in the map structure.
    """
    
    def __init__(self, node_id: str = None, position: Point2D = None, 
                 node_type: str = "regular", properties: Dict[str, Any] = None):
        """
        Initialize a map node.
        
        Args:
            node_id: Unique identifier for the node
            position: Position of the node
            node_type: Type of the node (e.g., regular, key_point, junction)
            properties: Additional properties of the node
        """
        self.node_id = node_id or str(uuid.uuid4())
        self.position = position or Point2D(0, 0)
        self.node_type = node_type
        self.properties = properties or {}
        self.edges = set()  # Set of connected node IDs
        
    def __repr__(self) -> str:
        """String representation of the node."""
        return f"MapNode(id={self.node_id}, pos={self.position}, type={self.node_type})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'position': {'x': self.position.x, 'y': self.position.y},
            'node_type': self.node_type,
            'properties': self.properties,
            'edges': list(self.edges)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MapNode':
        """Create node from dictionary representation."""
        node = cls(
            node_id=data.get('node_id'),
            position=Point2D(data['position']['x'], data['position']['y']),
            node_type=data.get('node_type', 'regular'),
            properties=data.get('properties', {})
        )
        node.edges = set(data.get('edges', []))
        return node


class MapEdge:
    """
    Representation of an edge in a map.
    
    Edges connect nodes and represent pathways or relationships.
    """
    
    def __init__(self, edge_id: str = None, start_node: str = None, end_node: str = None,
                 edge_type: str = "regular", properties: Dict[str, Any] = None):
        """
        Initialize a map edge.
        
        Args:
            edge_id: Unique identifier for the edge
            start_node: ID of the starting node
            end_node: ID of the ending node
            edge_type: Type of the edge (e.g., regular, road, restricted)
            properties: Additional properties of the edge
        """
        self.edge_id = edge_id or str(uuid.uuid4())
        self.start_node = start_node
        self.end_node = end_node
        self.edge_type = edge_type
        self.properties = properties or {}
        
    def __repr__(self) -> str:
        """String representation of the edge."""
        return f"MapEdge(id={self.edge_id}, {self.start_node} -> {self.end_node}, type={self.edge_type})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            'edge_id': self.edge_id,
            'start_node': self.start_node,
            'end_node': self.end_node,
            'edge_type': self.edge_type,
            'properties': self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MapEdge':
        """Create edge from dictionary representation."""
        return cls(
            edge_id=data.get('edge_id'),
            start_node=data.get('start_node'),
            end_node=data.get('end_node'),
            edge_type=data.get('edge_type', 'regular'),
            properties=data.get('properties', {})
        )


class Map(ABC, Serializable):
    """
    Abstract base class for all map implementations.
    
    Defines the common interface that all map types must implement.
    """
    
    def __init__(self, map_id: str = None, name: str = "map"):
        """
        Initialize a map.
        
        Args:
            map_id: Unique identifier for the map
            name: Name of the map
        """
        self.map_id = map_id or str(uuid.uuid4())
        self.name = name
        self.metadata = {}
        
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[MapNode]:
        """
        Get a node by its ID.
        
        Args:
            node_id: ID of the node
            
        Returns:
            MapNode or None: The node, or None if not found
        """
        pass
    
    @abstractmethod
    def get_edge(self, edge_id: str) -> Optional[MapEdge]:
        """
        Get an edge by its ID.
        
        Args:
            edge_id: ID of the edge
            
        Returns:
            MapEdge or None: The edge, or None if not found
        """
        pass
    
    @abstractmethod
    def add_node(self, node: MapNode) -> str:
        """
        Add a node to the map.
        
        Args:
            node: Node to add
            
        Returns:
            str: ID of the added node
        """
        pass
    
    @abstractmethod
    def add_edge(self, edge: MapEdge) -> str:
        """
        Add an edge to the map.
        
        Args:
            edge: Edge to add
            
        Returns:
            str: ID of the added edge
        """
        pass
    
    @abstractmethod
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the map.
        
        Args:
            node_id: ID of the node to remove
            
        Returns:
            bool: True if node was removed, False if not found
        """
        pass
    
    @abstractmethod
    def remove_edge(self, edge_id: str) -> bool:
        """
        Remove an edge from the map.
        
        Args:
            edge_id: ID of the edge to remove
            
        Returns:
            bool: True if edge was removed, False if not found
        """
        pass
    
    @abstractmethod
    def get_nodes(self) -> List[MapNode]:
        """
        Get all nodes in the map.
        
        Returns:
            List[MapNode]: List of all nodes
        """
        pass
    
    @abstractmethod
    def get_edges(self) -> List[MapEdge]:
        """
        Get all edges in the map.
        
        Returns:
            List[MapEdge]: List of all edges
        """
        pass
    
    @abstractmethod
    def get_connected_nodes(self, node_id: str) -> List[MapNode]:
        """
        Get all nodes connected to a given node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            List[MapNode]: List of connected nodes
        """
        pass
    
    @abstractmethod
    def get_node_edges(self, node_id: str) -> List[MapEdge]:
        """
        Get all edges connected to a given node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            List[MapEdge]: List of connected edges
        """
        pass
    
    @abstractmethod
    def find_path(self, start_node_id: str, end_node_id: str, 
                  weight_property: str = "weight") -> List[str]:
        """
        Find a path between two nodes.
        
        Args:
            start_node_id: ID of the starting node
            end_node_id: ID of the ending node
            weight_property: Edge property to use as weight
            
        Returns:
            List[str]: List of node IDs forming the path
            
        Raises:
            MapError: If no path exists or other error occurs
        """
        pass
    
    @abstractmethod
    def find_nearest_node(self, position: Point2D, 
                          filter_func: Optional[callable] = None) -> Optional[str]:
        """
        Find the nearest node to a given position.
        
        Args:
            position: Position to find nearest node to
            filter_func: Optional function to filter nodes
            
        Returns:
            str or None: ID of the nearest node, or None if no nodes
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> BoundingBox:
        """
        Get the bounding box of the map.
        
        Returns:
            BoundingBox: Bounding box of the map
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert map to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Map':
        """
        Create map from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Map: New map instance
        """
        pass