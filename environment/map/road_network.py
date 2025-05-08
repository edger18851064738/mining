"""
Road network map implementation for the mining dispatch system.

Provides a road network representation of the environment, with nodes
connected by roads of various types and properties.
"""

import math
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
import uuid
from scipy import spatial

from utils.geo.coordinates import Point2D, BoundingBox
from utils.logger import get_logger
from utils.math.vectors import Vector2D
from utils.io.serialization import Serializable

from environment.map.interfaces import Map, MapNode, MapEdge, MapError


logger = get_logger("road_network")


class RoadType:
    """Constants for different road types."""
    MAIN = "main"
    SECONDARY = "secondary"
    ACCESS = "access"
    TRANSIT = "transit"
    RESTRICTED = "restricted"


class RoadNetwork(Map):
    """
    Road network map implementation.
    
    Represents the environment as a network of interconnected roads,
    with junctions, intersections, and various road types.
    """
    
    def __init__(self, map_id: str = None, name: str = "road_network"):
        """
        Initialize a road network map.
        
        Args:
            map_id: Unique identifier for the map
            name: Name of the map
        """
        super().__init__(map_id, name)
        
        # Initialize nodes and edges
        self.nodes = {}  # node_id -> node
        self.edges = {}  # edge_id -> edge
        
        # Initialize networkx graph for algorithms
        self.graph = nx.DiGraph()
        
        # Initialize spatial index for efficient nearest node queries
        self.spatial_index = None
        self._spatial_index_dirty = True
        
        logger.info(f"Road network map '{name}' initialized")
    
    def _update_spatial_index(self) -> None:
        """Update the spatial index for efficient nearest node queries."""
        if not self.nodes:
            self.spatial_index = None
            self._spatial_index_dirty = False
            return
        
        # Create lists of node IDs and positions
        node_ids = list(self.nodes.keys())
        positions = [(node.position.x, node.position.y) for node in self.nodes.values()]
        
        # Create KD-tree
        self.spatial_index = (spatial.KDTree(positions), node_ids)
        self._spatial_index_dirty = False
    
    def get_node(self, node_id: str) -> Optional[MapNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[MapEdge]:
        """Get an edge by its ID."""
        return self.edges.get(edge_id)
    
    def add_node(self, node: MapNode) -> str:
        """Add a node to the map."""
        # Add to nodes dictionary
        self.nodes[node.node_id] = node
        
        # Add to networkx graph
        self.graph.add_node(
            node.node_id,
            position=(node.position.x, node.position.y),
            node_type=node.node_type,
            properties=node.properties
        )
        
        # Mark spatial index as dirty
        self._spatial_index_dirty = True
        
        return node.node_id
    
    def add_edge(self, edge: MapEdge) -> str:
        """Add an edge to the map."""
        # Ensure both nodes exist
        if edge.start_node not in self.nodes or edge.end_node not in self.nodes:
            raise MapError("Cannot add edge: one or both nodes do not exist")
        
        # Add to edges dictionary
        self.edges[edge.edge_id] = edge
        
        # Add to networkx graph
        self.graph.add_edge(
            edge.start_node,
            edge.end_node,
            edge_id=edge.edge_id,
            edge_type=edge.edge_type,
            properties=edge.properties,
            weight=edge.properties.get('weight', 1.0)
        )
        
        # Update node connectivity
        self.nodes[edge.start_node].edges.add(edge.end_node)
        
        return edge.edge_id
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the map."""
        if node_id not in self.nodes:
            return False
        
        # Remove from nodes dictionary
        self.nodes.pop(node_id)
        
        # Remove from networkx graph (this also removes connected edges)
        self.graph.remove_node(node_id)
        
        # Remove related edges from edges dictionary
        edges_to_remove = [edge_id for edge_id, edge in self.edges.items()
                          if edge.start_node == node_id or edge.end_node == node_id]
        
        for edge_id in edges_to_remove:
            self.edges.pop(edge_id)
        
        # Mark spatial index as dirty
        self._spatial_index_dirty = True
        
        return True
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the map."""
        if edge_id not in self.edges:
            return False
        
        edge = self.edges.pop(edge_id)
        
        # Remove from networkx graph
        if self.graph.has_edge(edge.start_node, edge.end_node):
            self.graph.remove_edge(edge.start_node, edge.end_node)
        
        # Update node connectivity
        if edge.start_node in self.nodes and edge.end_node in self.nodes[edge.start_node].edges:
            self.nodes[edge.start_node].edges.remove(edge.end_node)
        
        return True
    
    def get_nodes(self) -> List[MapNode]:
        """Get all nodes in the map."""
        return list(self.nodes.values())
    
    def get_edges(self) -> List[MapEdge]:
        """Get all edges in the map."""
        return list(self.edges.values())
    
    def get_connected_nodes(self, node_id: str) -> List[MapNode]:
        """Get all nodes connected to a given node."""
        if node_id not in self.nodes:
            return []
        
        # Get successor nodes from graph
        connected_ids = list(self.graph.successors(node_id))
        
        return [self.nodes[nid] for nid in connected_ids if nid in self.nodes]
    
    def get_node_edges(self, node_id: str) -> List[MapEdge]:
        """Get all edges connected to a given node."""
        if node_id not in self.nodes:
            return []
        
        # Get outgoing and incoming edges
        outgoing_edges = [
            self.edges[data['edge_id']]
            for _, _, data in self.graph.out_edges(node_id, data=True)
            if 'edge_id' in data and data['edge_id'] in self.edges
        ]
        
        incoming_edges = [
            self.edges[data['edge_id']]
            for _, _, data in self.graph.in_edges(node_id, data=True)
            if 'edge_id' in data and data['edge_id'] in self.edges
        ]
        
        return outgoing_edges + incoming_edges
    
    def find_path(self, start_node_id: str, end_node_id: str, 
                  weight_property: str = "weight") -> List[str]:
        """Find a path between two nodes."""
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            raise MapError("Start or end node does not exist")
        
        try:
            # Set edge weights based on property
            for u, v, data in self.graph.edges(data=True):
                if 'properties' in data and weight_property in data['properties']:
                    data['weight'] = data['properties'][weight_property]
                else:
                    data['weight'] = 1.0
            
            # Use networkx to find shortest path
            path = nx.shortest_path(self.graph, start_node_id, end_node_id, weight='weight')
            return path
        except nx.NetworkXNoPath:
            raise MapError(f"No path found from {start_node_id} to {end_node_id}")
        except Exception as e:
            raise MapError(f"Error finding path: {str(e)}")
    
    def find_nearest_node(self, position: Point2D, 
                          filter_func: Optional[Callable[[MapNode], bool]] = None) -> Optional[str]:
        """Find the nearest node to a given position."""
        if not self.nodes:
            return None
        
        # Update spatial index if needed
        if self._spatial_index_dirty or self.spatial_index is None:
            self._update_spatial_index()
        
        if self.spatial_index is None:
            return None
        
        # Query KD-tree
        kdtree, node_ids = self.spatial_index
        _, index = kdtree.query((position.x, position.y))
        
        # Get nearest node
        nearest_id = node_ids[index]
        nearest_node = self.nodes[nearest_id]
        
        # Apply filter if provided
        if filter_func and not filter_func(nearest_node):
            # If nearest doesn't match filter, check all nodes
            filtered_nodes = [node for node in self.nodes.values() if filter_func(node)]
            if not filtered_nodes:
                return None
            nearest_node = min(filtered_nodes, key=lambda node: node.position.distance_to(position))
            return nearest_node.node_id
        
        return nearest_id
    
    def get_bounds(self) -> BoundingBox:
        """Get the bounding box of the map."""
        if not self.nodes:
            # Default empty bounds
            return BoundingBox(Point2D(0, 0), Point2D(0, 0))
        
        # Find min and max coordinates
        min_x = min(node.position.x for node in self.nodes.values())
        min_y = min(node.position.y for node in self.nodes.values())
        max_x = max(node.position.x for node in self.nodes.values())
        max_y = max(node.position.y for node in self.nodes.values())
        
        return BoundingBox(Point2D(min_x, min_y), Point2D(max_x, max_y))
    
    def add_junction(self, position: Point2D, node_type: str = "junction", 
                    properties: Dict[str, Any] = None) -> str:
        """
        Add a junction node at the given position.
        
        Args:
            position: Position of the junction
            node_type: Type of the junction
            properties: Additional properties
            
        Returns:
            str: ID of the new junction node
        """
        node = MapNode(
            position=position,
            node_type=node_type,
            properties=properties or {}
        )
        
        return self.add_node(node)
    
    def add_road(self, start_node_id: str, end_node_id: str, road_type: str = RoadType.MAIN,
                bidirectional: bool = True, properties: Dict[str, Any] = None) -> List[str]:
        """
        Add a road between two nodes.
        
        Args:
            start_node_id: ID of the starting node
            end_node_id: ID of the ending node
            road_type: Type of the road
            bidirectional: Whether the road is bidirectional
            properties: Additional properties
            
        Returns:
            List[str]: IDs of the created edges
        """
        # Ensure both nodes exist
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            raise MapError("Cannot add road: one or both nodes do not exist")
        
        # Create properties if not provided
        if properties is None:
            properties = {}
        
        # Calculate road length and set as weight if not specified
        if 'weight' not in properties:
            start_pos = self.nodes[start_node_id].position
            end_pos = self.nodes[end_node_id].position
            road_length = start_pos.distance_to(end_pos)
            properties['weight'] = road_length
            properties['length'] = road_length
        
        # Set road type
        properties['road_type'] = road_type
        
        # Create forward edge
        forward_edge = MapEdge(
            start_node=start_node_id,
            end_node=end_node_id,
            edge_type="road",
            properties=properties.copy()
        )
        
        forward_id = self.add_edge(forward_edge)
        result = [forward_id]
        
        # Create backward edge if bidirectional
        if bidirectional:
            backward_edge = MapEdge(
                start_node=end_node_id,
                end_node=start_node_id,
                edge_type="road",
                properties=properties.copy()
            )
            
            backward_id = self.add_edge(backward_edge)
            result.append(backward_id)
        
        return result
    
    def generate_grid_network(self, rows: int, cols: int, cell_size: float, 
                             origin: Point2D = None) -> None:
        """
        Generate a grid road network.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            cell_size: Size of each cell in meters
            origin: Origin point of the grid
        """
        if origin is None:
            origin = Point2D(0, 0)
        
        # Clear existing network
        self.nodes.clear()
        self.edges.clear()
        self.graph.clear()
        self._spatial_index_dirty = True
        
        # Create junction nodes at grid intersections
        grid_nodes = {}
        
        for row in range(rows + 1):
            for col in range(cols + 1):
                # Calculate position
                x = origin.x + col * cell_size
                y = origin.y + row * cell_size
                
                # Create node
                node = MapNode(
                    position=Point2D(x, y),
                    node_type="junction",
                    properties={
                        'grid_row': row,
                        'grid_col': col
                    }
                )
                
                # Add node
                node_id = self.add_node(node)
                
                # Store in grid
                grid_nodes[(row, col)] = node_id
        
        # Connect nodes with roads
        for row in range(rows + 1):
            for col in range(cols + 1):
                current_id = grid_nodes[(row, col)]
                
                # Connect to right neighbor
                if col < cols:
                    right_id = grid_nodes[(row, col + 1)]
                    self.add_road(current_id, right_id, RoadType.MAIN)
                
                # Connect to bottom neighbor
                if row < rows:
                    bottom_id = grid_nodes[(row + 1, col)]
                    self.add_road(current_id, bottom_id, RoadType.MAIN)
        
        logger.info(f"Generated grid road network with {rows}x{cols} cells")
    
    def generate_radial_network(self, center: Point2D, radius: float, 
                               num_radials: int, num_rings: int) -> None:
        """
        Generate a radial road network.
        
        Args:
            center: Center point of the network
            radius: Outer radius of the network
            num_radials: Number of radial roads
            num_rings: Number of concentric ring roads
        """
        # Clear existing network
        self.nodes.clear()
        self.edges.clear()
        self.graph.clear()
        self._spatial_index_dirty = True
        
        # Create center node
        center_node = MapNode(
            position=center,
            node_type="junction",
            properties={
                'is_center': True
            }
        )
        center_id = self.add_node(center_node)
        
        # Create nodes for radials and rings
        radial_nodes = {}  # (ring_idx, radial_idx) -> node_id
        
        # Add rings
        for ring_idx in range(1, num_rings + 1):
            ring_radius = (ring_idx / num_rings) * radius
            
            for radial_idx in range(num_radials):
                angle = (radial_idx / num_radials) * 2 * math.pi
                
                x = center.x + ring_radius * math.cos(angle)
                y = center.y + ring_radius * math.sin(angle)
                
                node = MapNode(
                    position=Point2D(x, y),
                    node_type="junction",
                    properties={
                        'ring_idx': ring_idx,
                        'radial_idx': radial_idx
                    }
                )
                
                node_id = self.add_node(node)
                radial_nodes[(ring_idx, radial_idx)] = node_id
        
        # Connect radials to center
        for radial_idx in range(num_radials):
            inner_node_id = radial_nodes[(1, radial_idx)]
            self.add_road(center_id, inner_node_id, RoadType.MAIN)
        
        # Connect nodes along radials
        for radial_idx in range(num_radials):
            for ring_idx in range(1, num_rings):
                inner_node_id = radial_nodes[(ring_idx, radial_idx)]
                outer_node_id = radial_nodes[(ring_idx + 1, radial_idx)]
                self.add_road(inner_node_id, outer_node_id, RoadType.MAIN)
        
        # Connect nodes along rings
        for ring_idx in range(1, num_rings + 1):
            for radial_idx in range(num_radials):
                start_node_id = radial_nodes[(ring_idx, radial_idx)]
                end_node_id = radial_nodes[(ring_idx, (radial_idx + 1) % num_radials)]
                road_type = RoadType.MAIN if ring_idx == num_rings else RoadType.SECONDARY
                self.add_road(start_node_id, end_node_id, road_type)
        
        logger.info(f"Generated radial road network with {num_radials} radials and {num_rings} rings")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert map to dictionary for serialization."""
        return {
            'map_id': self.map_id,
            'name': self.name,
            'metadata': self.metadata,
            'type': 'road_network',
            'nodes': {
                node_id: node.to_dict() for node_id, node in self.nodes.items()
            },
            'edges': {
                edge_id: edge.to_dict() for edge_id, edge in self.edges.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoadNetwork':
        """Create map from dictionary representation."""
        # Create the map
        road_network = cls(
            map_id=data.get('map_id'),
            name=data.get('name', 'road_network')
        )
        
        # Set metadata
        road_network.metadata = data.get('metadata', {})
        
        # Add nodes
        if 'nodes' in data:
            for node_id, node_data in data['nodes'].items():
                node = MapNode.from_dict(node_data)
                road_network.add_node(node)
        
        # Add edges
        if 'edges' in data:
            for edge_id, edge_data in data['edges'].items():
                edge = MapEdge.from_dict(edge_data)
                try:
                    road_network.add_edge(edge)
                except MapError:
                    logger.warning(f"Could not add edge {edge_id}: nodes do not exist")
        
        return road_network