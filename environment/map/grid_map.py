"""
Grid map implementation for the mining dispatch system.

Provides a grid-based map representation, where the environment is divided
into cells of uniform size.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
import uuid
from scipy import spatial

from utils.geo.coordinates import Point2D, BoundingBox
from utils.logger import get_logger
from utils.io.serialization import Serializable

from environment.map.interfaces import Map, MapNode, MapEdge, MapError


logger = get_logger("grid_map")


class GridCell:
    """
    Representation of a cell in a grid map.
    
    Cells are the fundamental units of the grid, containing information
    about terrain, obstacles, and other properties.
    """
    
    def __init__(self, cell_id: str = None, row: int = 0, col: int = 0,
                 passable: bool = True, properties: Dict[str, Any] = None):
        """
        Initialize a grid cell.
        
        Args:
            cell_id: Unique identifier for the cell
            row: Row index of the cell
            col: Column index of the cell
            passable: Whether the cell is passable
            properties: Additional properties of the cell
        """
        self.cell_id = cell_id or str(uuid.uuid4())
        self.row = row
        self.col = col
        self.passable = passable
        self.properties = properties or {}
        
    def __repr__(self) -> str:
        """String representation of the cell."""
        return f"GridCell(id={self.cell_id}, row={self.row}, col={self.col}, passable={self.passable})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cell to dictionary for serialization."""
        return {
            'cell_id': self.cell_id,
            'row': self.row,
            'col': self.col,
            'passable': self.passable,
            'properties': self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridCell':
        """Create cell from dictionary representation."""
        return cls(
            cell_id=data.get('cell_id'),
            row=data.get('row', 0),
            col=data.get('col', 0),
            passable=data.get('passable', True),
            properties=data.get('properties', {})
        )


class GridMap(Map):
    """
    Grid-based map implementation.
    
    Represents the environment as a grid of cells, each with properties
    describing the terrain, obstacles, and other features.
    """
    
    def __init__(self, map_id: str = None, name: str = "grid_map",
                 rows: int = 100, cols: int = 100, cell_size: float = 1.0,
                 origin: Point2D = None):
        """
        Initialize a grid map.
        
        Args:
            map_id: Unique identifier for the map
            name: Name of the map
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            cell_size: Size of each cell in meters
            origin: Origin point of the grid in world coordinates
        """
        super().__init__(map_id, name)
        
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.origin = origin or Point2D(0, 0)
        
        # Initialize grid cells
        self.grid = [[None for _ in range(cols)] for _ in range(rows)]
        self.cells = {}  # cell_id -> cell
        
        # Initialize nodes and edges
        self.nodes = {}  # node_id -> node
        self.edges = {}  # edge_id -> edge
        
        # Create grid cells
        self._initialize_grid()
        
        # Create connectivity graph
        self._initialize_connectivity()
        
        logger.info(f"Grid map '{name}' initialized with {rows}x{cols} cells")
    
    def _initialize_grid(self) -> None:
        """Initialize the grid with cells."""
        for row in range(self.rows):
            for col in range(self.cols):
                # Create a new cell
                cell = GridCell(row=row, col=col)
                
                # Store the cell
                self.grid[row][col] = cell
                self.cells[cell.cell_id] = cell
    
    def _initialize_connectivity(self) -> None:
        """Initialize the connectivity graph with nodes and edges."""
        # Create a node for each cell
        for cell_id, cell in self.cells.items():
            # Calculate world position of cell center
            x = self.origin.x + (cell.col + 0.5) * self.cell_size
            y = self.origin.y + (cell.row + 0.5) * self.cell_size
            
            # Create a node for the cell
            node = MapNode(
                node_id=cell_id,  # Use cell ID as node ID
                position=Point2D(x, y),
                node_type="cell",
                properties={
                    'row': cell.row,
                    'col': cell.col,
                    'passable': cell.passable
                }
            )
            
            # Store the node
            self.nodes[node.node_id] = node
        
        # Create edges between adjacent cells (8-connectivity)
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.grid[row][col]
                
                # Skip non-passable cells
                if not cell.passable:
                    continue
                
                # Check all 8 adjacent cells
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        # Skip self
                        if dr == 0 and dc == 0:
                            continue
                        
                        # Check if adjacent cell is valid
                        adj_row = row + dr
                        adj_col = col + dc
                        
                        if 0 <= adj_row < self.rows and 0 <= adj_col < self.cols:
                            adj_cell = self.grid[adj_row][adj_col]
                            
                            # Skip non-passable adjacent cells
                            if not adj_cell.passable:
                                continue
                            
                            # Create an edge
                            edge_id = f"{cell.cell_id}_{adj_cell.cell_id}"
                            
                            # Calculate weight (diagonal edges are longer)
                            weight = 1.0
                            if dr != 0 and dc != 0:
                                weight = 1.414  # sqrt(2)
                            
                            edge = MapEdge(
                                edge_id=edge_id,
                                start_node=cell.cell_id,
                                end_node=adj_cell.cell_id,
                                edge_type="grid_connection",
                                properties={
                                    'weight': weight
                                }
                            )
                            
                            # Store the edge
                            self.edges[edge.edge_id] = edge
                            
                            # Update node connectivity
                            self.nodes[cell.cell_id].edges.add(adj_cell.cell_id)
    
    def cell_to_world(self, row: int, col: int) -> Point2D:
        """
        Convert grid coordinates to world coordinates.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            Point2D: World coordinates of cell center
        """
        x = self.origin.x + (col + 0.5) * self.cell_size
        y = self.origin.y + (row + 0.5) * self.cell_size
        return Point2D(x, y)
    
    def world_to_cell(self, position: Point2D) -> Tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            position: World coordinates
            
        Returns:
            Tuple[int, int]: (row, col) indices
        """
        col = int((position.x - self.origin.x) / self.cell_size)
        row = int((position.y - self.origin.y) / self.cell_size)
        
        # Clamp to grid bounds
        row = max(0, min(row, self.rows - 1))
        col = max(0, min(col, self.cols - 1))
        
        return row, col
    
    def is_valid_cell(self, row: int, col: int) -> bool:
        """
        Check if cell indices are valid.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            bool: True if cell is valid
        """
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def get_cell(self, row: int, col: int) -> Optional[GridCell]:
        """
        Get a cell by its grid coordinates.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            GridCell or None: The cell, or None if invalid coordinates
        """
        if not self.is_valid_cell(row, col):
            return None
        
        return self.grid[row][col]
    
    def get_cell_by_position(self, position: Point2D) -> Optional[GridCell]:
        """
        Get a cell by world position.
        
        Args:
            position: World coordinates
            
        Returns:
            GridCell or None: The cell, or None if outside grid
        """
        row, col = self.world_to_cell(position)
        return self.get_cell(row, col)
    
    def get_cell_by_id(self, cell_id: str) -> Optional[GridCell]:
        """
        Get a cell by its ID.
        
        Args:
            cell_id: ID of the cell
            
        Returns:
            GridCell or None: The cell, or None if not found
        """
        return self.cells.get(cell_id)
    
    def set_cell_passable(self, row: int, col: int, passable: bool) -> bool:
        """
        Set whether a cell is passable.
        
        Args:
            row: Row index
            col: Column index
            passable: Whether the cell is passable
            
        Returns:
            bool: True if successful, False if invalid coordinates
        """
        cell = self.get_cell(row, col)
        if cell is None:
            return False
        
        # Update cell
        cell.passable = passable
        
        # Update corresponding node
        node = self.nodes.get(cell.cell_id)
        if node:
            node.properties['passable'] = passable
        
        # Update connectivity based on passability
        self._update_cell_connectivity(cell)
        
        return True
    
    def _update_cell_connectivity(self, cell: GridCell) -> None:
        """
        Update connectivity for a cell based on its passability.
        
        Args:
            cell: Cell to update
        """
        # Get corresponding node
        node = self.nodes.get(cell.cell_id)
        if not node:
            return
        
        if cell.passable:
            # Cell became passable, add connections to adjacent cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # Skip self
                    if dr == 0 and dc == 0:
                        continue
                    
                    # Check if adjacent cell is valid
                    adj_row = cell.row + dr
                    adj_col = cell.col + dc
                    
                    if self.is_valid_cell(adj_row, adj_col):
                        adj_cell = self.grid[adj_row][adj_col]
                        
                        # Skip non-passable adjacent cells
                        if not adj_cell.passable:
                            continue
                        
                        # Create edges in both directions
                        edge_id_out = f"{cell.cell_id}_{adj_cell.cell_id}"
                        edge_id_in = f"{adj_cell.cell_id}_{cell.cell_id}"
                        
                        # Calculate weight (diagonal edges are longer)
                        weight = 1.0
                        if dr != 0 and dc != 0:
                            weight = 1.414  # sqrt(2)
                        
                        # Create outgoing edge
                        if edge_id_out not in self.edges:
                            edge = MapEdge(
                                edge_id=edge_id_out,
                                start_node=cell.cell_id,
                                end_node=adj_cell.cell_id,
                                edge_type="grid_connection",
                                properties={
                                    'weight': weight
                                }
                            )
                            self.edges[edge.edge_id] = edge
                            node.edges.add(adj_cell.cell_id)
                        
                        # Update adjacent node's connections
                        adj_node = self.nodes.get(adj_cell.cell_id)
                        if adj_node:
                            # Create incoming edge
                            if edge_id_in not in self.edges:
                                edge = MapEdge(
                                    edge_id=edge_id_in,
                                    start_node=adj_cell.cell_id,
                                    end_node=cell.cell_id,
                                    edge_type="grid_connection",
                                    properties={
                                        'weight': weight
                                    }
                                )
                                self.edges[edge.edge_id] = edge
                                adj_node.edges.add(cell.cell_id)
        else:
            # Cell became impassable, remove all connections
            # Find all edges that involve this cell
            edges_to_remove = []
            
            for edge_id, edge in self.edges.items():
                if edge.start_node == cell.cell_id or edge.end_node == cell.cell_id:
                    edges_to_remove.append(edge_id)
            
            # Remove the edges
            for edge_id in edges_to_remove:
                edge = self.edges.pop(edge_id)
                
                # Update node connectivity
                if edge.start_node in self.nodes and edge.end_node in self.nodes.get(edge.start_node).edges:
                    self.nodes[edge.start_node].edges.remove(edge.end_node)
    
    def get_node(self, node_id: str) -> Optional[MapNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[MapEdge]:
        """Get an edge by its ID."""
        return self.edges.get(edge_id)
    
    def add_node(self, node: MapNode) -> str:
        """Add a node to the map."""
        self.nodes[node.node_id] = node
        return node.node_id
    
    def add_edge(self, edge: MapEdge) -> str:
        """Add an edge to the map."""
        # Ensure both nodes exist
        if edge.start_node not in self.nodes or edge.end_node not in self.nodes:
            raise MapError("Cannot add edge: one or both nodes do not exist")
        
        self.edges[edge.edge_id] = edge
        
        # Update node connectivity
        self.nodes[edge.start_node].edges.add(edge.end_node)
        
        return edge.edge_id
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the map."""
        if node_id not in self.nodes:
            return False
        
        # Remove all edges connected to the node
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.start_node == node_id or edge.end_node == node_id:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)
        
        # Remove the node
        self.nodes.pop(node_id)
        
        return True
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the map."""
        if edge_id not in self.edges:
            return False
        
        edge = self.edges.pop(edge_id)
        
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
        
        node = self.nodes[node_id]
        return [self.nodes[edge_node_id] for edge_node_id in node.edges if edge_node_id in self.nodes]
    
    def get_node_edges(self, node_id: str) -> List[MapEdge]:
        """Get all edges connected to a given node."""
        if node_id not in self.nodes:
            return []
        
        return [edge for edge in self.edges.values() 
                if edge.start_node == node_id or edge.end_node == node_id]
    
    def find_path(self, start_node_id: str, end_node_id: str, 
                  weight_property: str = "weight") -> List[str]:
        """Find a path between two nodes using A* algorithm."""
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            raise MapError("Start or end node does not exist")
        
        start_node = self.nodes[start_node_id]
        end_node = self.nodes[end_node_id]
        
        # Check if nodes are passable
        if start_node.properties.get('passable') is False or end_node.properties.get('passable') is False:
            raise MapError("Start or end node is not passable")
        
        # A* algorithm
        import heapq
        
        # Helper functions
        def heuristic(node1_id, node2_id):
            """Manhattan distance heuristic."""
            node1 = self.nodes[node1_id]
            node2 = self.nodes[node2_id]
            return abs(node1.position.x - node2.position.x) + abs(node1.position.y - node2.position.y)
        
        def get_edge_weight(from_id, to_id):
            """Get weight of edge between two nodes."""
            edge_id = f"{from_id}_{to_id}"
            if edge_id in self.edges:
                edge = self.edges[edge_id]
                return edge.properties.get(weight_property, 1.0)
            return 1.0
        
        # Initialize data structures
        open_set = []  # Priority queue of (f_score, node_id)
        closed_set = set()  # Set of visited nodes
        g_score = {start_node_id: 0}  # Cost from start to node
        f_score = {start_node_id: heuristic(start_node_id, end_node_id)}  # Estimated total cost
        came_from = {}  # Parent node in path
        
        # Add start node to open set
        heapq.heappush(open_set, (f_score[start_node_id], start_node_id))
        
        while open_set:
            # Get node with lowest f_score
            current_f, current_node_id = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if current_node_id == end_node_id:
                # Reconstruct and return the path
                path = [current_node_id]
                while current_node_id in came_from:
                    current_node_id = came_from[current_node_id]
                    path.append(current_node_id)
                path.reverse()
                return path
            
            # Skip if already processed
            if current_node_id in closed_set:
                continue
            
            # Mark as processed
            closed_set.add(current_node_id)
            
            # Process neighbors
            for neighbor_id in self.nodes[current_node_id].edges:
                # Skip if neighbor is not passable
                neighbor = self.nodes.get(neighbor_id)
                if not neighbor or neighbor.properties.get('passable') is False:
                    continue
                
                # Skip if neighbor already processed
                if neighbor_id in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current_node_id] + get_edge_weight(current_node_id, neighbor_id)
                
                # Check if this path is better than any previous one
                if neighbor_id not in g_score or tentative_g < g_score[neighbor_id]:
                    # Record this path
                    came_from[neighbor_id] = current_node_id
                    g_score[neighbor_id] = tentative_g
                    f_score[neighbor_id] = tentative_g + heuristic(neighbor_id, end_node_id)
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))
        
        # No path found
        raise MapError(f"No path found from {start_node_id} to {end_node_id}")
    
    def find_nearest_node(self, position: Point2D, 
                          filter_func: Optional[Callable[[MapNode], bool]] = None) -> Optional[str]:
        """Find the nearest node to a given position."""
        if not self.nodes:
            return None
        
        # Get all nodes
        nodes_list = list(self.nodes.values())
        
        # Apply filter if provided
        if filter_func:
            nodes_list = [node for node in nodes_list if filter_func(node)]
        
        if not nodes_list:
            return None
        
        # Find nearest node
        nearest_node = min(nodes_list, key=lambda node: node.position.distance_to(position))
        return nearest_node.node_id
    
    def get_bounds(self) -> BoundingBox:
        """Get the bounding box of the map."""
        min_x = self.origin.x
        min_y = self.origin.y
        max_x = min_x + self.cols * self.cell_size
        max_y = min_y + self.rows * self.cell_size
        
        return BoundingBox(Point2D(min_x, min_y), Point2D(max_x, max_y))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert map to dictionary for serialization."""
        return {
            'map_id': self.map_id,
            'name': self.name,
            'metadata': self.metadata,
            'type': 'grid_map',
            'rows': self.rows,
            'cols': self.cols,
            'cell_size': self.cell_size,
            'origin': {
                'x': self.origin.x,
                'y': self.origin.y
            },
            'cells': {
                cell_id: cell.to_dict() for cell_id, cell in self.cells.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridMap':
        """Create map from dictionary representation."""
        origin = Point2D(data['origin']['x'], data['origin']['y'])
        
        # Create the map
        grid_map = cls(
            map_id=data.get('map_id'),
            name=data.get('name', 'grid_map'),
            rows=data.get('rows', 100),
            cols=data.get('cols', 100),
            cell_size=data.get('cell_size', 1.0),
            origin=origin
        )
        
        # Set metadata
        grid_map.metadata = data.get('metadata', {})
        
        # Update cells from data
        if 'cells' in data:
            for cell_id, cell_data in data['cells'].items():
                cell = GridCell.from_dict(cell_data)
                
                # Update the grid and cells dictionary
                if grid_map.is_valid_cell(cell.row, cell.col):
                    grid_map.grid[cell.row][cell.col] = cell
                    grid_map.cells[cell.cell_id] = cell
        
        # Reinitialize connectivity after updating cells
        grid_map._initialize_connectivity()
        
        return grid_map