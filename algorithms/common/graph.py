"""
Graph data structures for search algorithms in the mining dispatch system.

Provides efficient graph representations and operations used in path finding,
scheduling, and other graph-based algorithms.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from enum import Enum
from typing import Dict, List, Set, Tuple, Any, Optional, TypeVar, Generic, Callable, Iterable
from dataclasses import dataclass, field
import heapq
import math

from utils.logger import get_logger

# Get logger
logger = get_logger("algorithms.graph")

# Type variables for generic functions
T = TypeVar('T')  # Node type
K = TypeVar('K')  # Key type


class EdgeType(Enum):
    """Types of edges in a graph."""
    DIRECTED = 1    # Directed edge (one-way)
    UNDIRECTED = 2  # Undirected edge (two-way)


@dataclass
class Edge(Generic[T]):
    """
    Edge in a graph connecting two nodes.
    
    An edge represents a connection between nodes with an associated cost.
    """
    source: T                     # Source node
    target: T                     # Target node
    cost: float = 1.0             # Edge cost/weight
    edge_type: EdgeType = EdgeType.DIRECTED  # Edge type
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def __lt__(self, other: 'Edge[T]') -> bool:
        """Compare edges by cost for priority queue."""
        return self.cost < other.cost


class Graph(Generic[T]):
    """
    Generic graph data structure.
    
    Provides operations for constructing and manipulating graphs with
    arbitrary node types.
    """
    
    def __init__(self):
        """Initialize an empty graph."""
        self.nodes: Set[T] = set()  # Set of all nodes
        self.edges: Dict[T, List[Edge[T]]] = {}  # Adjacency list representation
        self.metadata: Dict[str, Any] = {}  # Graph metadata
    
    def add_node(self, node: T, **metadata) -> None:
        """
        Add a node to the graph.
        
        Args:
            node: Node to add
            **metadata: Optional node metadata
        """
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = []
    
    def add_edge(self, source: T, target: T, cost: float = 1.0, 
                edge_type: EdgeType = EdgeType.DIRECTED, **metadata) -> None:
        """
        Add an edge to the graph.
        
        Args:
            source: Source node
            target: Target node
            cost: Edge cost/weight
            edge_type: Edge type
            **metadata: Optional edge metadata
        """
        # Ensure nodes exist
        self.add_node(source)
        self.add_node(target)
        
        # Create edge
        edge_metadata = metadata if metadata else {}
        edge = Edge(source, target, cost, edge_type, edge_metadata)
        
        # Add to adjacency list
        self.edges[source].append(edge)
        
        # If undirected, add reverse edge
        if edge_type == EdgeType.UNDIRECTED:
            reverse_edge = Edge(target, source, cost, edge_type, edge_metadata.copy())
            self.edges[target].append(reverse_edge)
    
    def remove_node(self, node: T) -> None:
        """
        Remove a node and all its edges from the graph.
        
        Args:
            node: Node to remove
        """
        if node not in self.nodes:
            return
        
        # Remove from nodes set
        self.nodes.remove(node)
        
        # Remove outgoing edges
        if node in self.edges:
            del self.edges[node]
        
        # Remove incoming edges
        for source in self.edges:
            self.edges[source] = [edge for edge in self.edges[source] if edge.target != node]
    
    def remove_edge(self, source: T, target: T) -> None:
        """
        Remove an edge from the graph.
        
        Args:
            source: Source node
            target: Target node
        """
        if source not in self.edges:
            return
        
        # Find edge
        edge_to_remove = None
        for edge in self.edges[source]:
            if edge.target == target:
                edge_to_remove = edge
                break
        
        # Remove edge
        if edge_to_remove:
            self.edges[source].remove(edge_to_remove)
            
            # If undirected, remove reverse edge
            if edge_to_remove.edge_type == EdgeType.UNDIRECTED:
                for edge in self.edges[target]:
                    if edge.target == source:
                        self.edges[target].remove(edge)
                        break
    
    def get_neighbors(self, node: T) -> List[T]:
        """
        Get all neighbors of a node.
        
        Args:
            node: Node to get neighbors for
            
        Returns:
            List[T]: List of neighboring nodes
        """
        if node not in self.edges:
            return []
        
        return [edge.target for edge in self.edges[node]]
    
    def get_edges(self, node: T) -> List[Edge[T]]:
        """
        Get all edges from a node.
        
        Args:
            node: Node to get edges for
            
        Returns:
            List[Edge[T]]: List of edges
        """
        if node not in self.edges:
            return []
        
        return self.edges[node]
    
    def get_edge(self, source: T, target: T) -> Optional[Edge[T]]:
        """
        Get the edge between source and target.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            Optional[Edge[T]]: Edge if exists, None otherwise
        """
        if source not in self.edges:
            return None
        
        for edge in self.edges[source]:
            if edge.target == target:
                return edge
        
        return None
    
    def get_edge_cost(self, source: T, target: T) -> float:
        """
        Get the cost of the edge between source and target.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            float: Edge cost if exists, infinity otherwise
        """
        edge = self.get_edge(source, target)
        return edge.cost if edge else float('inf')
    
    def has_node(self, node: T) -> bool:
        """
        Check if a node exists in the graph.
        
        Args:
            node: Node to check
            
        Returns:
            bool: True if node exists
        """
        return node in self.nodes
    
    def has_edge(self, source: T, target: T) -> bool:
        """
        Check if an edge exists in the graph.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            bool: True if edge exists
        """
        edge = self.get_edge(source, target)
        return edge is not None
    
    def node_count(self) -> int:
        """
        Get the number of nodes in the graph.
        
        Returns:
            int: Number of nodes
        """
        return len(self.nodes)
    
    def edge_count(self) -> int:
        """
        Get the number of edges in the graph.
        
        Returns:
            int: Number of edges
        """
        return sum(len(edges) for edges in self.edges.values())
    
    def get_all_edges(self) -> List[Edge[T]]:
        """
        Get all edges in the graph.
        
        Returns:
            List[Edge[T]]: List of all edges
        """
        all_edges = []
        for node in self.edges:
            all_edges.extend(self.edges[node])
        return all_edges
    
    def is_directed(self) -> bool:
        """
        Check if the graph is directed.
        
        Returns:
            bool: True if all edges are directed
        """
        for node in self.edges:
            for edge in self.edges[node]:
                if edge.edge_type == EdgeType.UNDIRECTED:
                    return False
        return True
    
    def get_connected_components(self) -> List[Set[T]]:
        """
        Get the connected components of the graph.
        
        Returns:
            List[Set[T]]: List of connected components
        """
        components = []
        visited = set()
        
        for node in self.nodes:
            if node in visited:
                continue
            
            # Start a new component
            component = set()
            queue = [node]
            
            while queue:
                current = queue.pop(0)
                if current in component:
                    continue
                
                component.add(current)
                visited.add(current)
                
                # Add neighbors
                for neighbor in self.get_neighbors(current):
                    if neighbor not in component:
                        queue.append(neighbor)
            
            components.append(component)
        
        return components
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the graph to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        # Note: This assumes nodes can be stringified for dict keys
        node_dict = {}
        
        for node in self.nodes:
            node_str = str(node)
            node_dict[node_str] = {
                "neighbors": [str(edge.target) for edge in self.edges.get(node, [])],
                "edges": [
                    {
                        "target": str(edge.target),
                        "cost": edge.cost,
                        "type": edge.edge_type.name,
                        "metadata": edge.metadata
                    }
                    for edge in self.edges.get(node, [])
                ]
            }
        
        return {
            "nodes": node_dict,
            "metadata": self.metadata
        }


class GridGraph(Graph[Tuple[int, int]]):
    """
    Grid-based graph representation.
    
    Specialization of Graph for 2D grid environments common in path planning.
    """
    
    def __init__(self, width: int, height: int, diagonal: bool = True):
        """
        Initialize a grid graph.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            diagonal: Whether to include diagonal connections
        """
        super().__init__()
        self.width = width
        self.height = height
        self.diagonal = diagonal
        self.obstacles: Set[Tuple[int, int]] = set()
        
        # Generate grid
        self._generate_grid()
    
    def _generate_grid(self) -> None:
        """Generate the grid graph structure."""
        # Create nodes
        for x in range(self.width):
            for y in range(self.height):
                self.add_node((x, y))
        
        # Create edges
        for x in range(self.width):
            for y in range(self.height):
                # Orthogonal neighbors
                neighbors = [
                    (x + 1, y),      # Right
                    (x - 1, y),      # Left
                    (x, y + 1),      # Up
                    (x, y - 1)       # Down
                ]
                
                # Diagonal neighbors
                if self.diagonal:
                    neighbors.extend([
                        (x + 1, y + 1),  # Up-Right
                        (x - 1, y + 1),  # Up-Left
                        (x + 1, y - 1),  # Down-Right
                        (x - 1, y - 1)   # Down-Left
                    ])
                
                # Add edges
                for nx, ny in neighbors:
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        # Calculate cost (sqrt(2) for diagonal, 1 for orthogonal)
                        cost = 1.0
                        if self.diagonal and (nx != x and ny != y):
                            cost = math.sqrt(2.0)
                        
                        self.add_edge((x, y), (nx, ny), cost, EdgeType.UNDIRECTED)
    
    def set_obstacle(self, x: int, y: int, is_obstacle: bool = True) -> None:
        """
        Set a cell as an obstacle.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            is_obstacle: Whether the cell is an obstacle
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        
        if is_obstacle:
            self.obstacles.add((x, y))
            
            # Remove all edges to/from this node
            for neighbor in self.get_neighbors((x, y)):
                self.remove_edge((x, y), neighbor)
                self.remove_edge(neighbor, (x, y))
        else:
            if (x, y) in self.obstacles:
                self.obstacles.remove((x, y))
                
                # Reconnect edges
                neighbors = []
                
                # Orthogonal neighbors
                potential_neighbors = [
                    (x + 1, y),      # Right
                    (x - 1, y),      # Left
                    (x, y + 1),      # Up
                    (x, y - 1)       # Down
                ]
                
                # Diagonal neighbors
                if self.diagonal:
                    potential_neighbors.extend([
                        (x + 1, y + 1),  # Up-Right
                        (x - 1, y + 1),  # Up-Left
                        (x + 1, y - 1),  # Down-Right
                        (x - 1, y - 1)   # Down-Left
                    ])
                
                # Filter valid neighbors
                for nx, ny in potential_neighbors:
                    if (0 <= nx < self.width and 0 <= ny < self.height and 
                        (nx, ny) not in self.obstacles):
                        neighbors.append((nx, ny))
                
                # Add edges
                for neighbor in neighbors:
                    # Calculate cost (sqrt(2) for diagonal, 1 for orthogonal)
                    cost = 1.0
                    if self.diagonal and (neighbor[0] != x and neighbor[1] != y):
                        cost = math.sqrt(2.0)
                    
                    self.add_edge((x, y), neighbor, cost, EdgeType.UNDIRECTED)
    
    def is_obstacle(self, x: int, y: int) -> bool:
        """
        Check if a cell is an obstacle.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            bool: True if the cell is an obstacle
        """
        return (x, y) in self.obstacles
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """
        Check if a position is valid (in bounds and not an obstacle).
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            bool: True if the position is valid
        """
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                (x, y) not in self.obstacles)
    
    def set_cost_modifier(self, x: int, y: int, modifier: float) -> None:
        """
        Set a cost modifier for a cell.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            modifier: Cost multiplier
        """
        if not self.is_valid_position(x, y):
            return
        
        # Adjust costs for all edges to/from this cell
        cell = (x, y)
        
        # Update outgoing edges
        for edge in self.edges.get(cell, []):
            base_cost = 1.0
            if self.diagonal and (edge.target[0] != x and edge.target[1] != y):
                base_cost = math.sqrt(2.0)
            
            edge.cost = base_cost * modifier
        
        # Update incoming edges
        for neighbor in self.get_neighbors(cell):
            edge = self.get_edge(neighbor, cell)
            if edge:
                base_cost = 1.0
                if self.diagonal and (neighbor[0] != x and neighbor[1] != y):
                    base_cost = math.sqrt(2.0)
                
                edge.cost = base_cost * modifier
    
    def get_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """
        Calculate the total cost of a path.
        
        Args:
            path: List of grid cells forming the path
            
        Returns:
            float: Total path cost
        """
        if not path or len(path) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            edge = self.get_edge(path[i], path[i + 1])
            if edge:
                total_cost += edge.cost
            else:
                # If no direct edge, assume manhattan distance
                dx = abs(path[i + 1][0] - path[i][0])
                dy = abs(path[i + 1][1] - path[i][1])
                total_cost += dx + dy
        
        return total_cost


class WeightedGraph(Graph[T]):
    """
    Weighted graph with specialized operations.
    
    Extends the base Graph with operations optimized for weighted graphs.
    """
    
    def get_minimum_spanning_tree(self) -> 'WeightedGraph[T]':
        """
        Get the minimum spanning tree using Kruskal's algorithm.
        
        Returns:
            WeightedGraph[T]: Minimum spanning tree
        """
        # Create a new graph for the MST
        mst = WeightedGraph()
        
        # Add all nodes
        for node in self.nodes:
            mst.add_node(node)
        
        # Sort all edges by cost
        edges = self.get_all_edges()
        edges.sort(key=lambda e: e.cost)
        
        # Union-find data structure for cycle detection
        parent = {node: node for node in self.nodes}
        rank = {node: 0 for node in self.nodes}
        
        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]
        
        def union(node1, node2):
            root1 = find(node1)
            root2 = find(node2)
            
            if root1 != root2:
                if rank[root1] < rank[root2]:
                    parent[root1] = root2
                else:
                    parent[root2] = root1
                    if rank[root1] == rank[root2]:
                        rank[root1] += 1
        
        # Kruskal's algorithm
        for edge in edges:
            if find(edge.source) != find(edge.target):
                mst.add_edge(edge.source, edge.target, edge.cost, edge.edge_type, **edge.metadata)
                union(edge.source, edge.target)
        
        return mst
    
    def get_shortest_path_tree(self, source: T) -> 'WeightedGraph[T]':
        """
        Get the shortest path tree using Dijkstra's algorithm.
        
        Args:
            source: Source node
            
        Returns:
            WeightedGraph[T]: Shortest path tree
        """
        # Dijkstra's algorithm
        dist = {node: float('inf') for node in self.nodes}
        prev = {node: None for node in self.nodes}
        dist[source] = 0
        
        # Priority queue
        pq = [(0, source)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            # Skip if we've already found a shorter path
            if current_dist > dist[current]:
                continue
            
            # Update distances to neighbors
            for edge in self.edges.get(current, []):
                neighbor = edge.target
                weight = edge.cost
                distance = dist[current] + weight
                
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    prev[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        # Create the shortest path tree
        spt = WeightedGraph()
        
        # Add all nodes
        for node in self.nodes:
            spt.add_node(node)
        
        # Add edges from prev pointers
        for node in self.nodes:
            if prev[node] is not None:
                # Get original edge
                edge = self.get_edge(prev[node], node)
                if edge:
                    spt.add_edge(prev[node], node, edge.cost, edge.edge_type, **edge.metadata)
        
        return spt


# Helper functions

def grid_to_graph(grid: List[List[bool]], diagonal: bool = True) -> GridGraph:
    """
    Convert a 2D grid to a graph.
    
    Args:
        grid: 2D grid where True represents an obstacle
        diagonal: Whether to include diagonal connections
        
    Returns:
        GridGraph: Graph representation of the grid
    """
    height = len(grid)
    if height == 0:
        return GridGraph(0, 0, diagonal)
    
    width = len(grid[0])
    graph = GridGraph(width, height, diagonal)
    
    # Set obstacles
    for y in range(height):
        for x in range(width):
            if grid[y][x]:
                graph.set_obstacle(x, y)
    
    return graph


def build_visibility_graph(obstacles: List[List[Tuple[float, float]]], add_points: List[Tuple[float, float]] = None) -> Graph[Tuple[float, float]]:
    """
    Build a visibility graph for polygonal obstacles.
    
    Args:
        obstacles: List of obstacles, each a list of (x, y) points
        add_points: Additional points to include in the graph
        
    Returns:
        Graph[Tuple[float, float]]: Visibility graph
    """
    # Create graph
    graph = Graph()
    
    # Collect all points
    points = []
    for obstacle in obstacles:
        points.extend(obstacle)
    
    # Add additional points
    if add_points:
        points.extend(add_points)
    
    # Add all nodes
    for point in points:
        graph.add_node(point)
    
    # TODO: Implement visibility checking between points
    # This is a complex algorithm and beyond the scope of this implementation
    
    return graph