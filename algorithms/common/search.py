"""
Common search algorithms for the mining dispatch system.

Provides implementations of various search algorithms including A*, Dijkstra's,
breadth-first search, and depth-first search, which are used throughout the system.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import heapq
import math
from typing import Dict, List, Set, Tuple, Any, Optional, TypeVar, Generic, Callable, Iterable
from dataclasses import dataclass, field
import time

from utils.logger import get_logger, timed

from algorithms.common.graph import Graph, Edge, EdgeType

# Get logger
logger = get_logger("algorithms.search")

# Type variables for generic functions
T = TypeVar('T')  # Node type


@dataclass
class SearchResult(Generic[T]):
    """Result of a graph search operation."""
    success: bool = False             # Whether a path was found
    path: List[T] = field(default_factory=list)  # Path from start to goal
    cost: float = float('inf')        # Total path cost
    visited: Set[T] = field(default_factory=set)  # Set of visited nodes
    expanded: int = 0                 # Number of nodes expanded
    computation_time: float = 0.0     # Computation time in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class Search:
    """
    Static class providing various search algorithms.
    """
    
    @staticmethod
    @timed("breadth_first_search")
    def breadth_first_search(graph: Graph[T], start: T, goal: T) -> SearchResult[T]:
        """
        Perform breadth-first search on a graph.
        
        Args:
            graph: Graph to search
            start: Start node
            goal: Goal node
            
        Returns:
            SearchResult[T]: Search result
        """
        start_time = time.time()
        
        # Check if start or goal nodes don't exist
        if not graph.has_node(start) or not graph.has_node(goal):
            return SearchResult(
                success=False,
                computation_time=time.time() - start_time
            )
        
        # Initialize
        visited = set([start])
        queue = [(start, [start])]  # (node, path)
        expanded = 0
        
        # BFS loop
        while queue:
            current, path = queue.pop(0)
            expanded += 1
            
            # Check if goal reached
            if current == goal:
                return SearchResult(
                    success=True,
                    path=path,
                    cost=len(path) - 1,  # Cost is number of edges
                    visited=visited,
                    expanded=expanded,
                    computation_time=time.time() - start_time
                )
            
            # Expand neighbors
            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        # No path found
        return SearchResult(
            success=False,
            visited=visited,
            expanded=expanded,
            computation_time=time.time() - start_time
        )
    
    @staticmethod
    @timed("depth_first_search")
    def depth_first_search(graph: Graph[T], start: T, goal: T) -> SearchResult[T]:
        """
        Perform depth-first search on a graph.
        
        Args:
            graph: Graph to search
            start: Start node
            goal: Goal node
            
        Returns:
            SearchResult[T]: Search result
        """
        start_time = time.time()
        
        # Check if start or goal nodes don't exist
        if not graph.has_node(start) or not graph.has_node(goal):
            return SearchResult(
                success=False,
                computation_time=time.time() - start_time
            )
        
        # Initialize
        visited = set([start])
        stack = [(start, [start])]  # (node, path)
        expanded = 0
        
        # DFS loop
        while stack:
            current, path = stack.pop()
            expanded += 1
            
            # Check if goal reached
            if current == goal:
                return SearchResult(
                    success=True,
                    path=path,
                    cost=len(path) - 1,  # Cost is number of edges
                    visited=visited,
                    expanded=expanded,
                    computation_time=time.time() - start_time
                )
            
            # Expand neighbors in reverse order (to visit left-to-right in tree structures)
            neighbors = list(graph.get_neighbors(current))
            neighbors.reverse()
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))
        
        # No path found
        return SearchResult(
            success=False,
            visited=visited,
            expanded=expanded,
            computation_time=time.time() - start_time
        )
    
    @staticmethod
    @timed("dijkstra")
    def dijkstra(graph: Graph[T], start: T, goal: Optional[T] = None) -> SearchResult[T]:
        """
        Perform Dijkstra's algorithm on a graph.
        
        Args:
            graph: Graph to search
            start: Start node
            goal: Goal node, or None to compute single-source shortest paths
            
        Returns:
            SearchResult[T]: Search result
        """
        start_time = time.time()
        
        # Check if start node doesn't exist
        if not graph.has_node(start):
            return SearchResult(
                success=False,
                computation_time=time.time() - start_time
            )
        
        # Check if goal node doesn't exist
        if goal is not None and not graph.has_node(goal):
            return SearchResult(
                success=False,
                computation_time=time.time() - start_time
            )
        
        # Initialize
        dist = {node: float('inf') for node in graph.nodes}
        prev = {node: None for node in graph.nodes}
        dist[start] = 0
        visited = set()
        expanded = 0
        
        # Priority queue
        pq = [(0, start)]  # (cost, node)
        
        # Dijkstra's algorithm
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            # Skip if already visited
            if current in visited:
                continue
            
            # Mark as visited
            visited.add(current)
            expanded += 1
            
            # Check if goal reached
            if goal is not None and current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node:
                    path.append(node)
                    node = prev[node]
                path.reverse()
                
                return SearchResult(
                    success=True,
                    path=path,
                    cost=dist[goal],
                    visited=visited,
                    expanded=expanded,
                    computation_time=time.time() - start_time,
                    metadata={"distances": dist, "predecessors": prev}
                )
            
            # Update distances
            for edge in graph.get_edges(current):
                neighbor = edge.target
                weight = edge.cost
                distance = dist[current] + weight
                
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    prev[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        # If we're computing single-source shortest paths
        if goal is None:
            return SearchResult(
                success=True,
                visited=visited,
                expanded=expanded,
                computation_time=time.time() - start_time,
                metadata={"distances": dist, "predecessors": prev}
            )
        
        # No path found
        return SearchResult(
            success=False,
            visited=visited,
            expanded=expanded,
            computation_time=time.time() - start_time,
            metadata={"distances": dist, "predecessors": prev}
        )
    
    @staticmethod
    @timed("a_star")
    def a_star(graph: Graph[T], start: T, goal: T, 
              heuristic: Callable[[T, T], float]) -> SearchResult[T]:
        """
        Perform A* search on a graph.
        
        Args:
            graph: Graph to search
            start: Start node
            goal: Goal node
            heuristic: Heuristic function h(node, goal) -> estimated cost
            
        Returns:
            SearchResult[T]: Search result
        """
        start_time = time.time()
        
        # Check if start or goal nodes don't exist
        if not graph.has_node(start) or not graph.has_node(goal):
            return SearchResult(
                success=False,
                computation_time=time.time() - start_time
            )
        
        # Initialize
        g_score = {node: float('inf') for node in graph.nodes}
        f_score = {node: float('inf') for node in graph.nodes}
        g_score[start] = 0
        f_score[start] = heuristic(start, goal)
        
        open_set = set([start])
        closed_set = set()
        prev = {node: None for node in graph.nodes}
        expanded = 0
        
        # Priority queue
        pq = [(f_score[start], start)]  # (f_score, node)
        
        # A* algorithm
        while open_set:
            _, current = heapq.heappop(pq)
            
            # Remove from open set
            open_set.remove(current)
            
            # Check if goal reached
            if current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node:
                    path.append(node)
                    node = prev[node]
                path.reverse()
                
                return SearchResult(
                    success=True,
                    path=path,
                    cost=g_score[goal],
                    visited=open_set.union(closed_set),
                    expanded=expanded,
                    computation_time=time.time() - start_time,
                    metadata={"g_scores": g_score, "f_scores": f_score}
                )
            
            # Move to closed set
            closed_set.add(current)
            expanded += 1
            
            # Expand neighbors
            for edge in graph.get_edges(current):
                neighbor = edge.target
                
                # Skip if in closed set
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + edge.cost
                
                # Check if new path is better
                if tentative_g < g_score[neighbor]:
                    # Update path
                    prev[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    
                    # Add to open set if not already there
                    if neighbor not in open_set:
                        open_set.add(neighbor)
                        heapq.heappush(pq, (f_score[neighbor], neighbor))
        
        # No path found
        return SearchResult(
            success=False,
            visited=open_set.union(closed_set),
            expanded=expanded,
            computation_time=time.time() - start_time,
            metadata={"g_scores": g_score, "f_scores": f_score}
        )
    
    @staticmethod
    @timed("bidirectional_search")
    def bidirectional_search(graph: Graph[T], start: T, goal: T) -> SearchResult[T]:
        """
        Perform bidirectional breadth-first search on a graph.
        
        Args:
            graph: Graph to search
            start: Start node
            goal: Goal node
            
        Returns:
            SearchResult[T]: Search result
        """
        start_time = time.time()
        
        # Check if start or goal nodes don't exist
        if not graph.has_node(start) or not graph.has_node(goal):
            return SearchResult(
                success=False,
                computation_time=time.time() - start_time
            )
        
        # Initialize forward search
        forward_visited = {start: None}  # Maps node to its predecessor
        forward_queue = [start]
        
        # Initialize backward search
        backward_visited = {goal: None}  # Maps node to its predecessor
        backward_queue = [goal]
        
        # Intersection node
        intersection = None
        expanded = 0
        
        # Bidirectional BFS
        while forward_queue and backward_queue:
            # Expand forward
            if forward_queue:
                current = forward_queue.pop(0)
                expanded += 1
                
                for neighbor in graph.get_neighbors(current):
                    if neighbor not in forward_visited:
                        forward_visited[neighbor] = current
                        forward_queue.append(neighbor)
                    
                    # Check if intersection found
                    if neighbor in backward_visited:
                        intersection = neighbor
                        break
            
            # Check if intersection found
            if intersection:
                break
            
            # Expand backward
            if backward_queue:
                current = backward_queue.pop(0)
                expanded += 1
                
                for neighbor in graph.get_neighbors(current):
                    if neighbor not in backward_visited:
                        backward_visited[neighbor] = current
                        backward_queue.append(neighbor)
                    
                    # Check if intersection found
                    if neighbor in forward_visited:
                        intersection = neighbor
                        break
            
            # Check if intersection found
            if intersection:
                break
        
        # If intersection found, construct path
        if intersection:
            # Reconstruct forward path
            forward_path = []
            node = intersection
            while node:
                forward_path.append(node)
                node = forward_visited[node]
            forward_path.reverse()
            
            # Reconstruct backward path
            backward_path = []
            node = intersection
            while node:
                backward_path.append(node)
                node = backward_visited[node]
            backward_path.pop(0)  # Remove intersection node (already in forward_path)
            
            # Combine paths
            path = forward_path + backward_path
            
            # Calculate cost
            cost = 0
            for i in range(len(path) - 1):
                edge = graph.get_edge(path[i], path[i + 1])
                cost += edge.cost if edge else 1
            
            return SearchResult(
                success=True,
                path=path,
                cost=cost,
                visited=set(forward_visited.keys()).union(set(backward_visited.keys())),
                expanded=expanded,
                computation_time=time.time() - start_time
            )
        
        # No path found
        return SearchResult(
            success=False,
            visited=set(forward_visited.keys()).union(set(backward_visited.keys())),
            expanded=expanded,
            computation_time=time.time() - start_time
        )


# Specialized heuristics

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculate Manhattan distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        float: Manhattan distance
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        float: Euclidean distance
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def diagonal_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculate diagonal distance between two points (for 8-connected grid).
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        float: Diagonal distance
    """
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def chebyshev_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculate Chebyshev distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        float: Chebyshev distance (maximum of the absolute differences)
    """
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


# Path utility functions

def smooth_path(path: List[T], max_deviation: float = 1.0) -> List[T]:
    """
    Smooth a path by removing unnecessary waypoints.
    
    Uses the Ramer-Douglas-Peucker algorithm for path simplification.
    
    Args:
        path: Original path
        max_deviation: Maximum allowed deviation from original path
        
    Returns:
        List[T]: Smoothed path
    """
    if len(path) <= 2:
        return path
    
    # Define distance function for points
    def point_line_distance(point, line_start, line_end):
        if hasattr(point, 'x') and hasattr(point, 'y'):
            # If points have x, y attributes
            x, y = point.x, point.y
            x1, y1 = line_start.x, line_start.y
            x2, y2 = line_end.x, line_end.y
        else:
            # Assume points are tuples/lists
            x, y = point[0], point[1]
            x1, y1 = line_start[0], line_start[1]
            x2, y2 = line_end[0], line_end[1]
        
        # Handle case where line is just a point
        if x1 == x2 and y1 == y2:
            return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        
        # Calculate distance from point to line
        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return numerator / denominator
    
    # Ramer-Douglas-Peucker algorithm
    def rdp(points, epsilon):
        if len(points) <= 2:
            return points
        
        # Find point with maximum distance
        dmax = 0
        index = 0
        
        for i in range(1, len(points) - 1):
            d = point_line_distance(points[i], points[0], points[-1])
            if d > dmax:
                dmax = d
                index = i
        
        # If max distance > epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            rec_results1 = rdp(points[:index + 1], epsilon)
            rec_results2 = rdp(points[index:], epsilon)
            
            # Build the result list
            return rec_results1[:-1] + rec_results2
        else:
            return [points[0], points[-1]]
    
    # Apply RDP algorithm
    return rdp(path, max_deviation)