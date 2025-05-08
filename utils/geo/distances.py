"""
Distance calculation utilities for the mining dispatch system.

Provides various distance metrics and path length calculations.
"""

import math
from typing import List, Tuple, Union, Callable, Optional
from functools import lru_cache

from utils.geo.coordinates import Point2D, Point3D, normalize_to_point2d, normalize_to_point3d

# Define distance calculation types
DistanceFunc = Callable[[Union[Point2D, Point3D], Union[Point2D, Point3D]], float]


def euclidean_distance_2d(p1: Union[Tuple, List, Point2D], 
                          p2: Union[Tuple, List, Point2D]) -> float:
    """
    Calculate the 2D Euclidean distance between two points.
    
    Args:
        p1: First point
        p2: Second point
        
    Returns:
        float: Euclidean distance
    """
    point1 = normalize_to_point2d(p1)
    point2 = normalize_to_point2d(p2)
    return point1.distance_to(point2)


def euclidean_distance_3d(p1: Union[Tuple, List, Point2D, Point3D], 
                          p2: Union[Tuple, List, Point2D, Point3D]) -> float:
    """
    Calculate the 3D Euclidean distance between two points.
    
    Args:
        p1: First point
        p2: Second point
        
    Returns:
        float: 3D Euclidean distance
    """
    point1 = normalize_to_point3d(p1)
    point2 = normalize_to_point3d(p2)
    return point1.distance_to(point2)


def manhattan_distance(p1: Union[Tuple, List, Point2D], 
                      p2: Union[Tuple, List, Point2D]) -> float:
    """
    Calculate the Manhattan (L1) distance between two points.
    
    Args:
        p1: First point
        p2: Second point
        
    Returns:
        float: Manhattan distance
    """
    point1 = normalize_to_point2d(p1)
    point2 = normalize_to_point2d(p2)
    return abs(point1.x - point2.x) + abs(point1.y - point2.y)


@lru_cache(maxsize=1000)
def cached_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Cached version of Euclidean distance for frequently used calculations.
    Points must be passed as tuples for cache to work.
    
    Args:
        p1: First point as a tuple (x, y)
        p2: Second point as a tuple (x, y)
        
    Returns:
        float: Euclidean distance
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def weighted_distance(p1: Point2D, p2: Point2D, weights: Tuple[float, float] = (1.0, 1.0)) -> float:
    """
    Calculate weighted Euclidean distance with different weights for x and y.
    
    Args:
        p1: First point
        p2: Second point
        weights: Tuple of (x_weight, y_weight)
        
    Returns:
        float: Weighted distance
    """
    return math.sqrt(weights[0] * (p1.x - p2.x)**2 + weights[1] * (p1.y - p2.y)**2)


def path_length(points: List[Union[Point2D, Tuple, List]], 
               distance_func: DistanceFunc = euclidean_distance_2d) -> float:
    """
    Calculate the total length of a path consisting of multiple points.
    
    Args:
        points: List of points forming the path
        distance_func: Function to use for distance calculation between adjacent points
        
    Returns:
        float: Total path length
        
    Raises:
        ValueError: If path has fewer than 2 points
    """
    if len(points) < 2:
        return 0.0
        
    total_length = 0.0
    for i in range(len(points) - 1):
        total_length += distance_func(points[i], points[i+1])
    
    return total_length


def point_to_segment_distance(p: Point2D, segment_start: Point2D, segment_end: Point2D) -> float:
    """
    Calculate the shortest distance from a point to a line segment.
    
    Args:
        p: Point
        segment_start: Start point of the line segment
        segment_end: End point of the line segment
        
    Returns:
        float: Shortest distance from point to line segment
    """
    # Vector from segment_start to segment_end
    segment_vector_x = segment_end.x - segment_start.x
    segment_vector_y = segment_end.y - segment_start.y
    
    # Vector from segment_start to p
    point_vector_x = p.x - segment_start.x
    point_vector_y = p.y - segment_start.y
    
    # Squared length of the segment
    segment_length_squared = segment_vector_x**2 + segment_vector_y**2
    
    # Handle degenerate case where segment is a point
    if segment_length_squared < 1e-10:
        return euclidean_distance_2d(p, segment_start)
    
    # Calculate projection ratio (0-1 means point projects onto segment)
    projection_ratio = (point_vector_x * segment_vector_x + 
                       point_vector_y * segment_vector_y) / segment_length_squared
    
    if projection_ratio < 0:
        # Closest to segment_start
        return euclidean_distance_2d(p, segment_start)
    elif projection_ratio > 1:
        # Closest to segment_end
        return euclidean_distance_2d(p, segment_end)
    else:
        # Closest to some point on the segment
        closest_x = segment_start.x + projection_ratio * segment_vector_x
        closest_y = segment_start.y + projection_ratio * segment_vector_y
        closest_point = Point2D(closest_x, closest_y)
        return euclidean_distance_2d(p, closest_point)


def segment_to_segment_distance(s1_start: Point2D, s1_end: Point2D, 
                               s2_start: Point2D, s2_end: Point2D) -> float:
    """
    Calculate the shortest distance between two line segments.
    
    Args:
        s1_start: Start point of first segment
        s1_end: End point of first segment
        s2_start: Start point of second segment
        s2_end: End point of second segment
        
    Returns:
        float: Shortest distance between the segments
    """
    # Check if segments intersect
    if segments_intersect(s1_start, s1_end, s2_start, s2_end):
        return 0.0
    
    # Calculate distances from each endpoint to the other segment
    d1 = point_to_segment_distance(s1_start, s2_start, s2_end)
    d2 = point_to_segment_distance(s1_end, s2_start, s2_end)
    d3 = point_to_segment_distance(s2_start, s1_start, s1_end)
    d4 = point_to_segment_distance(s2_end, s1_start, s1_end)
    
    # Return minimum distance
    return min(d1, d2, d3, d4)


def segments_intersect(s1_start: Point2D, s1_end: Point2D, 
                       s2_start: Point2D, s2_end: Point2D) -> bool:
    """
    Check if two line segments intersect.
    
    Args:
        s1_start: Start point of first segment
        s1_end: End point of first segment
        s2_start: Start point of second segment
        s2_end: End point of second segment
        
    Returns:
        bool: True if segments intersect, False otherwise
    """
    def orientation(p, q, r):
        """
        Calculate orientation of triplet (p, q, r).
        Returns:
         0 --> Collinear
         1 --> Clockwise
         2 --> Counterclockwise
        """
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if abs(val) < 1e-9:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise
    
    def on_segment(p, q, r):
        """Check if point q lies on line segment pr."""
        return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))
    
    # Calculate orientations
    o1 = orientation(s1_start, s1_end, s2_start)
    o2 = orientation(s1_start, s1_end, s2_end)
    o3 = orientation(s2_start, s2_end, s1_start)
    o4 = orientation(s2_start, s2_end, s1_end)
    
    # General case
    if o1 != o2 and o3 != o4:
        return True
    
    # Special cases (collinear points)
    if o1 == 0 and on_segment(s1_start, s2_start, s1_end):
        return True
    if o2 == 0 and on_segment(s1_start, s2_end, s1_end):
        return True
    if o3 == 0 and on_segment(s2_start, s1_start, s2_end):
        return True
    if o4 == 0 and on_segment(s2_start, s1_end, s2_end):
        return True
    
    return False


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        
    Returns:
        float: Distance in meters
    """
    # Earth radius in meters
    radius = 6371000.0
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = radius * c
    
    return distance


def point_to_polygon_distance(point: Point2D, polygon: List[Point2D]) -> float:
    """
    Calculate the minimum distance from a point to a polygon.
    
    Args:
        point: The point
        polygon: List of points forming a polygon (assume closed)
        
    Returns:
        float: Minimum distance from point to polygon edge
        
    Raises:
        ValueError: If polygon has fewer than 3 points
    """
    if len(polygon) < 3:
        raise ValueError("Polygon must have at least 3 points")
    
    # If point is inside polygon, distance is 0
    if point_in_polygon(point, polygon):
        return 0.0
    
    # Calculate minimum distance to any edge
    min_distance = float('inf')
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        segment_start = polygon[i]
        segment_end = polygon[j]
        
        distance = point_to_segment_distance(point, segment_start, segment_end)
        min_distance = min(min_distance, distance)
    
    return min_distance


def point_in_polygon(point: Point2D, polygon: List[Point2D]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: The point to check
        polygon: List of points forming a polygon (assume closed)
        
    Returns:
        bool: True if point is inside polygon, False otherwise
    """
    if len(polygon) < 3:
        return False
    
    inside = False
    j = len(polygon) - 1
    
    for i in range(len(polygon)):
        # Check if the ray from point to positive x-direction intersects with the edge
        if ((polygon[i].y > point.y) != (polygon[j].y > point.y)) and \
           (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / 
            (polygon[j].y - polygon[i].y) + polygon[i].x):
            inside = not inside
        j = i
    
    return inside