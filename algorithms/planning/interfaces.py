"""
Path planning interfaces for the mining dispatch system.

Defines abstract interfaces for path planners and common data structures,
allowing different planning algorithms to be used interchangeably.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import abc
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Union, Tuple, Type
from dataclasses import dataclass

from utils.geo.coordinates import Point2D
from utils.logger import get_logger

from algorithms.planning.common import (
    PlanningConfig, PlanningConstraints, PlanningResult, PlanningStatus
)

# Get logger
logger = get_logger("planning.interfaces")


class PlannerType(Enum):
    """Types of path planners available in the system."""
    HYBRID_ASTAR = auto()       # Hybrid A* algorithm
    REEDS_SHEPP = auto()        # Reeds-Shepp curves
    RRT = auto()                # Rapidly-exploring Random Tree
    RRT_STAR = auto()           # RRT* (optimized RRT)
    STATE_LATTICE = auto()      # State lattice planner
    DUBINS = auto()             # Dubins curves


class PathPlanner(abc.ABC):
    """
    Abstract base class for all path planners.
    
    Defines the common interface that all path planners must implement.
    """
    
    @abc.abstractmethod
    def plan_path(self, start: Point2D, start_heading: float,
                 goal: Point2D, goal_heading: float,
                 obstacles: Optional[List[Any]] = None) -> PlanningResult:
        """
        Plan a path from start to goal.
        
        Args:
            start: Start point
            start_heading: Start heading in radians
            goal: Goal point
            goal_heading: Goal heading in radians
            obstacles: Optional list of obstacles
            
        Returns:
            PlanningResult: Path planning result
        """
        pass
    
    @abc.abstractmethod
    def set_config(self, config: PlanningConfig) -> None:
        """
        Update the planner configuration.
        
        Args:
            config: New planning configuration
        """
        pass
    
    @abc.abstractmethod
    def get_config(self) -> PlanningConfig:
        """
        Get the current planner configuration.
        
        Returns:
            PlanningConfig: Current configuration
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def get_planner_type(cls) -> PlannerType:
        """
        Get the type of this planner.
        
        Returns:
            PlannerType: Planner type
        """
        pass


class HybridAStarPlannerBase(PathPlanner):
    """Base interface for Hybrid A* planners."""
    
    @classmethod
    def get_planner_type(cls) -> PlannerType:
        """Get the planner type."""
        return PlannerType.HYBRID_ASTAR


class ReedsSheppPlannerBase(PathPlanner):
    """Base interface for Reeds-Shepp curve planners."""
    
    @classmethod
    def get_planner_type(cls) -> PlannerType:
        """Get the planner type."""
        return PlannerType.REEDS_SHEPP


class RRTPlannerBase(PathPlanner):
    """Base interface for RRT planners."""
    
    @classmethod
    def get_planner_type(cls) -> PlannerType:
        """Get the planner type."""
        return PlannerType.RRT


class PlannerFactory:
    """
    Factory class for creating path planners.
    
    Provides methods to create different types of path planners with
    appropriate configurations.
    """
    
    _planner_registry: Dict[PlannerType, Type[PathPlanner]] = {}
    
    @classmethod
    def register_planner(cls, planner_type: PlannerType, planner_class: Type[PathPlanner]) -> None:
        """
        Register a planner implementation with the factory.
        
        Args:
            planner_type: Type of planner
            planner_class: Planner implementation class
        """
        cls._planner_registry[planner_type] = planner_class
        logger.info(f"Registered planner: {planner_type.name} -> {planner_class.__name__}")
    
    @classmethod
    def create_planner(cls, planner_type: PlannerType, config: Optional[PlanningConfig] = None) -> PathPlanner:
        """
        Create a planner of the specified type.
        
        Args:
            planner_type: Type of planner to create
            config: Optional configuration for the planner
            
        Returns:
            PathPlanner: Instantiated planner
            
        Raises:
            ValueError: If planner type is not registered
        """
        if planner_type not in cls._planner_registry:
            raise ValueError(f"Planner type {planner_type} not registered")
        
        planner_class = cls._planner_registry[planner_type]
        config = config or PlanningConfig()
        
        logger.debug(f"Creating planner of type {planner_type.name}")
        return planner_class(config)
    
    @classmethod
    def create_default_planner(cls, config: Optional[PlanningConfig] = None) -> PathPlanner:
        """
        Create a planner with the default type (Hybrid A*).
        
        Args:
            config: Optional configuration for the planner
            
        Returns:
            PathPlanner: Instantiated planner
        """
        return cls.create_planner(PlannerType.HYBRID_ASTAR, config)
    
    @classmethod
    def get_available_planners(cls) -> List[PlannerType]:
        """
        Get a list of all registered planner types.
        
        Returns:
            List[PlannerType]: List of available planner types
        """
        return list(cls._planner_registry.keys())


# Import concrete implementations (in actual use, you would implement 
# proper class loading to avoid circular imports)
def register_default_planners():
    """Register the default planner implementations."""
    try:
        # Import concrete implementations
        from algorithms.planning.hybrid_astar import HybridAStarPlanner
        from algorithms.planning.reeds_shepp import ReedsSheppPlanner
        
        # Ensure they implement the interfaces correctly
        class HybridAStarPlannerImpl(HybridAStarPlanner, HybridAStarPlannerBase):
            """Hybrid A* planner implementation."""
            
            def plan_path(self, start: Point2D, start_heading: float,
                        goal: Point2D, goal_heading: float,
                        obstacles: Optional[List[Any]] = None) -> PlanningResult:
                """Plan a path using Hybrid A*."""
                return super().plan_path(start, start_heading, goal, goal_heading, obstacles)
            
            def set_config(self, config: PlanningConfig) -> None:
                """Update planner configuration."""
                self.config = config
            
            def get_config(self) -> PlanningConfig:
                """Get current configuration."""
                return self.config
        
        class ReedsSheppPlannerImpl(ReedsSheppPlanner, ReedsSheppPlannerBase):
            """Reeds-Shepp planner implementation."""
            
            def plan_path(self, start: Point2D, start_heading: float,
                        goal: Point2D, goal_heading: float,
                        obstacles: Optional[List[Any]] = None) -> PlanningResult:
                """Plan a path using Reeds-Shepp curves."""
                return self.plan(
                    start.x, start.y, start_heading,
                    goal.x, goal.y, goal_heading,
                    obstacles
                )
            
            def set_config(self, config: PlanningConfig) -> None:
                """Update planner configuration."""
                self.config = config
                self.min_turning_radius = config.constraints.min_turning_radius
            
            def get_config(self) -> PlanningConfig:
                """Get current configuration."""
                return self.config
        
        # Register with factory
        PlannerFactory.register_planner(PlannerType.HYBRID_ASTAR, HybridAStarPlannerImpl)
        PlannerFactory.register_planner(PlannerType.REEDS_SHEPP, ReedsSheppPlannerImpl)
        
        logger.info("Default planners registered successfully")
    except ImportError as e:
        logger.warning(f"Could not register all default planners: {str(e)}")


# Register default planners when module is imported
register_default_planners()