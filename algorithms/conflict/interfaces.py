import abc
from enum import Enum, auto
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from utils.math.trajectories import Path
from utils.logger import get_logger

logger = get_logger(__name__)

class ConflictResolverType(Enum):
    """冲突解决器类型枚举"""
    NONE = auto()
    CBS = auto()  # Conflict-Based Search
    ECBS = auto() # Enhanced Conflict-Based Search
    PRIORITY = auto() # 基于优先级的冲突解决
    TEMPORAL = auto() # 时间分离法

@dataclass
class ConflictResolutionConfig:
    """冲突解决配置"""
    resolver_type: ConflictResolverType = ConflictResolverType.CBS
    vehicle_radius: float = 1.0
    time_horizon: int = 100
    max_runtime: float = 10.0
    max_iterations: int = 1000
    max_constraints: int = 100
    

class ConflictResolver(abc.ABC):
    """冲突解决器抽象基类"""
    
    @abc.abstractmethod
    def find_conflicts(self, paths: Dict[str, Path], 
                      vehicles: Dict[str, Any] = None) -> List[Any]:
        """
        检测给定路径集合中的冲突
        
        Args:
            paths: 车辆ID到路径的映射
            vehicles: 车辆ID到车辆对象的映射(可选)，用于获取车辆属性
            
        Returns:
            冲突列表
        """
        pass
        
    @abc.abstractmethod
    def resolve_conflicts(self, paths: Dict[str, Path], 
                         path_planner: Any,
                         vehicles: Dict[str, Any] = None, 
                         obstacles: List[Any] = None) -> Dict[str, Path]:
        """
        解决给定路径集合中的冲突
        
        Args:
            paths: 车辆ID到路径的映射
            path_planner: 用于规划单个车辆路径的规划器
            vehicles: 车辆ID到车辆对象的映射
            obstacles: 障碍物列表
            
        Returns:
            无冲突的路径映射表
        """
        pass


class ConflictResolverFactory:
    """冲突解决器工厂类"""
    
    _resolvers = {}
    
    @classmethod
    def register_resolver(cls, resolver_type: ConflictResolverType, resolver_class):
        """注册冲突解决器类"""
        cls._resolvers[resolver_type] = resolver_class
        logger.info(f"Registered conflict resolver: {resolver_type.name}")
    
    @classmethod
    def create_resolver(cls, resolver_type: ConflictResolverType, config: Optional[Any] = None) -> ConflictResolver:
        """创建冲突解决器实例"""
        if resolver_type not in cls._resolvers:
            logger.warning(f"No resolver registered for type {resolver_type.name}, using CBS")
            resolver_type = ConflictResolverType.CBS
        
        resolver_class = cls._resolvers[resolver_type]
        return resolver_class(config)
    
    @classmethod
    def get_available_resolvers(cls) -> List[ConflictResolverType]:
        """获取可用的冲突解决器类型"""
        return list(cls._resolvers.keys())


def register_default_resolvers():
    """注册默认冲突解决器"""
    from algorithms.conflict.cbs import CBSResolver, CBSConfig
    
    ConflictResolverFactory.register_resolver(
        ConflictResolverType.CBS,
        CBSResolver
    )
    
    # 其他解决器注册可以在这里添加
    # ConflictResolverFactory.register_resolver(
    #     ConflictResolverType.ECBS,
    #     ECBSResolver
    # )