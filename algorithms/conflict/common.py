import math
from enum import Enum, auto
from typing import List, Dict, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass, field
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from utils.geo.coordinates import Point2D
from utils.math.trajectories import Path
from utils.logger import get_logger

logger = get_logger(__name__)

class ConflictType(Enum):
    """冲突类型枚举"""
    VERTEX = auto()  # 节点冲突（两车同一时间在同一位置）
    EDGE = auto()    # 边冲突（两车在同一时间段内交换位置）
    FOLLOWING = auto()  # 跟随冲突（两车距离过近）
    DEADLOCK = auto()   # 死锁冲突（互相阻塞）

class ConstraintType(Enum):
    """约束类型枚举"""
    VERTEX = auto()  # 禁止在特定时间点访问特定位置
    EDGE = auto()    # 禁止在特定时间段内通过特定边
    TEMPORAL = auto() # 时间约束（延迟车辆出发时间）

@dataclass
class Conflict:
    """冲突数据结构"""
    conflict_type: ConflictType
    vehicle_a_id: str
    vehicle_b_id: str
    location_a: Point2D
    location_b: Point2D
    time_step: int
    # 对于边冲突，需要记录先前位置
    prev_location_a: Optional[Point2D] = None
    prev_location_b: Optional[Point2D] = None
    # 额外信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (f"Conflict({self.conflict_type.name}, "
                f"vehicles: [{self.vehicle_a_id}, {self.vehicle_b_id}], "
                f"time: {self.time_step})")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "conflict_type": self.conflict_type.name,
            "vehicle_a_id": self.vehicle_a_id,
            "vehicle_b_id": self.vehicle_b_id,
            "location_a": [self.location_a.x, self.location_a.y],
            "location_b": [self.location_b.x, self.location_b.y],
            "time_step": self.time_step,
            "prev_location_a": [self.prev_location_a.x, self.prev_location_a.y] if self.prev_location_a else None,
            "prev_location_b": [self.prev_location_b.x, self.prev_location_b.y] if self.prev_location_b else None,
            "metadata": self.metadata
        }

@dataclass
class Constraint:
    """约束数据结构"""
    constraint_type: ConstraintType
    vehicle_id: str
    location: Point2D
    time_step: int
    # 对于边约束，需要记录目标位置
    target_location: Optional[Point2D] = None
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """哈希函数以支持在集合中使用"""
        if self.target_location:
            return hash((self.constraint_type, self.vehicle_id, self.location.x, self.location.y, 
                        self.target_location.x, self.target_location.y, self.time_step))
        return hash((self.constraint_type, self.vehicle_id, self.location.x, self.location.y, self.time_step))
    
    def __eq__(self, other):
        """判断两个约束是否相等"""
        if not isinstance(other, Constraint):
            return False
        
        if self.constraint_type != other.constraint_type or self.vehicle_id != other.vehicle_id or self.time_step != other.time_step:
            return False
            
        if abs(self.location.x - other.location.x) > 1e-6 or abs(self.location.y - other.location.y) > 1e-6:
            return False
            
        if self.target_location and other.target_location:
            return (abs(self.target_location.x - other.target_location.x) <= 1e-6 and 
                    abs(self.target_location.y - other.target_location.y) <= 1e-6)
        
        return self.target_location is None and other.target_location is None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "constraint_type": self.constraint_type.name,
            "vehicle_id": self.vehicle_id,
            "location": [self.location.x, self.location.y],
            "time_step": self.time_step,
            "metadata": self.metadata
        }
        
        if self.target_location:
            result["target_location"] = [self.target_location.x, self.target_location.y]
            
        return result

@dataclass
class PathWithTimesteps:
    """带时间步的路径"""
    vehicle_id: str
    points: List[Point2D]
    timesteps: List[int]
    
    def __post_init__(self):
        """确保时间步和点的数量一致"""
        if len(self.points) != len(self.timesteps):
            raise ValueError(f"Points list length ({len(self.points)}) must match timesteps list length ({len(self.timesteps)})")
    
    def position_at_time(self, timestep: int) -> Optional[Point2D]:
        """获取指定时间的位置"""
        if not self.timesteps:
            return None
            
        # 如果时间步在路径时间范围之前，返回起点
        if timestep <= self.timesteps[0]:
            return self.points[0]
            
        # 如果时间步在路径时间范围之后，返回终点
        if timestep >= self.timesteps[-1]:
            return self.points[-1]
            
        # 查找时间步对应的点
        for i, ts in enumerate(self.timesteps):
            if ts == timestep:
                return self.points[i]
                
        # 如果没有精确匹配，进行插值
        for i in range(len(self.timesteps) - 1):
            if self.timesteps[i] < timestep < self.timesteps[i+1]:
                # 线性插值
                t = (timestep - self.timesteps[i]) / (self.timesteps[i+1] - self.timesteps[i])
                x = self.points[i].x + t * (self.points[i+1].x - self.points[i].x)
                y = self.points[i].y + t * (self.points[i+1].y - self.points[i].y)
                return Point2D(x, y)
                
        return None
    
    def get_timestep_range(self) -> Tuple[int, int]:
        """获取时间范围"""
        if not self.timesteps:
            return (0, 0)
        return (min(self.timesteps), max(self.timesteps))
    
    def violates_constraint(self, constraint: Constraint) -> bool:
        """检查路径是否违反约束"""
        if constraint.vehicle_id != self.vehicle_id:
            return False
            
        if constraint.constraint_type == ConstraintType.VERTEX:
            # 顶点约束: 在特定时间不能位于特定位置
            pos = self.position_at_time(constraint.time_step)
            if pos:
                return (abs(pos.x - constraint.location.x) < 1e-6 and 
                        abs(pos.y - constraint.location.y) < 1e-6)
        
        elif constraint.constraint_type == ConstraintType.EDGE:
            # 边约束: 在特定时间段不能通过特定边
            if constraint.time_step >= len(self.timesteps) - 1:
                return False
                
            idx = None
            for i, ts in enumerate(self.timesteps):
                if ts == constraint.time_step:
                    idx = i
                    break
                    
            if idx is not None and idx < len(self.points) - 1:
                curr_pos = self.points[idx]
                next_pos = self.points[idx + 1]
                
                return (abs(curr_pos.x - constraint.location.x) < 1e-6 and 
                        abs(curr_pos.y - constraint.location.y) < 1e-6 and
                        abs(next_pos.x - constraint.target_location.x) < 1e-6 and 
                        abs(next_pos.y - constraint.target_location.y) < 1e-6)
        
        elif constraint.constraint_type == ConstraintType.TEMPORAL:
            # 时间约束: 延迟出发时间
            return self.timesteps[0] < constraint.time_step
            
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "vehicle_id": self.vehicle_id,
            "points": [[p.x, p.y] for p in self.points],
            "timesteps": self.timesteps
        }

@dataclass
class CBSNode:
    """CBS搜索树节点"""
    constraints: Set[Constraint] = field(default_factory=set)
    paths: Dict[str, PathWithTimesteps] = field(default_factory=dict)
    cost: float = 0.0
    conflicts: List[Conflict] = field(default_factory=list)
    parent: Optional['CBSNode'] = None
    
    def __post_init__(self):
        """计算初始成本"""
        self.update_cost()
    
    def update_cost(self):
        """更新节点成本（所有路径的总长度）"""
        self.cost = sum(len(path.points) for path in self.paths.values())
    
    def add_constraint(self, constraint: Constraint):
        """添加约束"""
        self.constraints.add(constraint)
    
    def get_constraints_for_vehicle(self, vehicle_id: str) -> List[Constraint]:
        """获取特定车辆的约束列表"""
        return [c for c in self.constraints if c.vehicle_id == vehicle_id]
    
    def is_solution(self) -> bool:
        """检查是否为解决方案（无冲突）"""
        return len(self.conflicts) == 0
    
    def __lt__(self, other):
        """比较函数，用于优先队列排序"""
        if not isinstance(other, CBSNode):
            return NotImplemented
        return self.cost < other.cost


def detect_conflicts(paths: Dict[str, PathWithTimesteps], 
                    vehicle_radius: float = 1.0,
                    time_horizon: int = 100) -> List[Conflict]:
    """
    检测多个车辆路径之间的冲突
    
    Args:
        paths: 车辆ID到带时间步路径的映射
        vehicle_radius: 车辆半径，用于碰撞检测
        time_horizon: 检测冲突的时间步范围
        
    Returns:
        冲突列表
    """
    conflicts = []
    vehicle_ids = list(paths.keys())
    
    # 遍历所有车辆对
    for i in range(len(vehicle_ids)):
        vehicle_a_id = vehicle_ids[i]
        path_a = paths[vehicle_a_id]
        
        for j in range(i + 1, len(vehicle_ids)):
            vehicle_b_id = vehicle_ids[j]
            path_b = paths[vehicle_b_id]
            
            # 确定时间范围
            min_time = max(path_a.timesteps[0], path_b.timesteps[0])
            max_time = min(path_a.timesteps[-1], path_b.timesteps[-1], min_time + time_horizon)
            
            # 检查每个时间步
            for t in range(min_time, max_time + 1):
                pos_a = path_a.position_at_time(t)
                pos_b = path_b.position_at_time(t)
                
                if pos_a and pos_b:
                    # 检查顶点冲突 (两车同一时间在同一位置)
                    distance = pos_a.distance_to(pos_b)
                    if distance < 2 * vehicle_radius:
                        conflicts.append(Conflict(
                            conflict_type=ConflictType.VERTEX,
                            vehicle_a_id=vehicle_a_id,
                            vehicle_b_id=vehicle_b_id,
                            location_a=pos_a,
                            location_b=pos_b,
                            time_step=t,
                            metadata={"distance": distance}
                        ))
                        continue
                
                # 检查边冲突 (两车在相邻时间步交换位置)
                if t < max_time:
                    next_pos_a = path_a.position_at_time(t + 1)
                    next_pos_b = path_b.position_at_time(t + 1)
                    
                    if pos_a and pos_b and next_pos_a and next_pos_b:
                        if (abs(pos_a.x - next_pos_b.x) < 1e-6 and abs(pos_a.y - next_pos_b.y) < 1e-6 and
                            abs(pos_b.x - next_pos_a.x) < 1e-6 and abs(pos_b.y - next_pos_a.y) < 1e-6):
                            conflicts.append(Conflict(
                                conflict_type=ConflictType.EDGE,
                                vehicle_a_id=vehicle_a_id,
                                vehicle_b_id=vehicle_b_id,
                                location_a=pos_a,
                                location_b=pos_b,
                                prev_location_a=next_pos_a,
                                prev_location_b=next_pos_b,
                                time_step=t,
                                metadata={}
                            ))
                            
    return conflicts


def create_constraints_from_conflict(conflict: Conflict) -> Tuple[Constraint, Constraint]:
    """
    从冲突创建约束对
    
    Args:
        conflict: 冲突实例
        
    Returns:
        两个约束，分别应用于冲突中的两个车辆
    """
    if conflict.conflict_type == ConflictType.VERTEX:
        # 顶点冲突: 创建顶点约束
        constraint_a = Constraint(
            constraint_type=ConstraintType.VERTEX,
            vehicle_id=conflict.vehicle_a_id,
            location=conflict.location_a,
            time_step=conflict.time_step
        )
        
        constraint_b = Constraint(
            constraint_type=ConstraintType.VERTEX,
            vehicle_id=conflict.vehicle_b_id,
            location=conflict.location_b,
            time_step=conflict.time_step
        )
        
        return constraint_a, constraint_b
        
    elif conflict.conflict_type == ConflictType.EDGE:
        # 边冲突: 创建边约束
        constraint_a = Constraint(
            constraint_type=ConstraintType.EDGE,
            vehicle_id=conflict.vehicle_a_id,
            location=conflict.location_a,
            target_location=conflict.prev_location_a,
            time_step=conflict.time_step
        )
        
        constraint_b = Constraint(
            constraint_type=ConstraintType.EDGE,
            vehicle_id=conflict.vehicle_b_id,
            location=conflict.location_b,
            target_location=conflict.prev_location_b,
            time_step=conflict.time_step
        )
        
        return constraint_a, constraint_b
    
    # 其他类型冲突: 默认创建顶点约束
    logger.warning(f"Unknown conflict type {conflict.conflict_type}, creating vertex constraints")
    constraint_a = Constraint(
        constraint_type=ConstraintType.VERTEX,
        vehicle_id=conflict.vehicle_a_id,
        location=conflict.location_a,
        time_step=conflict.time_step
    )
    
    constraint_b = Constraint(
        constraint_type=ConstraintType.VERTEX,
        vehicle_id=conflict.vehicle_b_id,
        location=conflict.location_b,
        time_step=conflict.time_step
    )
    
    return constraint_a, constraint_b


def path_to_timesteps(path: Path, speed: float = 1.0, start_time: int = 0) -> PathWithTimesteps:
    """
    将路径转换为带时间步的路径
    
    Args:
        path: 原始路径
        speed: 车辆速度 (单位/时间步)
        start_time: 起始时间步
        
    Returns:
        带时间步的路径
    """
    if not path.points:
        return PathWithTimesteps(vehicle_id="", points=[], timesteps=[])
    
    points = path.points.copy()
    timesteps = [start_time]
    
    # 计算每个点的时间步
    for i in range(1, len(points)):
        distance = points[i-1].distance_to(points[i])
        time_delta = math.ceil(distance / speed)
        timesteps.append(timesteps[-1] + time_delta)
    
    return PathWithTimesteps(vehicle_id="", points=points, timesteps=timesteps)


def timesteps_to_path(path_with_timesteps: PathWithTimesteps) -> Path:
    """
    将带时间步的路径转换为普通路径
    
    Args:
        path_with_timesteps: 带时间步的路径
        
    Returns:
        普通路径
    """
    return Path(points=path_with_timesteps.points)