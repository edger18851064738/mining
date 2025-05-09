import heapq
import time
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import copy
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from utils.geo.coordinates import Point2D
from utils.math.trajectories import Path
from utils.logger import get_logger, timed

from algorithms.planning.common import (
    PlanningStatus, PlanningResult, PlanningConfig, PlanningConstraints
)
from algorithms.conflict.common import (
    Conflict, Constraint, PathWithTimesteps, CBSNode, 
    ConflictType, ConstraintType, detect_conflicts, 
    create_constraints_from_conflict, path_to_timesteps, timesteps_to_path
)
from algorithms.conflict.interfaces import ConflictResolver

logger = get_logger(__name__)

@dataclass
class CBSConfig:
    """CBS算法配置"""
    max_iterations: int = 1000
    max_runtime: float = 10.0  # 秒
    vehicle_radius: float = 1.0
    time_horizon: int = 100
    default_speed: float = 1.0
    max_constraints: int = 100
    use_cardinal_conflicts: bool = True
    use_disjoint_splitting: bool = True
    use_bypass: bool = True
    use_priority_inheritance: bool = True


class CBSResolver(ConflictResolver):
    """基于冲突的搜索(CBS)算法实现"""
    
    def __init__(self, config: Optional[CBSConfig] = None):
        """
        初始化CBS解算器
        
        Args:
            config: CBS配置，如果为None则使用默认配置
        """
        self.config = config or CBSConfig()
        self.path_planner = None  # 将在resolve_conflicts中设置
        self.open_set = []  # 用于CBS的优先队列
        self.closed_set = set()  # 已探索节点集合
        logger.info(f"Initialized CBS resolver with config: {self.config}")
    
    @timed("cbs_find_conflicts")
    def find_conflicts(self, paths: Dict[str, Path], 
                      vehicles: Dict[str, Any] = None) -> List[Conflict]:
        """
        检测给定路径集合中的冲突
        
        Args:
            paths: 车辆ID到路径的映射
            vehicles: 车辆ID到车辆对象的映射(可选)，用于获取车辆属性
            
        Returns:
            冲突列表
        """
        # 转换为带时间步的路径
        paths_with_timesteps = {}
        for vehicle_id, path in paths.items():
            # 获取车辆速度，如果没有提供车辆则使用默认速度
            speed = self.config.default_speed
            if vehicles and vehicle_id in vehicles:
                speed = getattr(vehicles[vehicle_id], 'max_speed', self.config.default_speed)
            
            path_ts = path_to_timesteps(path, speed=speed)
            path_ts.vehicle_id = vehicle_id
            paths_with_timesteps[vehicle_id] = path_ts
        
        # 检测冲突
        conflicts = detect_conflicts(
            paths_with_timesteps, 
            vehicle_radius=self.config.vehicle_radius,
            time_horizon=self.config.time_horizon
        )
        
        logger.debug(f"Found {len(conflicts)} conflicts among {len(paths)} paths")
        return conflicts
    
    @timed("cbs_resolve_conflicts")
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
        self.path_planner = path_planner
        start_time = time.time()
        
        # 初始化CBS搜索树
        root = self._create_root_node(paths, vehicles)
        
        if root.is_solution():
            logger.info("Initial paths have no conflicts, returning as is")
            return paths
        
        # 重置搜索状态
        self.open_set = [root]
        self.closed_set = set()
        num_iterations = 0
        
        # 开始CBS搜索
        while self.open_set and num_iterations < self.config.max_iterations:
            # 检查时间限制
            if time.time() - start_time > self.config.max_runtime:
                logger.warning(f"CBS timeout after {num_iterations} iterations")
                # 返回当前最佳解决方案
                best_node = min(self.open_set, key=lambda node: len(node.conflicts))
                return self._node_to_paths(best_node)
            
            # 弹出成本最低的节点
            current = heapq.heappop(self.open_set)
            
            # 如果没有冲突，找到解决方案
            if current.is_solution():
                logger.info(f"Found conflict-free solution after {num_iterations} iterations")
                return self._node_to_paths(current)
            
            # 选择要解决的冲突
            conflict = self._select_conflict(current.conflicts)
            logger.debug(f"Selected conflict to resolve: {conflict}")
            
            # 为冲突创建约束
            constraint_a, constraint_b = create_constraints_from_conflict(conflict)
            
            # 创建两个子节点，每个应用一个约束
            for constraint in [constraint_a, constraint_b]:
                # 避免过多约束
                if len(current.constraints) >= self.config.max_constraints:
                    logger.warning("Reached maximum number of constraints")
                    continue
                
                # 创建新节点
                new_node = self._create_child_node(current, constraint)
                
                # 如果成功创建节点（找到了符合约束的路径）
                if new_node:
                    # 检查是否已经探索过
                    node_hash = self._compute_node_hash(new_node)
                    if node_hash not in self.closed_set:
                        # 检测新路径中的冲突
                        new_node.conflicts = detect_conflicts(
                            new_node.paths,
                            vehicle_radius=self.config.vehicle_radius,
                            time_horizon=self.config.time_horizon
                        )
                        
                        # 添加到开放集
                        heapq.heappush(self.open_set, new_node)
                        self.closed_set.add(node_hash)
                        
                        # 检查是否为解决方案
                        if not new_node.conflicts:
                            logger.info(f"Found conflict-free solution after {num_iterations} iterations")
                            return self._node_to_paths(new_node)
            
            num_iterations += 1
        
        logger.warning(f"CBS failed to find solution after {num_iterations} iterations")
        
        # 如果没有找到完全无冲突的解决方案，返回冲突最少的
        if self.open_set:
            best_node = min(self.open_set, key=lambda node: len(node.conflicts))
            logger.info(f"Returning best solution with {len(best_node.conflicts)} conflicts")
            return self._node_to_paths(best_node)
        else:
            # 如果开放集为空，返回原始路径
            logger.info("Open set is empty, returning original paths")
            return paths
    
    def _create_root_node(self, paths: Dict[str, Path], vehicles: Dict[str, Any] = None) -> CBSNode:
        """创建CBS搜索树的根节点"""
        # 转换路径为带时间步的路径
        paths_with_timesteps = {}
        for vehicle_id, path in paths.items():
            # 获取车辆速度
            speed = self.config.default_speed
            if vehicles and vehicle_id in vehicles:
                speed = getattr(vehicles[vehicle_id], 'max_speed', self.config.default_speed)
            
            path_ts = path_to_timesteps(path, speed=speed)
            path_ts.vehicle_id = vehicle_id
            paths_with_timesteps[vehicle_id] = path_ts
        
        # 检测初始冲突
        conflicts = detect_conflicts(
            paths_with_timesteps,
            vehicle_radius=self.config.vehicle_radius,
            time_horizon=self.config.time_horizon
        )
        
        # 创建根节点
        root = CBSNode(
            constraints=set(),
            paths=paths_with_timesteps,
            conflicts=conflicts
        )
        
        return root
    
    def _create_child_node(self, parent: CBSNode, constraint: Constraint) -> Optional[CBSNode]:
        """创建符合新约束的子节点"""
        # 复制父节点的约束和路径
        new_constraints = parent.constraints.copy()
        new_constraints.add(constraint)
        
        # 创建新节点
        new_node = CBSNode(
            constraints=new_constraints,
            paths=copy.deepcopy(parent.paths),
            parent=parent
        )
        
        # 获取受约束影响的车辆
        vehicle_id = constraint.vehicle_id
        
        # 获取该车辆的所有约束
        vehicle_constraints = new_node.get_constraints_for_vehicle(vehicle_id)
        
        # 重新规划路径
        original_path = parent.paths.get(vehicle_id)
        if not original_path:
            logger.error(f"Vehicle {vehicle_id} not found in parent paths")
            return None
        
        # 获取路径起点和终点
        start_point = original_path.points[0]
        end_point = original_path.points[-1]
        
        # 创建规划约束
        planning_constraints = self._convert_to_planning_constraints(
            vehicle_constraints, 
            start_point, 
            original_path.timesteps[0]
        )
        
        # 使用提供的路径规划器重新规划路径
        if hasattr(self.path_planner, 'plan_with_constraints'):
            plan_result = self.path_planner.plan_with_constraints(
                start=start_point,
                goal=end_point,
                constraints=planning_constraints
            )
        else:
            # 如果规划器不支持约束，使用普通规划
            logger.warning("Path planner does not support constraints, using regular planning")
            plan_result = self.path_planner.plan_path(
                start=start_point,
                goal=end_point
            )
        
        # 检查规划结果
        if plan_result.status != PlanningStatus.SUCCESS:
            logger.warning(f"Failed to find path for vehicle {vehicle_id} with constraints")
            return None
        
        # 将新路径转换为带时间步的路径
        new_path = plan_result.path
        path_ts = path_to_timesteps(
            new_path, 
            speed=self.config.default_speed, 
            start_time=original_path.timesteps[0]
        )
        path_ts.vehicle_id = vehicle_id
        
        # 验证新路径是否满足约束
        for c in vehicle_constraints:
            if path_ts.violates_constraint(c):
                logger.warning(f"New path violates constraint {c}")
                return None
        
        # 更新节点中的路径
        new_node.paths[vehicle_id] = path_ts
        new_node.update_cost()
        
        return new_node
    
    def _convert_to_planning_constraints(self, 
                                        constraints: List[Constraint], 
                                        start_point: Point2D, 
                                        start_time: int) -> PlanningConstraints:
        """将CBS约束转换为路径规划约束"""
        # 创建规划约束对象
        planning_constraints = PlanningConstraints()
        
        # 添加禁止区域
        forbidden_areas = []
        for constraint in constraints:
            if constraint.constraint_type == ConstraintType.VERTEX:
                # 顶点约束: 在特定时间不能位于特定位置
                forbidden_areas.append({
                    "point": constraint.location,
                    "time": constraint.time_step - start_time,
                    "radius": self.config.vehicle_radius
                })
            elif constraint.constraint_type == ConstraintType.EDGE:
                # 边约束: 在特定时间段不能通过特定边
                forbidden_areas.append({
                    "from_point": constraint.location,
                    "to_point": constraint.target_location,
                    "time": constraint.time_step - start_time,
                    "radius": self.config.vehicle_radius
                })
        
        planning_constraints.forbidden_areas = forbidden_areas
        
        # 设置最早出发时间
        earliest_departure = start_time
        for constraint in constraints:
            if constraint.constraint_type == ConstraintType.TEMPORAL:
                earliest_departure = max(earliest_departure, constraint.time_step)
        
        planning_constraints.earliest_departure_time = earliest_departure - start_time
        
        return planning_constraints
    
    def _select_conflict(self, conflicts: List[Conflict]) -> Conflict:
        """选择要解决的冲突"""
        if not conflicts:
            raise ValueError("No conflicts to select from")
        
        # 优先选择基数冲突(cardinal conflict)
        if self.config.use_cardinal_conflicts:
            for conflict in conflicts:
                if conflict.metadata.get("is_cardinal", False):
                    return conflict
        
        # 其次选择顶点冲突
        for conflict in conflicts:
            if conflict.conflict_type == ConflictType.VERTEX:
                return conflict
                
        # 默认返回第一个冲突
        return conflicts[0]
    
    def _node_to_paths(self, node: CBSNode) -> Dict[str, Path]:
        """将CBS节点转换为原始路径"""
        paths = {}
        for vehicle_id, path_ts in node.paths.items():
            paths[vehicle_id] = timesteps_to_path(path_ts)
        return paths
    
    def _compute_node_hash(self, node: CBSNode) -> int:
        """计算节点哈希值以进行重复检测"""
        # 简单哈希: 使用约束集合和车辆位置
        constraints_hash = hash(frozenset(node.constraints))
        
        # 对每个车辆的最终路径进行哈希
        paths_hash = 0
        for vehicle_id, path in sorted(node.paths.items()):
            # 只对关键点进行哈希以减少计算量
            path_points = [(p.x, p.y) for p in path.points[::10]]  # 每10个点采样一个
            if path.points:
                path_points.append((path.points[-1].x, path.points[-1].y))  # 确保包含终点
            paths_hash ^= hash((vehicle_id, tuple(path_points)))
        
        return hash((constraints_hash, paths_hash))