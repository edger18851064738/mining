# 露天矿多车协同调度系统 - 代码文档

## 文档目的

本文档记录露天矿多车协同调度系统的所有类和函数，提供完整的API参考，便于调试和维护。

## 目录结构

- [1. 领域模型层 (Domain Model)](#1-领域模型层-domain-model)
- [2. 算法服务层 (Algorithm Services)](#2-算法服务层-algorithm-services)
- [3. 协调调度层 (Coordination)](#3-协调调度层-coordination)
- [4. 地图与环境层 (Map & Environment)](#4-地图与环境层-map--environment)
- [5. 界面层 (UI)](#5-界面层-ui)
- [6. 支持工具层 (Utilities)](#6-支持工具层-utilities)

## 类图和关系

[此处可以插入关键类的UML图]

---

## 1. 领域模型层 (Domain Model)

### 1.1 vehicles 模块

#### 1.1.1 Vehicle (base.py)

**描述**: 车辆基类，定义所有车辆共有的属性和方法

```python
class Vehicle(ABC):
    """
    车辆基类接口
    """
    
    def __init__(self, vehicle_id: str, position: Point2D, orientation: float):
        """
        初始化车辆
        
        参数:
            vehicle_id (str): 车辆唯一标识
            position (Point2D): 车辆当前位置
            orientation (float): 车辆朝向角度(弧度)
        """
        # 实现代码...
    
    @abstractmethod
    def move_to(self, target_position: Point2D) -> None:
        """
        移动车辆到目标位置
        
        参数:
            target_position (Point2D): 目标位置
        """
        pass
    
    # 其他方法...
```

#### 1.1.2 MiningVehicle (mining_vehicle.py)

**描述**: 矿用车辆实现

```python
class MiningVehicle(Vehicle):
    """
    矿用车辆类
    """
    
    def __init__(self, vehicle_id: str, position: Point2D, orientation: float, 
                 capacity: float, vehicle_type: VehicleType):
        """
        初始化矿用车辆
        
        参数:
            vehicle_id (str): 车辆唯一标识
            position (Point2D): 车辆当前位置
            orientation (float): 车辆朝向角度(弧度)
            capacity (float): 车辆载重容量
            vehicle_type (VehicleType): 车辆类型(如挖掘机、运输车等)
        """
        # 实现代码...
    
    def move_to(self, target_position: Point2D) -> None:
        """
        实现移动方法
        
        参数:
            target_position (Point2D): 目标位置
        """
        # 实现代码...
    
    # 其他方法...
```

### 1.2 tasks 模块

#### 1.2.1 Task (base.py)

**描述**: 任务基类，定义系统中任务的基本结构

```python
class Task(ABC):
    """
    任务基类接口
    """
    
    def __init__(self, task_id: str, priority: int, status: TaskStatus):
        """
        初始化任务
        
        参数:
            task_id (str): 任务唯一标识
            priority (int): 任务优先级
            status (TaskStatus): 任务状态
        """
        # 实现代码...
    
    @abstractmethod
    def is_completed(self) -> bool:
        """
        检查任务是否完成
        
        返回:
            bool: 任务是否已完成
        """
        pass
    
    # 其他方法...
```

[以此格式继续添加其他类和函数...]

## 2. 算法服务层 (Algorithm Services)

### 2.1 planning 模块

#### 2.1.1 PathPlanner (interfaces.py)

**描述**: 路径规划器接口

```python
class PathPlanner(ABC):
    """
    路径规划器接口
    """
    
    @abstractmethod
    def plan_path(self, start: Point2D, goal: Point2D, vehicle=None) -> Path:
        """
        规划从起点到终点的路径
        
        参数:
            start (Point2D): 起始位置
            goal (Point2D): 目标位置
            vehicle (Vehicle, 可选): 执行路径的车辆
            
        返回:
            Path: 规划的路径
        """
        pass
```

[以此格式继续添加其他类和函数...]

## 3. 协调调度层 (Coordination)

[按照相同的格式继续记录]

## 4. 地图与环境层 (Map & Environment)

[按照相同的格式继续记录]

## 5. 界面层 (UI)

[按照相同的格式继续记录]

## 6. 支持工具层 (Utilities)

[按照相同的格式继续记录]

---

## 附录

### A. 数据类型定义

```python
class Point2D:
    """二维点类型"""
    
    def __init__(self, x: float, y: float):
        """
        初始化二维点
        
        参数:
            x (float): X坐标
            y (float): Y坐标
        """
        self.x = x
        self.y = y
    
    def distance_to(self, other: 'Point2D') -> float:
        """
        计算到另一点的距离
        
        参数:
            other (Point2D): 另一个点
            
        返回:
            float: 两点之间的欧几里得距离
        """
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
```

[其他核心数据类型...]

### B. 枚举类型

```python
class VehicleType(Enum):
    """车辆类型枚举"""
    EXCAVATOR = "excavator"      # 挖掘机
    TRUCK = "truck"              # 运输车
    LOADER = "loader"            # 装载机
```

```python
class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"          # 待处理
    ASSIGNED = "assigned"        # 已分配
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
```

[其他枚举类型...]

### C. 错误和异常

```python
class PathPlanningError(Exception):
    """路径规划错误"""
    pass
```

[其他自定义异常...]