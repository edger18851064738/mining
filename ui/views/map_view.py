"""
Map view for the mining dispatch system.

Provides a visualization of the mining environment with vehicles and tasks.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import math
from typing import Dict, List, Optional, Any, Tuple

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QCheckBox, QSlider, QPushButton, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPathItem,
    QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsSimpleTextItem, QMenu,
    QAction, QToolBar, QToolButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtGui import (
    QPen, QBrush, QColor, QPainterPath, QPolygonF, QFont, 
    QMouseEvent, QWheelEvent, QPainter, QTransform
)

from utils.logger import get_logger
from utils.config import get_config
from utils.geo.coordinates import Point2D


class VehicleItem(QGraphicsPolygonItem):
    """Graphics item representing a vehicle on the map."""
    
    def __init__(self, vehicle_id: str, pos: QPointF, size: float = 10.0, parent=None):
        """
        Initialize a vehicle item.
        
        Args:
            vehicle_id: ID of the vehicle
            pos: Position on the map
            size: Size of the vehicle icon
            parent: Parent graphics item
        """
        # Create a triangle polygon for the vehicle
        points = [
            QPointF(0, -size),  # Top
            QPointF(-size/2, size/2),  # Bottom left
            QPointF(size/2, size/2)  # Bottom right
        ]
        polygon = QPolygonF(points)
        
        super().__init__(polygon, parent)
        
        self.vehicle_id = vehicle_id
        self.setPos(pos)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setToolTip(f"Vehicle {vehicle_id}")
        
        # Default pen and brush
        self.setPen(QPen(Qt.black, 1))
        self.setBrush(QBrush(QColor(48, 112, 176)))  # Blue
        
        # Add text label
        self.label = QGraphicsSimpleTextItem(vehicle_id, self)
        self.label.setPos(-self.label.boundingRect().width()/2, size/2 + 2)
        
        # Path for visualization
        self.path_item = None
    
    def set_state(self, state: str, loaded: bool = False):
        """
        Set vehicle state for visualization.
        
        Args:
            state: Current state of the vehicle
            loaded: Whether vehicle is loaded
        """
        # Change color based on state
        if state in ["LOADING", "UNLOADING"]:
            self.setBrush(QBrush(QColor(176, 48, 112)))  # Magenta
        elif state == "IDLE":
            self.setBrush(QBrush(QColor(48, 176, 112)))  # Green
        elif state == "MOVING" or state == "EN_ROUTE":
            self.setBrush(QBrush(QColor(48, 112, 176)))  # Blue
        elif state == "WAITING":
            self.setBrush(QBrush(QColor(176, 176, 48)))  # Yellow
        elif "ERROR" in state or "FAULT" in state:
            self.setBrush(QBrush(QColor(176, 48, 48)))  # Red
        else:
            self.setBrush(QBrush(QColor(112, 112, 112)))  # Gray
            
        # Filled when loaded, outline when empty
        if loaded:
            self.setPen(QPen(Qt.black, 2))
        else:
            self.setPen(QPen(Qt.black, 1))
            
        # Update tooltip with state
        self.setToolTip(f"Vehicle {self.vehicle_id}\nState: {state}\nLoaded: {loaded}")
    
    def set_path(self, path_points: List[QPointF]):
        """
        Set path for vehicle visualization.
        
        Args:
            path_points: List of points forming the path
        """
        if not path_points:
            # Remove path item if exists
            if self.path_item:
                scene = self.scene()
                if scene:
                    scene.removeItem(self.path_item)
                self.path_item = None
            return
            
        # Create path
        path = QPainterPath()
        path.moveTo(path_points[0])
        for point in path_points[1:]:
            path.lineTo(point)
            
        # Create or update path item
        if not self.path_item:
            self.path_item = QGraphicsPathItem(path)
            self.path_item.setPen(QPen(QColor(48, 112, 176, 128), 1, Qt.DashLine))
            scene = self.scene()
            if scene:
                scene.addItem(self.path_item)
        else:
            self.path_item.setPath(path)


class TaskItem(QGraphicsEllipseItem):
    """Graphics item representing a task on the map."""
    
    def __init__(self, task_id: str, pos: QPointF, size: float = 8.0, parent=None):
        """
        Initialize a task item.
        
        Args:
            task_id: ID of the task
            pos: Position on the map
            size: Size of the task icon
            parent: Parent graphics item
        """
        super().__init__(-size/2, -size/2, size, size, parent)
        
        self.task_id = task_id
        self.setPos(pos)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setToolTip(f"Task {task_id}")
        
        # Default pen and brush
        self.setPen(QPen(Qt.black, 1))
        self.setBrush(QBrush(QColor(255, 128, 0)))  # Orange
        
        # Add text label
        self.label = QGraphicsSimpleTextItem(task_id, self)
        self.label.setPos(-self.label.boundingRect().width()/2, size/2 + 2)
    
    def set_state(self, state: str, priority: int = 1):
        """
        Set task state for visualization.
        
        Args:
            state: Current state of the task
            priority: Task priority (1-10)
        """
        # Change color based on state
        if state == "PENDING":
            self.setBrush(QBrush(QColor(255, 128, 0)))  # Orange
        elif state == "ASSIGNED":
            self.setBrush(QBrush(QColor(255, 192, 0)))  # Yellow-orange
        elif state == "IN_PROGRESS":
            self.setBrush(QBrush(QColor(0, 192, 255)))  # Light blue
        elif state == "COMPLETED":
            self.setBrush(QBrush(QColor(0, 192, 0)))  # Green
        elif state == "FAILED":
            self.setBrush(QBrush(QColor(192, 0, 0)))  # Red
        else:
            self.setBrush(QBrush(QColor(128, 128, 128)))  # Gray
            
        # Change size based on priority (1-10 scale)
        size = 6.0 + (priority / 10.0) * 4.0
        self.setRect(-size/2, -size/2, size, size)
        
        # Update label position
        self.label.setPos(-self.label.boundingRect().width()/2, size/2 + 2)
        
        # Update tooltip with state and priority
        self.setToolTip(f"Task {self.task_id}\nState: {state}\nPriority: {priority}")


class MapView(QWidget):
    """
    Map view for visualizing the mining environment.
    
    Displays vehicles, tasks, and the environment with pan and zoom capabilities.
    """
    
    # Signal when an item is selected on the map
    selection_changed = pyqtSignal(str, str)  # (item_type, item_id)
    
    def __init__(self, parent=None):
        """
        Initialize the map view.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("MapView")
        
        # Initialize state
        self.simulator = None
        self.vehicles = {}  # vehicle_id -> VehicleItem
        self.tasks = {}  # task_id -> TaskItem
        self.highlighted_area = None  # Highlighted area on map
        
        # Initialize UI
        self._init_ui()
        
        # Initialize map update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_view)
        self.update_timer.start(500)  # Update every 500ms
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar
        toolbar = QToolBar("Map Controls")
        main_layout.addWidget(toolbar)
        
        # Add zoom controls
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        
        reset_view_action = QAction("Reset View", self)
        reset_view_action.triggered.connect(self.reset_view)
        toolbar.addAction(reset_view_action)
        
        toolbar.addSeparator()
        
        # Add display options
        show_vehicles_cb = QCheckBox("Vehicles", self)
        show_vehicles_cb.setChecked(True)
        show_vehicles_cb.toggled.connect(self.toggle_vehicles)
        toolbar.addWidget(show_vehicles_cb)
        
        show_tasks_cb = QCheckBox("Tasks", self)
        show_tasks_cb.setChecked(True)
        show_tasks_cb.toggled.connect(self.toggle_tasks)
        toolbar.addWidget(show_tasks_cb)
        
        show_paths_cb = QCheckBox("Paths", self)
        show_paths_cb.setChecked(True)
        show_paths_cb.toggled.connect(self.toggle_paths)
        toolbar.addWidget(show_paths_cb)
        
        show_grid_cb = QCheckBox("Grid", self)
        show_grid_cb.setChecked(True)
        show_grid_cb.toggled.connect(self.toggle_grid)
        toolbar.addWidget(show_grid_cb)
        
        # Add view options
        toolbar.addSeparator()
        
        view_label = QLabel("View Mode:", self)
        toolbar.addWidget(view_label)
        
        view_combo = QComboBox(self)
        view_combo.addItem("Normal")
        view_combo.addItem("Heat Map")
        view_combo.addItem("Terrain")
        view_combo.addItem("Congestion")
        view_combo.currentTextChanged.connect(self.change_view_mode)
        toolbar.addWidget(view_combo)
        
        # Add graphics view
        self.scene = QGraphicsScene(self)
        self.view = CustomGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        main_layout.addWidget(self.view)
        
        # Initialize grid
        self.grid_items = []
        self.show_grid = True
        
        # Initialize the scene
        self._setup_scene()
    
    def _setup_scene(self):
        """Set up the graphics scene with initial content."""
        # Clear scene
        self.scene.clear()
        
        # Get configuration
        config = get_config()
        grid_size = config.map.grid_size
        
        # Set scene bounds
        self.scene.setSceneRect(0, 0, grid_size, grid_size)
        
        # Add grid
        self._create_grid(grid_size, config.map.grid_nodes)
        
        # Add key locations
        self._add_key_locations()
    
    def _create_grid(self, size: int, nodes: int):
        """
        Create grid lines on the scene.
        
        Args:
            size: Size of the grid
            nodes: Number of grid nodes
        """
        # Remove previous grid
        for item in self.grid_items:
            self.scene.removeItem(item)
        self.grid_items = []
        
        # Skip if grid is disabled
        if not self.show_grid:
            return
            
        # Get configuration
        config = get_config()
        color = QColor(config.ui.map_grid_color)
        
        pen = QPen(color, 0.5)  # Thin lines for grid
        step = size / nodes
        
        # Create vertical lines
        for i in range(nodes + 1):
            x = i * step
            line = QGraphicsLineItem(x, 0, x, size)
            line.setPen(pen)
            self.scene.addItem(line)
            self.grid_items.append(line)
        
        # Create horizontal lines
        for i in range(nodes + 1):
            y = i * step
            line = QGraphicsLineItem(0, y, size, y)
            line.setPen(pen)
            self.scene.addItem(line)
            self.grid_items.append(line)
    
    def _add_key_locations(self):
        """Add key locations to the map."""
        # Get configuration
        config = get_config()
        
        for name, coords in config.map.key_locations.items():
            # Convert percentage to absolute coordinates
            x = coords[0] * config.map.grid_size
            y = coords[1] * config.map.grid_size
            
            # Create a marker for the location
            size = 12.0
            rect = QGraphicsRectItem(x - size/2, y - size/2, size, size)
            rect.setPen(QPen(Qt.black, 1))
            
            # Choose color based on location type
            if "parking" in name.lower():
                rect.setBrush(QBrush(QColor(128, 128, 255)))  # Blue
            elif "load" in name.lower():
                rect.setBrush(QBrush(QColor(255, 128, 128)))  # Red
            elif "unload" in name.lower():
                rect.setBrush(QBrush(QColor(128, 255, 128)))  # Green
            else:
                rect.setBrush(QBrush(QColor(192, 192, 192)))  # Gray
            
            rect.setToolTip(f"Location: {name}")
            self.scene.addItem(rect)
            
            # Add label
            label = QGraphicsSimpleTextItem(name)
            label.setPos(x - label.boundingRect().width()/2, y + size/2 + 2)
            self.scene.addItem(label)
    
    def set_simulator(self, simulator):
        """
        Set the simulator instance.
        
        Args:
            simulator: Simulator instance
        """
        self.simulator = simulator
        
        # Reset view when simulator changes
        self.reset_view()
        
        # Update view
        self.update_view()
    
    @pyqtSlot()
    def update_view(self):
        """Update the view with current simulation state."""
        if not self.simulator:
            return
            
        try:
            # Update environment visualization if needed
            self._update_environment()
            
            # Update vehicles
            self._update_vehicles()
            
            # Update tasks
            self._update_tasks()
            
        except Exception as e:
            self.logger.error(f"Error updating map view: {str(e)}", exc_info=True)
    
    def _update_environment(self):
        """Update environment visualization."""
        # Implementation depends on environment representation
        # This is a placeholder for future implementation
        pass
    
    def _update_vehicles(self):
        """Update vehicle representations on the map."""
        if not self.simulator or not hasattr(self.simulator, 'dispatcher'):
            return
            
        # Get vehicles from dispatcher
        dispatcher = self.simulator.dispatcher
        if not dispatcher:
            return
            
        try:
            vehicles_data = dispatcher.get_all_vehicles()
            
            # Remove vehicles that no longer exist
            for vehicle_id in list(self.vehicles.keys()):
                if vehicle_id not in vehicles_data:
                    self.scene.removeItem(self.vehicles[vehicle_id])
                    del self.vehicles[vehicle_id]
            
            # Update or add vehicles
            for vehicle_id, vehicle in vehicles_data.items():
                # Skip if no location
                if not hasattr(vehicle, 'current_location'):
                    continue
                    
                # Get position
                pos = vehicle.current_location
                if not isinstance(pos, Point2D):
                    if hasattr(pos, 'x') and hasattr(pos, 'y'):
                        pos = Point2D(pos.x, pos.y)
                    else:
                        continue
                
                # Create or update vehicle item
                if vehicle_id in self.vehicles:
                    # Update existing vehicle
                    vehicle_item = self.vehicles[vehicle_id]
                    vehicle_item.setPos(QPointF(pos.x, pos.y))
                    
                    # Set heading if available
                    if hasattr(vehicle, 'heading'):
                        vehicle_item.setRotation(-math.degrees(vehicle.heading))
                    
                    # Update state
                    state = "UNKNOWN"
                    if hasattr(vehicle, 'state'):
                        state = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                        
                    loaded = False
                    if hasattr(vehicle, 'is_loaded'):
                        loaded = vehicle.is_loaded
                        
                    vehicle_item.set_state(state, loaded)
                    
                    # Update path
                    if hasattr(vehicle, 'current_path') and vehicle.current_path:
                        path = vehicle.current_path
                        if isinstance(path, list):
                            path_points = [QPointF(p.x, p.y) for p in path]
                        else:
                            # Assuming Path object with points attribute
                            path_points = [QPointF(p.x, p.y) for p in path.points]
                        vehicle_item.set_path(path_points)
                    else:
                        vehicle_item.set_path([])
                        
                else:
                    # Create new vehicle item
                    vehicle_item = VehicleItem(vehicle_id, QPointF(pos.x, pos.y))
                    self.scene.addItem(vehicle_item)
                    self.vehicles[vehicle_id] = vehicle_item
                    
                    # Set initial state
                    state = "UNKNOWN"
                    if hasattr(vehicle, 'state'):
                        state = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                        
                    loaded = False
                    if hasattr(vehicle, 'is_loaded'):
                        loaded = vehicle.is_loaded
                        
                    vehicle_item.set_state(state, loaded)
        
        except Exception as e:
            self.logger.error(f"Error updating vehicles: {str(e)}", exc_info=True)
    
    def _update_tasks(self):
        """Update task representations on the map."""
        if not self.simulator or not hasattr(self.simulator, 'dispatcher'):
            return
            
        # Get tasks from dispatcher
        dispatcher = self.simulator.dispatcher
        if not dispatcher:
            return
            
        try:
            tasks_data = dispatcher.get_all_tasks()
            
            # Remove tasks that no longer exist
            for task_id in list(self.tasks.keys()):
                if task_id not in tasks_data:
                    self.scene.removeItem(self.tasks[task_id])
                    del self.tasks[task_id]
            
            # Update or add tasks
            for task_id, task in tasks_data.items():
                # Determine task position
                pos = None
                
                # Check various attributes that might contain position
                if hasattr(task, 'location'):
                    pos = task.location
                elif hasattr(task, 'start_point'):
                    pos = task.start_point
                elif hasattr(task, 'position'):
                    pos = task.position
                
                # Skip if no valid position
                if not pos:
                    continue
                    
                # Ensure pos is a Point2D
                if not isinstance(pos, Point2D):
                    if hasattr(pos, 'x') and hasattr(pos, 'y'):
                        pos = Point2D(pos.x, pos.y)
                    else:
                        continue
                
                # Determine task priority
                priority = 1
                if hasattr(task, 'priority'):
                    if hasattr(task.priority, 'value'):
                        priority = task.priority.value
                    else:
                        priority = int(task.priority)
                
                # Create or update task item
                if task_id in self.tasks:
                    # Update existing task
                    task_item = self.tasks[task_id]
                    task_item.setPos(QPointF(pos.x, pos.y))
                    
                    # Update state
                    state = "UNKNOWN"
                    if hasattr(task, 'status'):
                        state = task.status.name if hasattr(task.status, 'name') else str(task.status)
                        
                    task_item.set_state(state, priority)
                        
                else:
                    # Create new task item
                    task_item = TaskItem(task_id, QPointF(pos.x, pos.y))
                    self.scene.addItem(task_item)
                    self.tasks[task_id] = task_item
                    
                    # Set initial state
                    state = "UNKNOWN"
                    if hasattr(task, 'status'):
                        state = task.status.name if hasattr(task.status, 'name') else str(task.status)
                        
                    task_item.set_state(state, priority)
        
        except Exception as e:
            self.logger.error(f"Error updating tasks: {str(e)}", exc_info=True)
    
    def zoom_in(self):
        """Zoom in the view."""
        self.view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """Zoom out the view."""
        self.view.scale(1/1.2, 1/1.2)
    
    def reset_view(self):
        """Reset the view to default."""
        self.view.resetTransform()
        self.view.centerOn(0, 0)
        
        # Fit the scene in the view
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def toggle_vehicles(self, show: bool):
        """
        Toggle vehicles visibility.
        
        Args:
            show: Whether to show vehicles
        """
        for vehicle_item in self.vehicles.values():
            vehicle_item.setVisible(show)
    
    def toggle_tasks(self, show: bool):
        """
        Toggle tasks visibility.
        
        Args:
            show: Whether to show tasks
        """
        for task_item in self.tasks.values():
            task_item.setVisible(show)
    
    def toggle_paths(self, show: bool):
        """
        Toggle path visibility.
        
        Args:
            show: Whether to show paths
        """
        for vehicle_item in self.vehicles.values():
            if vehicle_item.path_item:
                vehicle_item.path_item.setVisible(show)
    
    def toggle_grid(self, show: bool):
        """
        Toggle grid visibility.
        
        Args:
            show: Whether to show grid
        """
        self.show_grid = show
        
        # Update grid
        config = get_config()
        self._create_grid(config.map.grid_size, config.map.grid_nodes)
    
    def change_view_mode(self, mode: str):
        """
        Change the view mode.
        
        Args:
            mode: View mode to switch to
        """
        # This is a placeholder for future implementation
        # Different view modes would show different data visualizations
        self.logger.info(f"View mode changed to: {mode}")
    
    def highlight_area(self, rect: QRectF):
        """
        Highlight an area on the map.
        
        Args:
            rect: Rectangle to highlight
        """
        # Remove previous highlight
        if self.highlighted_area:
            self.scene.removeItem(self.highlighted_area)
            self.highlighted_area = None
        
        # Create highlight rectangle
        self.highlighted_area = QGraphicsRectItem(rect)
        self.highlighted_area.setPen(QPen(QColor(255, 0, 0, 128), 2))
        self.highlighted_area.setBrush(QBrush(QColor(255, 0, 0, 64)))
        self.scene.addItem(self.highlighted_area)
    
    def clear_highlight(self):
        """Clear highlighted area."""
        if self.highlighted_area:
            self.scene.removeItem(self.highlighted_area)
            self.highlighted_area = None


class CustomGraphicsView(QGraphicsView):
    """Custom graphics view with additional features."""
    
    def __init__(self, scene, parent=None):
        """
        Initialize the custom graphics view.
        
        Args:
            scene: Graphics scene
            parent: Parent widget
        """
        super().__init__(scene, parent)
        self.setMouseTracking(True)
    
    def wheelEvent(self, event: QWheelEvent):
        """
        Handle wheel events for zooming.
        
        Args:
            event: Wheel event
        """
        # Zoom factor
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        
        # Save the scene pos
        old_pos = self.mapToScene(event.pos())
        
        # Zoom
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
            
        self.scale(zoom_factor, zoom_factor)
        
        # Get the new position
        new_pos = self.mapToScene(event.pos())
        
        # Move scene to old position
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
    
    def mousePressEvent(self, event: QMouseEvent):
        """
        Handle mouse press events.
        
        Args:
            event: Mouse event
        """
        # Middle button for panning
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            
            # Create a fake left button press event to start drag
            fake_event = QMouseEvent(
                QMouseEvent.MouseButtonPress,
                event.pos(),
                Qt.LeftButton,
                Qt.LeftButton,
                event.modifiers()
            )
            super().mousePressEvent(fake_event)
        else:
            super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Handle mouse release events.
        
        Args:
            event: Mouse event
        """
        # Reset drag mode when middle button is released
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.NoDrag)
            
            # Create a fake left button release event to end drag
            fake_event = QMouseEvent(
                QMouseEvent.MouseButtonRelease,
                event.pos(),
                Qt.LeftButton,
                Qt.LeftButton,
                event.modifiers()
            )
            super().mouseReleaseEvent(fake_event)
        else:
            super().mouseReleaseEvent(event)