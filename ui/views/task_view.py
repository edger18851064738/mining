"""
Task view for the mining dispatch system.

Provides a list view of tasks with details and controls.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from typing import Dict, List, Optional, Any
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QLabel, 
    QTableWidget, QTableWidgetItem, QPushButton, QComboBox, 
    QHeaderView, QAbstractItemView, QMenu, QAction, QTreeWidget,
    QTreeWidgetItem, QSplitter, QFrame, QDialog, QFormLayout, 
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QDialogButtonBox,
    QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QModelIndex
from PyQt5.QtGui import QColor, QBrush, QIcon, QPixmap

from utils.logger import get_logger
from utils.config import get_config
from utils.geo.coordinates import Point2D


class TaskDetailsDialog(QDialog):
    """Dialog to display detailed task information."""
    
    def __init__(self, task_id: str, task_data: Any, parent=None):
        """
        Initialize task details dialog.
        
        Args:
            task_id: ID of the task
            task_data: Task data object
            parent: Parent widget
        """
        super().__init__(parent)
        self.task_id = task_id
        self.task_data = task_data
        
        self.setWindowTitle(f"Task Details - {task_id}")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Task ID and type
        header_layout = QHBoxLayout()
        id_label = QLabel(f"<h2>Task {self.task_id}</h2>")
        header_layout.addWidget(id_label)
        
        # Task type if available
        task_type = "Unknown"
        if hasattr(self.task_data, 'task_type'):
            task_type = self.task_data.task_type
        elif hasattr(self.task_data, '__class__'):
            task_type = self.task_data.__class__.__name__
            
        type_label = QLabel(f"Type: <b>{task_type}</b>")
        header_layout.addWidget(type_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Create form layout for details
        form_layout = QFormLayout()
        layout.addLayout(form_layout)
        
        # Add status
        status = "Unknown"
        if hasattr(self.task_data, 'status'):
            status = self.task_data.status.name if hasattr(self.task_data.status, 'name') else str(self.task_data.status)
        
        status_label = QLabel(status)
        
        # Set status color
        if status == "PENDING":
            status_label.setStyleSheet("color: orange;")
        elif status == "ASSIGNED":
            status_label.setStyleSheet("color: blue;")
        elif status == "IN_PROGRESS":
            status_label.setStyleSheet("color: purple;")
        elif status == "COMPLETED":
            status_label.setStyleSheet("color: green;")
        elif status == "FAILED":
            status_label.setStyleSheet("color: red;")
        
        form_layout.addRow("Status:", status_label)
        
        # Add priority if available
        if hasattr(self.task_data, 'priority'):
            priority = self.task_data.priority
            if hasattr(priority, 'name'):
                priority = priority.name
            elif hasattr(priority, 'value'):
                priority = priority.value
            form_layout.addRow("Priority:", QLabel(str(priority)))
        
        # Add location if available
        if hasattr(self.task_data, 'location'):
            loc = self.task_data.location
            loc_label = QLabel(f"({loc.x:.2f}, {loc.y:.2f})")
            form_layout.addRow("Location:", loc_label)
        elif hasattr(self.task_data, 'start_point'):
            loc = self.task_data.start_point
            loc_label = QLabel(f"({loc.x:.2f}, {loc.y:.2f})")
            form_layout.addRow("Start Point:", loc_label)
            
            if hasattr(self.task_data, 'end_point'):
                loc = self.task_data.end_point
                loc_label = QLabel(f"({loc.x:.2f}, {loc.y:.2f})")
                form_layout.addRow("End Point:", loc_label)
        
        # Add deadlines if available
        if hasattr(self.task_data, 'deadline') and self.task_data.deadline:
            deadline = self.task_data.deadline
            if isinstance(deadline, datetime):
                deadline_str = deadline.strftime("%Y-%m-%d %H:%M:%S")
                form_layout.addRow("Deadline:", QLabel(deadline_str))
        
        # Add creation time if available
        if hasattr(self.task_data, 'creation_time') and self.task_data.creation_time:
            creation_time = self.task_data.creation_time
            if isinstance(creation_time, datetime):
                time_str = creation_time.strftime("%Y-%m-%d %H:%M:%S")
                form_layout.addRow("Created:", QLabel(time_str))
        
        # Add material type and amount for transport tasks
        if hasattr(self.task_data, 'material_type'):
            form_layout.addRow("Material:", QLabel(str(self.task_data.material_type)))
            
            if hasattr(self.task_data, 'amount'):
                amount_label = QLabel(f"{self.task_data.amount:.2f}")
                form_layout.addRow("Amount:", amount_label)
        
        # Add progress if available
        if hasattr(self.task_data, 'progress'):
            progress_label = QLabel(f"{self.task_data.progress:.1f}%")
            form_layout.addRow("Progress:", progress_label)
        
        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Add assigned vehicle section
        layout.addWidget(QLabel("<h3>Assignment</h3>"))
        
        # Get assigned vehicle if any
        assigned_vehicle = None
        
        # Check if task has assignee info
        if hasattr(self.task_data, 'assignee_id') and self.task_data.assignee_id:
            assigned_vehicle = self.task_data.assignee_id
        
        # Check from dispatcher assignment if a dispatcher is available
        dispatcher = self.parent().simulator.dispatcher if hasattr(self.parent(), 'simulator') and self.parent().simulator and hasattr(self.parent().simulator, 'dispatcher') else None
        
        if not assigned_vehicle and dispatcher:
            try:
                assignments = dispatcher.get_assignments()
                for vehicle_id, task_ids in assignments.items():
                    if self.task_id in task_ids:
                        assigned_vehicle = vehicle_id
                        break
            except:
                pass
        
        if assigned_vehicle:
            vehicle_label = QLabel(f"<b>{assigned_vehicle}</b>")
            layout.addWidget(vehicle_label)
            
            # Add vehicle details if available
            vehicle_data = None
            if dispatcher:
                try:
                    vehicle_data = dispatcher.get_vehicle(assigned_vehicle)
                except:
                    pass
                
            if vehicle_data:
                details_layout = QFormLayout()
                
                # Vehicle state
                if hasattr(vehicle_data, 'state'):
                    state = vehicle_data.state.name if hasattr(vehicle_data.state, 'name') else str(vehicle_data.state)
                    details_layout.addRow("State:", QLabel(state))
                
                # Vehicle position
                if hasattr(vehicle_data, 'current_location'):
                    pos = vehicle_data.current_location
                    pos_label = QLabel(f"({pos.x:.1f}, {pos.y:.1f})")
                    details_layout.addRow("Position:", pos_label)
                
                layout.addLayout(details_layout)
        else:
            layout.addWidget(QLabel("Not assigned to any vehicle"))
        
        # Add buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        # Cancel task button if not completed
        if status not in ["COMPLETED", "FAILED", "CANCELED"]:
            cancel_button = QPushButton("Cancel Task")
            cancel_button.clicked.connect(self.cancel_task)
            button_layout.addWidget(cancel_button)
            
            button_layout.addStretch()
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        button_layout.addWidget(close_button)
    
    def cancel_task(self):
        """Cancel the task."""
        # Check if simulator and dispatcher are available
        dispatcher = self.parent().simulator.dispatcher if hasattr(self.parent(), 'simulator') and self.parent().simulator and hasattr(self.parent().simulator, 'dispatcher') else None
        
        if not dispatcher:
            QMessageBox.warning(self, "Warning", "Dispatcher not available")
            return
        
        # Confirm cancellation
        confirm = QMessageBox.question(
            self,
            "Confirm Cancellation",
            f"Are you sure you want to cancel task {self.task_id}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Try to get task and cancel it
                task = dispatcher.get_task(self.task_id)
                if hasattr(task, 'cancel') and callable(task.cancel):
                    task.cancel()
                    
                    # Update dialog
                    QMessageBox.information(self, "Success", f"Task {self.task_id} has been canceled")
                    self.accept()  # Close dialog
                else:
                    QMessageBox.warning(self, "Warning", "Task cannot be canceled")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to cancel task: {str(e)}")


class CreateTaskDialog(QDialog):
    """Dialog to create a new task."""
    
    def __init__(self, simulator, parent=None):
        """
        Initialize create task dialog.
        
        Args:
            simulator: Simulator instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.simulator = simulator
        
        self.setWindowTitle("Create New Task")
        self.setMinimumWidth(400)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Task type selection
        layout.addWidget(QLabel("<h3>Task Type</h3>"))
        
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItem("Transport Task")
        self.task_type_combo.addItem("Loading Task")
        self.task_type_combo.addItem("Unloading Task")
        layout.addWidget(self.task_type_combo)
        
        # Task details
        layout.addWidget(QLabel("<h3>Task Details</h3>"))
        
        form_layout = QFormLayout()
        layout.addLayout(form_layout)
        
        # Location selection
        form_layout.addRow("Location:", QLabel("Select on map or enter coordinates:"))
        
        loc_layout = QHBoxLayout()
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(0, 10000)
        self.x_spin.setValue(100)
        self.x_spin.setSuffix(" (x)")
        
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(0, 10000)
        self.y_spin.setValue(100)
        self.y_spin.setSuffix(" (y)")
        
        loc_layout.addWidget(self.x_spin)
        loc_layout.addWidget(self.y_spin)
        form_layout.addRow("", loc_layout)
        
        # For transport tasks, add destination
        dest_layout = QHBoxLayout()
        self.dest_x_spin = QDoubleSpinBox()
        self.dest_x_spin.setRange(0, 10000)
        self.dest_x_spin.setValue(200)
        self.dest_x_spin.setSuffix(" (x)")
        
        self.dest_y_spin = QDoubleSpinBox()
        self.dest_y_spin.setRange(0, 10000)
        self.dest_y_spin.setValue(200)
        self.dest_y_spin.setSuffix(" (y)")
        
        dest_layout.addWidget(self.dest_x_spin)
        dest_layout.addWidget(self.dest_y_spin)
        form_layout.addRow("Destination:", dest_layout)
        
        # Material type
        self.material_combo = QComboBox()
        self.material_combo.addItem("ore")
        self.material_combo.addItem("waste")
        self.material_combo.addItem("gravel")
        self.material_combo.addItem("coal")
        form_layout.addRow("Material:", self.material_combo)
        
        # Amount
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(1000, 100000)
        self.amount_spin.setValue(50000)
        self.amount_spin.setSingleStep(1000)
        form_layout.addRow("Amount:", self.amount_spin)
        
        # Priority
        self.priority_combo = QComboBox()
        self.priority_combo.addItem("LOW")
        self.priority_combo.addItem("NORMAL")
        self.priority_combo.addItem("HIGH")
        self.priority_combo.addItem("URGENT")
        self.priority_combo.addItem("CRITICAL")
        self.priority_combo.setCurrentText("NORMAL")
        form_layout.addRow("Priority:", self.priority_combo)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_task_data(self) -> Dict[str, Any]:
        """
        Get task data from dialog inputs.
        
        Returns:
            Dict[str, Any]: Task data
        """
        task_data = {
            "task_type": self.task_type_combo.currentText(),
            "location": Point2D(self.x_spin.value(), self.y_spin.value()),
            "destination": Point2D(self.dest_x_spin.value(), self.dest_y_spin.value()),
            "material_type": self.material_combo.currentText(),
            "amount": self.amount_spin.value(),
            "priority": self.priority_combo.currentText()
        }
        
        return task_data


class TaskView(QWidget):
    """
    Task view showing information about all tasks in the system.
    
    Displays task state, location, and assigned vehicles.
    """
    
    # Signal when a task is selected
    task_selected = pyqtSignal(str)  # task_id
    
    def __init__(self, parent=None):
        """
        Initialize the task view.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("TaskView")
        
        # Initialize state
        self.simulator = None
        self.tasks = {}  # task_id -> task_data
        
        # Initialize UI
        self._init_ui()
        
        # Initialize update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_view)
        self.update_timer.start(1000)  # Update every second
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar
        toolbar = QToolBar("Tasks Toolbar")
        layout.addWidget(toolbar)
        
        # Add filter controls
        filter_label = QLabel("Filter:")
        toolbar.addWidget(filter_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Tasks")
        self.filter_combo.addItem("Pending Tasks")
        self.filter_combo.addItem("Assigned Tasks")
        self.filter_combo.addItem("In Progress")
        self.filter_combo.addItem("Completed Tasks")
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        toolbar.addWidget(self.filter_combo)
        
        toolbar.addSeparator()
        
        # Add new task button
        new_task_action = QAction("New Task", self)
        new_task_action.triggered.connect(self.create_new_task)
        toolbar.addAction(new_task_action)
        
        # Add refresh button
        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.update_view)
        toolbar.addAction(refresh_action)
        
        # Create task table
        self.task_table = QTableWidget(0, 5)
        self.task_table.setHorizontalHeaderLabels([
            "ID", "Type", "Status", "Location", "Vehicle"
        ])
        self.task_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.task_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.task_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.task_table.itemDoubleClicked.connect(self.show_task_details)
        layout.addWidget(self.task_table)
    
    def set_simulator(self, simulator):
        """
        Set the simulator instance.
        
        Args:
            simulator: Simulator instance
        """
        self.simulator = simulator
        
        # Update view
        self.update_view()
    
    @pyqtSlot()
    def update_view(self):
        """Update the view with current task information."""
        if not self.simulator or not hasattr(self.simulator, 'dispatcher'):
            return
            
        # Get tasks from dispatcher
        dispatcher = self.simulator.dispatcher
        if not dispatcher:
            return
            
        try:
            # Get all tasks
            tasks_data = dispatcher.get_all_tasks()
            
            # Store task data
            self.tasks = tasks_data
            
            # Apply current filter
            self.apply_filter(self.filter_combo.currentText())
            
        except Exception as e:
            self.logger.error(f"Error updating task view: {str(e)}", exc_info=True)
    
    def apply_filter(self, filter_text: str):
        """
        Apply filter to task list.
        
        Args:
            filter_text: Filter to apply
        """
        if not self.tasks:
            return
            
        # Clear table
        self.task_table.setRowCount(0)
        
        # Get assignments if available
        assignments = {}
        vehicles_by_task = {}
        
        if self.simulator and self.simulator.dispatcher:
            try:
                assignments = self.simulator.dispatcher.get_assignments()
                
                # Create reverse mapping: task -> vehicle
                for vehicle_id, task_ids in assignments.items():
                    for task_id in task_ids:
                        vehicles_by_task[task_id] = vehicle_id
            except:
                pass
        
        # Row counter
        row = 0
        
        # Add tasks based on filter
        for task_id, task in self.tasks.items():
            # Apply filter
            if filter_text != "All Tasks":
                # Get task status
                status = "UNKNOWN"
                if hasattr(task, 'status'):
                    status = task.status.name if hasattr(task.status, 'name') else str(task.status)
                
                # Check if task matches filter
                if filter_text == "Pending Tasks":
                    if status != "PENDING":
                        continue
                elif filter_text == "Assigned Tasks":
                    if status != "ASSIGNED":
                        continue
                elif filter_text == "In Progress":
                    if status != "IN_PROGRESS":
                        continue
                elif filter_text == "Completed Tasks":
                    if status != "COMPLETED":
                        continue
            
            # Add row for task
            self.task_table.insertRow(row)
            
            # Task ID
            id_item = QTableWidgetItem(task_id)
            self.task_table.setItem(row, 0, id_item)
            
            # Task type
            task_type = "Unknown"
            if hasattr(task, 'task_type'):
                task_type = task.task_type
            elif hasattr(task, '__class__'):
                task_type = task.__class__.__name__
                
            type_item = QTableWidgetItem(task_type)
            self.task_table.setItem(row, 1, type_item)
            
            # Task status
            status = "UNKNOWN"
            if hasattr(task, 'status'):
                status = task.status.name if hasattr(task.status, 'name') else str(task.status)
            
            status_item = QTableWidgetItem(status)
            
            # Color based on status
            if status == "PENDING":
                status_item.setBackground(QBrush(QColor(255, 192, 128)))  # Light orange
            elif status == "ASSIGNED":
                status_item.setBackground(QBrush(QColor(192, 192, 255)))  # Light blue
            elif status == "IN_PROGRESS":
                status_item.setBackground(QBrush(QColor(192, 128, 255)))  # Light purple
            elif status == "COMPLETED":
                status_item.setBackground(QBrush(QColor(128, 255, 128)))  # Light green
            elif status == "FAILED":
                status_item.setBackground(QBrush(QColor(255, 128, 128)))  # Light red
            
            self.task_table.setItem(row, 2, status_item)
            
            # Task location
            location = "Unknown"
            if hasattr(task, 'location'):
                loc = task.location
                location = f"({loc.x:.1f}, {loc.y:.1f})"
            elif hasattr(task, 'start_point'):
                loc = task.start_point
                location = f"({loc.x:.1f}, {loc.y:.1f})"
            
            location_item = QTableWidgetItem(location)
            self.task_table.setItem(row, 3, location_item)
            
            # Assigned vehicle
            vehicle_id = vehicles_by_task.get(task_id, "")
            if not vehicle_id and hasattr(task, 'assignee_id'):
                vehicle_id = task.assignee_id or ""
                
            vehicle_item = QTableWidgetItem(vehicle_id)
            self.task_table.setItem(row, 4, vehicle_item)
            
            # Move to next row
            row += 1
    
    def show_task_details(self, item: QTableWidgetItem):
        """
        Show details for a selected task.
        
        Args:
            item: Selected table item
        """
        # Get selected row
        row = item.row()
        
        # Get task ID from first column
        task_id = self.task_table.item(row, 0).text()
        
        # Get task data
        if task_id in self.tasks:
            # Show details dialog
            dialog = TaskDetailsDialog(task_id, self.tasks[task_id], self)
            dialog.exec_()
            
            # Emit signal
            self.task_selected.emit(task_id)
            
            # Update view after dialog closes
            self.update_view()
    
    def create_new_task(self):
        """Create a new task."""
        if not self.simulator or not hasattr(self.simulator, 'dispatcher'):
            QMessageBox.warning(self, "Warning", "No active simulation")
            return
            
        # Show create task dialog
        dialog = CreateTaskDialog(self.simulator, self)
        if dialog.exec_() == QDialog.Accepted:
            # Get task data
            task_data = dialog.get_task_data()
            
            try:
                # Create task based on type
                task_type = task_data["task_type"]
                task = None
                
                if self.simulator.dispatcher:
                    # TODO: Create actual task object
                    self.logger.info(f"Creating new task: {task_type}")
                    self.logger.info(f"Task data: {task_data}")
                    
                    # This is a placeholder - replace with actual task creation
                    QMessageBox.information(
                        self, 
                        "Task Creation", 
                        f"Task creation not fully implemented yet.\n\nTask type: {task_type}"
                    )
                    
                    # Update view
                    self.update_view()
                else:
                    QMessageBox.warning(self, "Warning", "Dispatcher not available")
            except Exception as e:
                self.logger.error(f"Error creating task: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to create task: {str(e)}")
    
    def contextMenuEvent(self, event):
        """
        Show context menu for task operations.
        
        Args:
            event: Context menu event
        """
        # Get item at position
        item = self.task_table.itemAt(self.task_table.viewport().mapFrom(self, event.pos()))
        if not item:
            return
            
        # Get task ID
        row = item.row()
        task_id = self.task_table.item(row, 0).text()
        
        # Create context menu
        menu = QMenu(self)
        
        # Add actions
        details_action = QAction("View Details", self)
        details_action.triggered.connect(lambda: self.show_task_details(item))
        menu.addAction(details_action)
        
        # Check if dispatcher is available for other actions
        if self.simulator and self.simulator.dispatcher:
            menu.addSeparator()
            
            # Get task status
            status = self.task_table.item(row, 2).text()
            
            # Add assignment action if task is pending
            if status == "PENDING":
                assign_action = QAction("Assign to Vehicle...", self)
                assign_action.triggered.connect(lambda: self.assign_to_vehicle(task_id))
                menu.addAction(assign_action)
            
            # Add cancel action if task is not completed
            if status not in ["COMPLETED", "FAILED", "CANCELED"]:
                cancel_action = QAction("Cancel Task", self)
                cancel_action.triggered.connect(lambda: self.cancel_task(task_id))
                menu.addAction(cancel_action)
            
            # Add priority change action
            menu.addSeparator()
            
            priority_menu = menu.addMenu("Change Priority")
            
            for priority in ["LOW", "NORMAL", "HIGH", "URGENT", "CRITICAL"]:
                priority_action = QAction(priority, self)
                priority_action.triggered.connect(lambda p=priority: self.change_priority(task_id, p))
                priority_menu.addAction(priority_action)
        
        # Show menu
        menu.exec_(event.globalPos())
    
    def assign_to_vehicle(self, task_id: str):
        """
        Assign a task to a vehicle.
        
        Args:
            task_id: ID of the task
        """
        if not self.simulator or not self.simulator.dispatcher:
            return
            
        # TODO: Implement vehicle selection dialog
        self.logger.info(f"Assign task {task_id} to vehicle")
        
        # Placeholder: Just log the action
        QMessageBox.information(
            self, 
            "Task Assignment", 
            f"Manual task assignment not implemented yet for task {task_id}"
        )
    
    def cancel_task(self, task_id: str):
        """
        Cancel a task.
        
        Args:
            task_id: ID of the task
        """
        if not self.simulator or not self.simulator.dispatcher:
            return
            
        # Confirm cancellation
        confirm = QMessageBox.question(
            self,
            "Confirm Cancellation",
            f"Are you sure you want to cancel task {task_id}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Try to get task and cancel it
                task = self.simulator.dispatcher.get_task(task_id)
                if hasattr(task, 'cancel') and callable(task.cancel):
                    task.cancel()
                    
                    # Update view
                    self.update_view()
                    
                    QMessageBox.information(self, "Success", f"Task {task_id} has been canceled")
                else:
                    QMessageBox.warning(self, "Warning", "Task cannot be canceled")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to cancel task: {str(e)}")
    
    def change_priority(self, task_id: str, priority: str):
        """
        Change task priority.
        
        Args:
            task_id: ID of the task
            priority: New priority
        """
        if not self.simulator or not self.simulator.dispatcher:
            return
            
        self.logger.info(f"Change priority of task {task_id} to {priority}")
        
        try:
            # Try to get task and change priority
            task = self.simulator.dispatcher.get_task(task_id)
            
            # Check if task has a priority property that can be set
            if hasattr(task, 'priority'):
                # Determine how to set priority based on the type
                if hasattr(task.priority, '__class__') and task.priority.__class__.__name__ == 'TaskPriority':
                    # Enum priority
                    task.priority = getattr(task.priority.__class__, priority)
                else:
                    # String or other priority
                    task.priority = priority
                
                # Update view
                self.update_view()
                
                QMessageBox.information(self, "Success", f"Task {task_id} priority set to {priority}")
            else:
                QMessageBox.warning(self, "Warning", "Task priority cannot be changed")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to change task priority: {str(e)}")