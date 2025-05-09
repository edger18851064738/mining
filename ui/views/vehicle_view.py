"""
Vehicle view for the mining dispatch system.

Provides a list view of vehicles with details and controls.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from typing import Dict, List, Optional, Any

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


class VehicleDetailsDialog(QDialog):
    """Dialog to display detailed vehicle information."""
    
    def __init__(self, vehicle_id: str, vehicle_data: Any, parent=None):
        """
        Initialize vehicle details dialog.
        
        Args:
            vehicle_id: ID of the vehicle
            vehicle_data: Vehicle data object
            parent: Parent widget
        """
        super().__init__(parent)
        self.vehicle_id = vehicle_id
        self.vehicle_data = vehicle_data
        
        self.setWindowTitle(f"Vehicle Details - {vehicle_id}")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Vehicle ID and type
        header_layout = QHBoxLayout()
        id_label = QLabel(f"<h2>Vehicle {self.vehicle_id}</h2>")
        header_layout.addWidget(id_label)
        
        # Vehicle type if available
        if hasattr(self.vehicle_data, 'vehicle_type'):
            type_label = QLabel(f"Type: <b>{self.vehicle_data.vehicle_type}</b>")
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
        
        # Add state
        state = "Unknown"
        if hasattr(self.vehicle_data, 'state'):
            state = self.vehicle_data.state.name if hasattr(self.vehicle_data.state, 'name') else str(self.vehicle_data.state)
        form_layout.addRow("State:", QLabel(state))
        
        # Add position if available
        if hasattr(self.vehicle_data, 'current_location'):
            pos = self.vehicle_data.current_location
            pos_label = QLabel(f"({pos.x:.2f}, {pos.y:.2f})")
            form_layout.addRow("Position:", pos_label)
        
        # Add heading if available
        if hasattr(self.vehicle_data, 'heading'):
            heading_deg = self.vehicle_data.heading * 180 / 3.14159
            heading_label = QLabel(f"{heading_deg:.1f}°")
            form_layout.addRow("Heading:", heading_label)
        
        # Add speed if available
        if hasattr(self.vehicle_data, 'current_speed'):
            speed_label = QLabel(f"{self.vehicle_data.current_speed:.2f} m/s")
            form_layout.addRow("Speed:", speed_label)
        
        # Add capacity if available
        if hasattr(self.vehicle_data, 'max_capacity'):
            capacity_label = QLabel(f"{self.vehicle_data.max_capacity:.2f}")
            form_layout.addRow("Max Capacity:", capacity_label)
        
        # Add current load if available
        if hasattr(self.vehicle_data, 'current_load'):
            load_label = QLabel(f"{self.vehicle_data.current_load:.2f}")
            form_layout.addRow("Current Load:", load_label)
            
            # Add load ratio if available
            if hasattr(self.vehicle_data, 'load_ratio'):
                ratio_label = QLabel(f"{self.vehicle_data.load_ratio * 100:.1f}%")
                form_layout.addRow("Load Ratio:", ratio_label)
        
        # Add assigned tasks section
        layout.addWidget(QLabel("<h3>Assigned Tasks</h3>"))
        
        assigned_tasks = []
        
        # Check if vehicle has task info
        if hasattr(self.vehicle_data, 'assigned_task') and self.vehicle_data.assigned_task:
            assigned_tasks.append(self.vehicle_data.assigned_task)
        
        # Check from dispatcher assignment if a dispatcher is available
        dispatcher = self.parent().simulator.dispatcher if hasattr(self.parent(), 'simulator') and self.parent().simulator and hasattr(self.parent().simulator, 'dispatcher') else None
        
        if dispatcher:
            try:
                assignments = dispatcher.get_assignments()
                if self.vehicle_id in assignments:
                    for task_id in assignments[self.vehicle_id]:
                        if task_id not in assigned_tasks:
                            assigned_tasks.append(task_id)
            except:
                pass
        
        # Create task list
        if assigned_tasks:
            task_list = QTreeWidget()
            task_list.setHeaderLabels(["Task ID", "Status"])
            task_list.setRootIsDecorated(False)
            
            for task_id in assigned_tasks:
                item = QTreeWidgetItem([task_id, ""])
                task_list.addTopLevelItem(item)
                
                # Add task details if available
                if dispatcher:
                    try:
                        task = dispatcher.get_task(task_id)
                        if task:
                            if hasattr(task, 'status'):
                                status = task.status.name if hasattr(task.status, 'name') else str(task.status)
                                item.setText(1, status)
                    except:
                        pass
            
            task_list.setMaximumHeight(100)
            layout.addWidget(task_list)
        else:
            layout.addWidget(QLabel("No tasks assigned"))
        
        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Add buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        # Close button
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        button_layout.addWidget(buttons)


class VehicleView(QWidget):
    """
    Vehicle view showing information about all vehicles in the system.
    
    Displays vehicle state, position, and assigned tasks.
    """
    
    # Signal when a vehicle is selected
    vehicle_selected = pyqtSignal(str)  # vehicle_id
    
    def __init__(self, parent=None):
        """
        Initialize the vehicle view.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("VehicleView")
        
        # Initialize state
        self.simulator = None
        self.vehicles = {}  # vehicle_id -> vehicle_data
        
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
        toolbar = QToolBar("Vehicles Toolbar")
        layout.addWidget(toolbar)
        
        # Add filter controls
        filter_label = QLabel("Filter:")
        toolbar.addWidget(filter_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Vehicles")
        self.filter_combo.addItem("Active Vehicles")
        self.filter_combo.addItem("Idle Vehicles")
        self.filter_combo.addItem("Loading/Unloading")
        self.filter_combo.addItem("In Transit")
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        toolbar.addWidget(self.filter_combo)
        
        toolbar.addSeparator()
        
        # Add refresh button
        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.update_view)
        toolbar.addAction(refresh_action)
        
        # Create vehicle table
        self.vehicle_table = QTableWidget(0, 5)
        self.vehicle_table.setHorizontalHeaderLabels([
            "ID", "Type", "State", "Location", "Tasks"
        ])
        self.vehicle_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.vehicle_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.vehicle_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.vehicle_table.verticalHeader().setVisible(False)
        self.vehicle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.vehicle_table.itemDoubleClicked.connect(self.show_vehicle_details)
        layout.addWidget(self.vehicle_table)
    
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
        """Update the view with current vehicle information."""
        if not self.simulator or not hasattr(self.simulator, 'dispatcher'):
            return
            
        # Get vehicles from dispatcher
        dispatcher = self.simulator.dispatcher
        if not dispatcher:
            return
            
        try:
            # Get all vehicles
            vehicles_data = dispatcher.get_all_vehicles()
            
            # Store vehicle data
            self.vehicles = vehicles_data
            
            # Apply current filter
            self.apply_filter(self.filter_combo.currentText())
            
        except Exception as e:
            self.logger.error(f"Error updating vehicle view: {str(e)}", exc_info=True)
    
    def apply_filter(self, filter_text: str):
        """
        Apply filter to vehicle list.
        
        Args:
            filter_text: Filter to apply
        """
        if not self.vehicles:
            return
            
        # Clear table
        self.vehicle_table.setRowCount(0)
        
        # Get assignments if available
        assignments = {}
        if self.simulator and self.simulator.dispatcher:
            try:
                assignments = self.simulator.dispatcher.get_assignments()
            except:
                pass
        
        # Row counter
        row = 0
        
        # Add vehicles based on filter
        for vehicle_id, vehicle in self.vehicles.items():
            # Apply filter
            if filter_text != "All Vehicles":
                # Get vehicle state
                state = "UNKNOWN"
                if hasattr(vehicle, 'state'):
                    state = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                
                # Check if vehicle matches filter
                if filter_text == "Active Vehicles":
                    if state in ["IDLE", "WAITING", "OUT_OF_SERVICE", "MAINTENANCE"]:
                        continue
                elif filter_text == "Idle Vehicles":
                    if state != "IDLE":
                        continue
                elif filter_text == "Loading/Unloading":
                    if state not in ["LOADING", "UNLOADING"]:
                        continue
                elif filter_text == "In Transit":
                    if state not in ["MOVING", "EN_ROUTE"]:
                        continue
            
            # Add row for vehicle
            self.vehicle_table.insertRow(row)
            
            # Vehicle ID
            id_item = QTableWidgetItem(vehicle_id)
            self.vehicle_table.setItem(row, 0, id_item)
            
            # Vehicle type
            type_item = QTableWidgetItem(
                vehicle.vehicle_type if hasattr(vehicle, 'vehicle_type') else "Unknown"
            )
            self.vehicle_table.setItem(row, 1, type_item)
            
            # Vehicle state
            state = "UNKNOWN"
            if hasattr(vehicle, 'state'):
                state = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
            
            state_item = QTableWidgetItem(state)
            
            # Color based on state
            if state in ["LOADING", "UNLOADING"]:
                state_item.setBackground(QBrush(QColor(255, 192, 192)))  # Light red
            elif state == "IDLE":
                state_item.setBackground(QBrush(QColor(192, 255, 192)))  # Light green
            elif state in ["MOVING", "EN_ROUTE"]:
                state_item.setBackground(QBrush(QColor(192, 192, 255)))  # Light blue
            elif state == "WAITING":
                state_item.setBackground(QBrush(QColor(255, 255, 192)))  # Light yellow
            elif "ERROR" in state or "FAULT" in state:
                state_item.setBackground(QBrush(QColor(255, 128, 128)))  # Darker red
            
            self.vehicle_table.setItem(row, 2, state_item)
            
            # Vehicle location
            location = "Unknown"
            if hasattr(vehicle, 'current_location'):
                pos = vehicle.current_location
                location = f"({pos.x:.1f}, {pos.y:.1f})"
            
            location_item = QTableWidgetItem(location)
            self.vehicle_table.setItem(row, 3, location_item)
            
            # Vehicle tasks
            tasks = []
            if vehicle_id in assignments:
                tasks = assignments[vehicle_id]
            
            task_count = len(tasks)
            task_text = f"{task_count} task{'s' if task_count != 1 else ''}"
            if task_count > 0:
                task_text += f": {', '.join(tasks[:2])}"
                if task_count > 2:
                    task_text += f" + {task_count - 2} more"
            
            task_item = QTableWidgetItem(task_text)
            self.vehicle_table.setItem(row, 4, task_item)
            
            # Move to next row
            row += 1
    
    def show_vehicle_details(self, item: QTableWidgetItem):
        """
        Show details for a selected vehicle.
        
        Args:
            item: Selected table item
        """
        # Get selected row
        row = item.row()
        
        # Get vehicle ID from first column
        vehicle_id = self.vehicle_table.item(row, 0).text()
        
        # Get vehicle data
        if vehicle_id in self.vehicles:
            # Show details dialog
            dialog = VehicleDetailsDialog(vehicle_id, self.vehicles[vehicle_id], self)
            dialog.exec_()
            
            # Emit signal
            self.vehicle_selected.emit(vehicle_id)
    
    def contextMenuEvent(self, event):
        """
        Show context menu for vehicle operations.
        
        Args:
            event: Context menu event
        """
        # Get item at position
        item = self.vehicle_table.itemAt(self.vehicle_table.viewport().mapFrom(self, event.pos()))
        if not item:
            return
            
        # Get vehicle ID
        row = item.row()
        vehicle_id = self.vehicle_table.item(row, 0).text()
        
        # Create context menu
        menu = QMenu(self)
        
        # Add actions
        details_action = QAction("View Details", self)
        details_action.triggered.connect(lambda: self.show_vehicle_details(item))
        menu.addAction(details_action)
        
        # Check if dispatcher is available for other actions
        if self.simulator and self.simulator.dispatcher:
            menu.addSeparator()
            
            # Add task assignment action if vehicle is idle
            state = self.vehicle_table.item(row, 2).text()
            if state == "IDLE":
                assign_action = QAction("Assign Task...", self)
                assign_action.triggered.connect(lambda: self.assign_task(vehicle_id))
                menu.addAction(assign_action)
            
            # Add cancel task action if vehicle has tasks
            task_text = self.vehicle_table.item(row, 4).text()
            if not task_text.startswith("0 task"):
                cancel_action = QAction("Cancel Tasks", self)
                cancel_action.triggered.connect(lambda: self.cancel_tasks(vehicle_id))
                menu.addAction(cancel_action)
            
            menu.addSeparator()
            
            # Add maintenance action
            maintenance_action = QAction("Send for Maintenance", self)
            maintenance_action.triggered.connect(lambda: self.send_to_maintenance(vehicle_id))
            menu.addAction(maintenance_action)
        
        # Show menu
        menu.exec_(event.globalPos())
    
    def assign_task(self, vehicle_id: str):
        """
        Assign a task to a vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
        """
        if not self.simulator or not self.simulator.dispatcher:
            return
            
        # TODO: Implement task assignment dialog
        self.logger.info(f"Assign task to vehicle {vehicle_id}")
        
        # Placeholder: Just log the action
        QMessageBox.information(
            self, 
            "Task Assignment", 
            f"Task assignment for vehicle {vehicle_id} not implemented yet"
        )
    
    def cancel_tasks(self, vehicle_id: str):
        """
        Cancel tasks assigned to a vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
        """
        if not self.simulator or not self.simulator.dispatcher:
            return
            
        # TODO: Implement task cancellation
        self.logger.info(f"Cancel tasks for vehicle {vehicle_id}")
        
        # Placeholder: Just log the action
        QMessageBox.information(
            self, 
            "Task Cancellation", 
            f"Task cancellation for vehicle {vehicle_id} not implemented yet"
        )
    
    def send_to_maintenance(self, vehicle_id: str):
        """
        Send a vehicle to maintenance.
        
        Args:
            vehicle_id: ID of the vehicle
        """
        if not self.simulator or not self.simulator.dispatcher:
            return
            
        # TODO: Implement maintenance action
        self.logger.info(f"Send vehicle {vehicle_id} to maintenance")
        
        # Placeholder: Just log the action
        QMessageBox.information(
            self, 
            "Maintenance", 
            f"Sending vehicle {vehicle_id} to maintenance not implemented yet"
        )