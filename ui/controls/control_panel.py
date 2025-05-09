"""
Control panel for the mining dispatch system.

Provides controls for managing the dispatch system and simulation.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from typing import Dict, List, Optional, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTabWidget, QGroupBox, QFormLayout, QComboBox, QSpinBox, 
    QDoubleSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSlider, QSplitter,
    QRadioButton, QButtonGroup, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QBrush, QIcon, QPixmap

from utils.logger import get_logger
from utils.config import get_config


class StatisticsPanel(QWidget):
    """Panel displaying real-time statistics about the simulation."""
    
    def __init__(self, parent=None):
        """
        Initialize the statistics panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("StatisticsPanel")
        
        # Initialize state
        self.simulator = None
        
        # Initialize UI
        self._init_ui()
        
        # Initialize update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_statistics)
        self.update_timer.start(1000)  # Update every second
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Add title
        title = QLabel("<h3>Simulation Statistics</h3>")
        layout.addWidget(title)
        
        # Create form layout for statistics
        form_layout = QFormLayout()
        layout.addLayout(form_layout)
        
        # Add statistics fields
        self.vehicles_label = QLabel("0")
        form_layout.addRow("Total Vehicles:", self.vehicles_label)
        
        self.tasks_label = QLabel("0")
        form_layout.addRow("Total Tasks:", self.tasks_label)
        
        self.completed_label = QLabel("0")
        form_layout.addRow("Completed Tasks:", self.completed_label)
        
        self.failed_label = QLabel("0")
        form_layout.addRow("Failed Tasks:", self.failed_label)
        
        layout.addSpacing(10)
        
        # Add efficiency metrics
        efficiency_box = QGroupBox("Efficiency Metrics")
        efficiency_layout = QFormLayout(efficiency_box)
        
        self.throughput_label = QLabel("0.0 tasks/hour")
        efficiency_layout.addRow("Throughput:", self.throughput_label)
        
        self.avg_complete_label = QLabel("0.0 seconds")
        efficiency_layout.addRow("Avg. Completion Time:", self.avg_complete_label)
        
        self.avg_wait_label = QLabel("0.0 seconds")
        efficiency_layout.addRow("Avg. Waiting Time:", self.avg_wait_label)
        
        self.utilization_label = QLabel("0.0%")
        efficiency_layout.addRow("Vehicle Utilization:", self.utilization_label)
        
        layout.addWidget(efficiency_box)
        
        layout.addSpacing(10)
        
        # Add distance metrics
        distance_box = QGroupBox("Distance Metrics")
        distance_layout = QFormLayout(distance_box)
        
        self.total_distance_label = QLabel("0.0 meters")
        distance_layout.addRow("Total Distance:", self.total_distance_label)
        
        self.avg_distance_label = QLabel("0.0 meters/task")
        distance_layout.addRow("Avg. Distance/Task:", self.avg_distance_label)
        
        layout.addWidget(distance_box)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def set_simulator(self, simulator):
        """
        Set the simulator instance.
        
        Args:
            simulator: Simulator instance
        """
        self.simulator = simulator
        
        # Update statistics
        self.update_statistics()
    
    @pyqtSlot()
    def update_statistics(self):
        """Update statistics with current data."""
        if not self.simulator:
            return
            
        try:
            # Get metrics from simulator
            metrics = self.simulator.get_metrics()
            
            # Update basic statistics
            self.vehicles_label.setText(str(metrics.total_vehicles))
            self.tasks_label.setText(str(metrics.total_tasks))
            self.completed_label.setText(str(metrics.completed_tasks))
            self.failed_label.setText(str(metrics.failed_tasks))
            
            # Update efficiency metrics
            self.throughput_label.setText(f"{metrics.throughput:.2f} tasks/hour")
            self.avg_complete_label.setText(f"{metrics.average_task_completion_time:.2f} seconds")
            self.avg_wait_label.setText(f"{metrics.average_task_waiting_time:.2f} seconds")
            self.utilization_label.setText(f"{metrics.average_vehicle_utilization:.2f}%")
            
            # Update distance metrics
            self.total_distance_label.setText(f"{metrics.total_distance_traveled:.2f} meters")
            
            avg_distance = 0.0
            if metrics.completed_tasks > 0:
                avg_distance = metrics.total_distance_traveled / metrics.completed_tasks
            
            self.avg_distance_label.setText(f"{avg_distance:.2f} meters/task")
            
        except Exception as e:
            self.logger.error(f"Error updating statistics: {str(e)}", exc_info=True)


class DispatcherControlPanel(QWidget):
    """Panel for controlling the dispatcher configuration and behavior."""
    
    # Signals
    config_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the dispatcher control panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("DispatcherControlPanel")
        
        # Initialize state
        self.simulator = None
        self.dispatcher = None
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Add title
        title = QLabel("<h3>Dispatcher Controls</h3>")
        layout.addWidget(title)
        
        # Create strategy selection
        strategy_box = QGroupBox("Dispatch Strategy")
        strategy_layout = QVBoxLayout(strategy_box)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("Priority-based", "PRIORITY")
        self.strategy_combo.addItem("Nearest Vehicle", "NEAREST")
        self.strategy_combo.addItem("Balanced Load", "BALANCED")
        self.strategy_combo.addItem("Optimized (MIQP)", "OPTIMIZED")
        self.strategy_combo.currentIndexChanged.connect(self.on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo)
        
        layout.addWidget(strategy_box)
        
        # Create task configuration
        task_box = QGroupBox("Task Settings")
        task_layout = QFormLayout(task_box)
        
        self.max_tasks_spin = QSpinBox()
        self.max_tasks_spin.setRange(1, 10)
        self.max_tasks_spin.setValue(5)
        self.max_tasks_spin.valueChanged.connect(self.on_config_changed)
        task_layout.addRow("Max Tasks per Vehicle:", self.max_tasks_spin)
        
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 10.0)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setSuffix(" seconds")
        self.interval_spin.valueChanged.connect(self.on_config_changed)
        task_layout.addRow("Dispatch Interval:", self.interval_spin)
        
        layout.addWidget(task_box)
        
        # Create conflict resolution configuration
        conflict_box = QGroupBox("Conflict Resolution")
        conflict_layout = QVBoxLayout(conflict_box)
        
        self.conflict_check = QCheckBox("Enable Conflict Resolution")
        self.conflict_check.setChecked(True)
        self.conflict_check.stateChanged.connect(self.on_config_changed)
        conflict_layout.addWidget(self.conflict_check)
        
        layout.addWidget(conflict_box)
        
        # Create replan configuration
        replan_box = QGroupBox("Replanning")
        replan_layout = QVBoxLayout(replan_box)
        
        self.replan_check = QCheckBox("Replan on Environment Change")
        self.replan_check.setChecked(True)
        self.replan_check.stateChanged.connect(self.on_config_changed)
        replan_layout.addWidget(self.replan_check)
        
        layout.addWidget(replan_box)
        
        # Add Apply button
        self.apply_button = QPushButton("Apply Configuration")
        self.apply_button.clicked.connect(self.apply_configuration)
        layout.addWidget(self.apply_button)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def set_simulator(self, simulator):
        """
        Set the simulator instance.
        
        Args:
            simulator: Simulator instance
        """
        self.simulator = simulator
        if simulator:
            self.dispatcher = simulator.dispatcher
            
            # Update UI with dispatcher configuration
            self.update_from_dispatcher()
    
    def update_from_dispatcher(self):
        """Update UI controls from dispatcher configuration."""
        if not self.dispatcher:
            return
            
        try:
            # Get dispatcher configuration
            config = getattr(self.dispatcher, 'config', None)
            if not config:
                return
                
            # Update strategy combo
            strategy = getattr(config, 'dispatch_strategy', None)
            if strategy:
                strategy_name = strategy.name if hasattr(strategy, 'name') else str(strategy)
                index = self.strategy_combo.findData(strategy_name)
                if index >= 0:
                    self.strategy_combo.setCurrentIndex(index)
            
            # Update max tasks spin
            max_tasks = getattr(config, 'max_tasks_per_vehicle', None)
            if max_tasks is not None:
                self.max_tasks_spin.setValue(max_tasks)
            
            # Update interval spin
            interval = getattr(config, 'dispatch_interval', None)
            if interval is not None:
                self.interval_spin.setValue(interval)
            
            # Update conflict resolution checkbox
            conflict_enabled = getattr(config, 'conflict_resolution_enabled', None)
            if conflict_enabled is not None:
                self.conflict_check.setChecked(conflict_enabled)
            
            # Update replan checkbox
            replan = getattr(config, 'replan_on_change', None)
            if replan is not None:
                self.replan_check.setChecked(replan)
                
        except Exception as e:
            self.logger.error(f"Error updating from dispatcher: {str(e)}", exc_info=True)
    
    def on_strategy_changed(self, index):
        """
        Handle strategy change.
        
        Args:
            index: Selected index
        """
        self.on_config_changed()
    
    def on_config_changed(self):
        """Handle configuration changes."""
        # Enable apply button
        self.apply_button.setEnabled(True)
        
        # Emit signal
        self.config_changed.emit()
    
    def apply_configuration(self):
        """Apply configuration to dispatcher."""
        if not self.dispatcher:
            return
            
        try:
            # Get dispatcher configuration
            config = getattr(self.dispatcher, 'config', None)
            if not config:
                return
                
            # Update strategy
            strategy_data = self.strategy_combo.currentData()
            if hasattr(config, 'dispatch_strategy') and hasattr(config.dispatch_strategy, '__class__'):
                # Assuming DispatchStrategy is an enum
                strategy_enum = config.dispatch_strategy.__class__
                if hasattr(strategy_enum, strategy_data):
                    config.dispatch_strategy = getattr(strategy_enum, strategy_data)
            
            # Update max tasks
            config.max_tasks_per_vehicle = self.max_tasks_spin.value()
            
            # Update interval
            config.dispatch_interval = self.interval_spin.value()
            
            # Update conflict resolution
            config.conflict_resolution_enabled = self.conflict_check.isChecked()
            
            # Update replan
            config.replan_on_change = self.replan_check.isChecked()
            
            # Apply configuration to dispatcher
            if hasattr(self.dispatcher, 'set_config'):
                self.dispatcher.set_config(config)
                
            # Disable apply button
            self.apply_button.setEnabled(False)
            
            self.logger.info("Applied dispatcher configuration")
            
        except Exception as e:
            self.logger.error(f"Error applying configuration: {str(e)}", exc_info=True)


class ControlPanel(QWidget):
    """
    Main control panel for the mining dispatch system.
    
    Integrates various sub-panels for controlling different aspects
    of the system.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the control panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("ControlPanel")
        
        # Initialize state
        self.simulator = None
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Create scroll area for dispatcher controls
        dispatcher_scroll = QScrollArea()
        dispatcher_scroll.setWidgetResizable(True)
        dispatcher_scroll.setFrameShape(QFrame.NoFrame)
        
        # Create dispatcher control panel
        self.dispatcher_panel = DispatcherControlPanel()
        dispatcher_scroll.setWidget(self.dispatcher_panel)
        
        splitter.addWidget(dispatcher_scroll)
        
        # Create scroll area for statistics
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setFrameShape(QFrame.NoFrame)
        
        # Create statistics panel
        self.stats_panel = StatisticsPanel()
        stats_scroll.setWidget(self.stats_panel)
        
        splitter.addWidget(stats_scroll)
        
        # Set initial sizes
        splitter.setSizes([50, 50])  # Equal width
    
    def set_simulator(self, simulator):
        """
        Set the simulator instance.
        
        Args:
            simulator: Simulator instance
        """
        self.simulator = simulator
        
        # Set simulator in sub-panels
        self.dispatcher_panel.set_simulator(simulator)
        self.stats_panel.set_simulator(simulator)