"""
Settings dialog for the mining dispatch system.

Provides a dialog for configuring application and simulation settings.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from typing import Dict, List, Optional, Any

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTabWidget, QWidget, QFormLayout, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QLineEdit, QGroupBox, QColorDialog,
    QDialogButtonBox, QFileDialog, QSlider, QMessageBox
)
from PyQt5.QtCore import Qt, QSettings, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QIcon

from utils.logger import get_logger
from utils.config import get_config, set as set_config


class ColorButton(QPushButton):
    """Button that displays and allows selecting a color."""
    
    color_changed = pyqtSignal(QColor)
    
    def __init__(self, color=None, parent=None):
        """
        Initialize color button.
        
        Args:
            color: Initial color (QColor or string)
            parent: Parent widget
        """
        super().__init__(parent)
        self.setMinimumWidth(60)
        
        # Set initial color
        if color is None:
            color = QColor(Qt.white)
        elif isinstance(color, str):
            color = QColor(color)
            
        self.color = color
        
        # Update button style
        self._update_button()
        
        # Connect click handler
        self.clicked.connect(self._choose_color)
    
    def _update_button(self):
        """Update button style to show current color."""
        self.setStyleSheet(f"""
            ColorButton {{
                background-color: {self.color.name()};
                border: 1px solid #888888;
                border-radius: 3px;
            }}
        """)
    
    def _choose_color(self):
        """Show color dialog and update color if accepted."""
        color = QColorDialog.getColor(self.color, self.parent(), "Select Color")
        if color.isValid():
            self.set_color(color)
    
    def set_color(self, color):
        """
        Set the button color.
        
        Args:
            color: New color (QColor or string)
        """
        if isinstance(color, str):
            color = QColor(color)
            
        self.color = color
        self._update_button()
        self.color_changed.emit(color)
    
    def get_color(self):
        """
        Get the current color.
        
        Returns:
            QColor: Current color
        """
        return self.color


class GeneralSettingsTab(QWidget):
    """Tab for general application settings."""
    
    def __init__(self, parent=None):
        """
        Initialize general settings tab.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("GeneralSettings")
        
        # Load configuration
        self.config = get_config()
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QFormLayout()
        layout.addLayout(form_layout)
        
        # Add UI settings
        ui_group = QGroupBox("User Interface")
        ui_layout = QFormLayout(ui_group)
        
        # Window title
        self.title_edit = QLineEdit(self.config.ui.window_title)
        ui_layout.addRow("Window Title:", self.title_edit)
        
        # Window size
        size_layout = QHBoxLayout()
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(800, 3840)
        self.width_spin.setValue(self.config.ui.window_width)
        size_layout.addWidget(self.width_spin)
        
        size_layout.addWidget(QLabel("x"))
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(600, 2160)
        self.height_spin.setValue(self.config.ui.window_height)
        size_layout.addWidget(self.height_spin)
        
        ui_layout.addRow("Window Size:", size_layout)
        
        # Theme
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Dark", "dark")
        current_theme = getattr(self.config.ui, 'theme', 'light')
        index = self.theme_combo.findData(current_theme)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        ui_layout.addRow("Theme:", self.theme_combo)
        
        # Colors
        self.primary_color = ColorButton(self.config.ui.primary_color)
        ui_layout.addRow("Primary Color:", self.primary_color)
        
        self.secondary_color = ColorButton(self.config.ui.secondary_color)
        ui_layout.addRow("Secondary Color:", self.secondary_color)
        
        layout.addWidget(ui_group)
        
        # Add system settings
        system_group = QGroupBox("System")
        system_layout = QFormLayout(system_group)
        
        # Debug mode
        self.debug_check = QCheckBox("Enable Debug Mode")
        self.debug_check.setChecked(self.config.debug_mode)
        system_layout.addRow("", self.debug_check)
        
        # Multithreading
        self.threading_check = QCheckBox("Enable Multithreading")
        self.threading_check.setChecked(self.config.multithreading)
        system_layout.addRow("", self.threading_check)
        
        # Thread pool size
        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, 32)
        self.thread_spin.setValue(self.config.thread_pool_size)
        system_layout.addRow("Thread Pool Size:", self.thread_spin)
        
        layout.addWidget(system_group)
        
        # Add logging settings
        log_group = QGroupBox("Logging")
        log_layout = QFormLayout(log_group)
        
        # Log level
        self.log_level = QComboBox()
        self.log_level.addItem("Debug", "DEBUG")
        self.log_level.addItem("Info", "INFO")
        self.log_level.addItem("Warning", "WARNING")
        self.log_level.addItem("Error", "ERROR")
        
        current_level = getattr(self.config.logging, 'level', 'INFO')
        index = self.log_level.findData(current_level)
        if index >= 0:
            self.log_level.setCurrentIndex(index)
        log_layout.addRow("Log Level:", self.log_level)
        
        # File logging
        self.file_check = QCheckBox("Enable File Logging")
        self.file_check.setChecked(self.config.logging.file_output)
        log_layout.addRow("", self.file_check)
        
        # Log file
        self.log_file = QLineEdit(self.config.logging.file_path)
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.log_file)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_log_file)
        file_layout.addWidget(browse_button)
        
        log_layout.addRow("Log File:", file_layout)
        
        layout.addWidget(log_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def _browse_log_file(self):
        """Browse for log file location."""
        file_path = QFileDialog.getSaveFileName(
            self, "Select Log File", self.log_file.text(), "Log Files (*.log);;All Files (*)"
        )[0]
        
        if file_path:
            self.log_file.setText(file_path)
    
    def save_settings(self):
        """
        Save settings to configuration.
        
        Returns:
            bool: True if settings were saved successfully
        """
        try:
            # Save UI settings
            set_config("ui", "window_title", self.title_edit.text())
            set_config("ui", "window_width", self.width_spin.value())
            set_config("ui", "window_height", self.height_spin.value())
            set_config("ui", "theme", self.theme_combo.currentData())
            set_config("ui", "primary_color", self.primary_color.get_color().name())
            set_config("ui", "secondary_color", self.secondary_color.get_color().name())
            
            # Save system settings
            set_config("system", "debug_mode", self.debug_check.isChecked())
            set_config("system", "multithreading", self.threading_check.isChecked())
            set_config("system", "thread_pool_size", self.thread_spin.value())
            
            # Save logging settings
            set_config("logging", "level", self.log_level.currentData())
            set_config("logging", "file_output", self.file_check.isChecked())
            set_config("logging", "file_path", self.log_file.text())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {str(e)}", exc_info=True)
            return False


class MapSettingsTab(QWidget):
    """Tab for map settings."""
    
    def __init__(self, parent=None):
        """
        Initialize map settings tab.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("MapSettings")
        
        # Load configuration
        self.config = get_config()
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QFormLayout()
        layout.addLayout(form_layout)
        
        # Add map settings
        map_group = QGroupBox("Map Dimensions")
        map_layout = QFormLayout(map_group)
        
        # Grid size
        self.grid_size = QSpinBox()
        self.grid_size.setRange(100, 10000)
        self.grid_size.setValue(self.config.map.grid_size)
        map_layout.addRow("Grid Size:", self.grid_size)
        
        # Grid nodes
        self.grid_nodes = QSpinBox()
        self.grid_nodes.setRange(10, 1000)
        self.grid_nodes.setValue(self.config.map.grid_nodes)
        map_layout.addRow("Grid Nodes:", self.grid_nodes)
        
        layout.addWidget(map_group)
        
        # Add safety settings
        safety_group = QGroupBox("Safety Parameters")
        safety_layout = QFormLayout(safety_group)
        
        # Safe radius
        self.safe_radius = QSpinBox()
        self.safe_radius.setRange(1, 100)
        self.safe_radius.setValue(self.config.map.safe_radius)
        safety_layout.addRow("Safe Radius:", self.safe_radius)
        
        # Obstacle density
        self.obstacle_density = QDoubleSpinBox()
        self.obstacle_density.setRange(0.0, 1.0)
        self.obstacle_density.setValue(self.config.map.obstacle_density)
        self.obstacle_density.setSingleStep(0.05)
        safety_layout.addRow("Obstacle Density:", self.obstacle_density)
        
        layout.addWidget(safety_group)
        
        # Add visualization settings
        visual_group = QGroupBox("Visualization")
        visual_layout = QFormLayout(visual_group)
        
        # Show grid
        self.show_grid = QCheckBox("Show Grid")
        self.show_grid.setChecked(self.config.map.show_grid)
        visual_layout.addRow("", self.show_grid)
        
        # Grid color
        self.grid_color = ColorButton(self.config.map.grid_color)
        visual_layout.addRow("Grid Color:", self.grid_color)
        
        # Obstacle color
        self.obstacle_color = ColorButton(self.config.map.obstacle_color)
        visual_layout.addRow("Obstacle Color:", self.obstacle_color)
        
        layout.addWidget(visual_group)
        
        # Key locations editor
        locations_group = QGroupBox("Key Locations")
        locations_layout = QVBoxLayout(locations_group)
        
        # Table layout would go here
        locations_layout.addWidget(QLabel("Key locations editor not implemented yet"))
        
        layout.addWidget(locations_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def save_settings(self):
        """
        Save settings to configuration.
        
        Returns:
            bool: True if settings were saved successfully
        """
        try:
            # Save map settings
            set_config("map", "grid_size", self.grid_size.value())
            set_config("map", "grid_nodes", self.grid_nodes.value())
            
            # Save safety settings
            set_config("map", "safe_radius", self.safe_radius.value())
            set_config("map", "obstacle_density", self.obstacle_density.value())
            
            # Save visualization settings
            set_config("map", "show_grid", self.show_grid.isChecked())
            set_config("map", "grid_color", self.grid_color.get_color().name())
            set_config("map", "obstacle_color", self.obstacle_color.get_color().name())
            
            # Key locations would be saved here
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {str(e)}", exc_info=True)
            return False


class AlgorithmSettingsTab(QWidget):
    """Tab for algorithm settings."""
    
    def __init__(self, parent=None):
        """
        Initialize algorithm settings tab.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("AlgorithmSettings")
        
        # Load configuration
        self.config = get_config()
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QFormLayout()
        layout.addLayout(form_layout)
        
        # Add path planning settings
        planning_group = QGroupBox("Path Planning")
        planning_layout = QFormLayout(planning_group)
        
        # Planner type
        self.planner_combo = QComboBox()
        self.planner_combo.addItem("Hybrid A*", "hybrid_astar")
        self.planner_combo.addItem("RRT", "rrt")
        self.planner_combo.addItem("RRT*", "rrt_star")
        
        current_planner = getattr(self.config.algorithm, 'path_planner', 'hybrid_astar')
        index = self.planner_combo.findData(current_planner)
        if index >= 0:
            self.planner_combo.setCurrentIndex(index)
        planning_layout.addRow("Path Planner:", self.planner_combo)
        
        # Step size
        self.step_size = QDoubleSpinBox()
        self.step_size.setRange(0.1, 10.0)
        self.step_size.setValue(self.config.algorithm.step_size)
        self.step_size.setSingleStep(0.1)
        planning_layout.addRow("Step Size:", self.step_size)
        
        # Grid resolution
        self.grid_resolution = QDoubleSpinBox()
        self.grid_resolution.setRange(0.1, 10.0)
        self.grid_resolution.setValue(self.config.algorithm.grid_resolution)
        self.grid_resolution.setSingleStep(0.1)
        planning_layout.addRow("Grid Resolution:", self.grid_resolution)
        
        # Max iterations
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(100, 50000)
        self.max_iterations.setValue(self.config.algorithm.max_iterations)
        self.max_iterations.setSingleStep(100)
        planning_layout.addRow("Max Iterations:", self.max_iterations)
        
        layout.addWidget(planning_group)
        
        # Add conflict resolution settings
        conflict_group = QGroupBox("Conflict Resolution")
        conflict_layout = QFormLayout(conflict_group)
        
        # Resolution method
        self.conflict_combo = QComboBox()
        self.conflict_combo.addItem("Conflict-Based Search (CBS)", "cbs")
        self.conflict_combo.addItem("Enhanced CBS", "ecbs")
        self.conflict_combo.addItem("Priority-Based", "priority")
        
        current_resolution = getattr(self.config.algorithm, 'conflict_resolution', 'cbs')
        index = self.conflict_combo.findData(current_resolution)
        if index >= 0:
            self.conflict_combo.setCurrentIndex(index)
        conflict_layout.addRow("Method:", self.conflict_combo)
        
        # Collision threshold
        self.collision_threshold = QDoubleSpinBox()
        self.collision_threshold.setRange(0.1, 10.0)
        self.collision_threshold.setValue(self.config.algorithm.collision_threshold)
        self.collision_threshold.setSingleStep(0.1)
        conflict_layout.addRow("Collision Threshold:", self.collision_threshold)
        
        # Max replanning attempts
        self.max_replanning = QSpinBox()
        self.max_replanning.setRange(1, 20)
        self.max_replanning.setValue(self.config.algorithm.max_replanning_attempts)
        conflict_layout.addRow("Max Replanning:", self.max_replanning)
        
        layout.addWidget(conflict_group)
        
        # Add task allocation settings
        allocation_group = QGroupBox("Task Allocation")
        allocation_layout = QFormLayout(allocation_group)
        
        # Allocator type
        self.allocator_combo = QComboBox()
        self.allocator_combo.addItem("Priority-Based", "priority")
        self.allocator_combo.addItem("Auction-Based", "auction")
        self.allocator_combo.addItem("Mixed Integer Programming", "miqp")
        
        current_allocator = getattr(self.config.algorithm, 'task_allocator', 'priority')
        index = self.allocator_combo.findData(current_allocator)
        if index >= 0:
            self.allocator_combo.setCurrentIndex(index)
        allocation_layout.addRow("Allocator:", self.allocator_combo)
        
        layout.addWidget(allocation_group)
        
        # Add advanced settings
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout(advanced_group)
        
        # Enable smoothing
        self.smoothing_check = QCheckBox("Enable Path Smoothing")
        self.smoothing_check.setChecked(self.config.algorithm.smoothing_enabled)
        advanced_layout.addRow("", self.smoothing_check)
        
        # Smoothing factor
        self.smoothing_factor = QDoubleSpinBox()
        self.smoothing_factor.setRange(0.0, 1.0)
        self.smoothing_factor.setValue(self.config.algorithm.smoothing_factor)
        self.smoothing_factor.setSingleStep(0.05)
        advanced_layout.addRow("Smoothing Factor:", self.smoothing_factor)
        
        # Use RS curves
        self.rs_curves_check = QCheckBox("Use Reeds-Shepp Curves")
        self.rs_curves_check.setChecked(self.config.algorithm.use_rs_curves)
        advanced_layout.addRow("", self.rs_curves_check)
        
        layout.addWidget(advanced_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def save_settings(self):
        """
        Save settings to configuration.
        
        Returns:
            bool: True if settings were saved successfully
        """
        try:
            # Save path planning settings
            set_config("algorithm", "path_planner", self.planner_combo.currentData())
            set_config("algorithm", "step_size", self.step_size.value())
            set_config("algorithm", "grid_resolution", self.grid_resolution.value())
            set_config("algorithm", "max_iterations", self.max_iterations.value())
            
            # Save conflict resolution settings
            set_config("algorithm", "conflict_resolution", self.conflict_combo.currentData())
            set_config("algorithm", "collision_threshold", self.collision_threshold.value())
            set_config("algorithm", "max_replanning_attempts", self.max_replanning.value())
            
            # Save task allocation settings
            set_config("algorithm", "task_allocator", self.allocator_combo.currentData())
            
            # Save advanced settings
            set_config("algorithm", "smoothing_enabled", self.smoothing_check.isChecked())
            set_config("algorithm", "smoothing_factor", self.smoothing_factor.value())
            set_config("algorithm", "use_rs_curves", self.rs_curves_check.isChecked())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {str(e)}", exc_info=True)
            return False


class SettingsDialog(QDialog):
    """
    Dialog for configuring application settings.
    
    Provides tabs for different setting categories.
    """
    
    def __init__(self, parent=None):
        """
        Initialize settings dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.setWindowTitle("Settings")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Add tabs
        self.general_tab = GeneralSettingsTab()
        tab_widget.addTab(self.general_tab, "General")
        
        self.map_tab = MapSettingsTab()
        tab_widget.addTab(self.map_tab, "Map")
        
        self.algorithm_tab = AlgorithmSettingsTab()
        tab_widget.addTab(self.algorithm_tab, "Algorithms")
        
        # Add button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def accept(self):
        """Handle dialog acceptance."""
        # Save settings from all tabs
        general_success = self.general_tab.save_settings()
        map_success = self.map_tab.save_settings()
        algorithm_success = self.algorithm_tab.save_settings()
        
        if general_success and map_success and algorithm_success:
            super().accept()
        else:
            QMessageBox.warning(
                self,
                "Settings Error",
                "Some settings could not be saved. Check the log for details."
            )