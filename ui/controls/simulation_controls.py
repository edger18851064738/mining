"""
Simulation controls for the mining dispatch system.

Provides controls for managing the simulation speed and time.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox,
    QFormLayout, QCheckBox, QTimeEdit, QDateTimeEdit, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QDateTime, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon

from utils.logger import get_logger
from utils.config import get_config


class SimulationControls(QWidget):
    """
    Controls for managing simulation parameters and execution.
    
    Allows controlling simulation speed, time, and provides
    play/pause/stop/step functionality.
    """
    
    def __init__(self, parent=None):
        """
        Initialize simulation controls.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("SimulationControls")
        
        # Initialize state
        self.simulator = None
        self.is_running = False
        
        # Initialize UI
        self._init_ui()
        
        # Initialize update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_controls)
        self.update_timer.start(500)  # Update every 500ms
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create time display layout
        time_layout = QHBoxLayout()
        layout.addLayout(time_layout)
        
        # Current time display
        time_layout.addWidget(QLabel("Current Time:"))
        self.time_display = QLabel("00:00:00")
        self.time_display.setStyleSheet("font-size: 14px; font-weight: bold;")
        time_layout.addWidget(self.time_display)
        
        time_layout.addSpacing(20)
        
        # Elapsed time display
        time_layout.addWidget(QLabel("Elapsed:"))
        self.elapsed_display = QLabel("00:00:00")
        time_layout.addWidget(self.elapsed_display)
        
        time_layout.addStretch()
        
        # Status display
        time_layout.addWidget(QLabel("Status:"))
        self.status_display = QLabel("Stopped")
        self.status_display.setStyleSheet("font-weight: bold;")
        time_layout.addWidget(self.status_display)
        
        # Add separator
        layout.addSpacing(5)
        
        # Create control buttons layout
        buttons_layout = QHBoxLayout()
        layout.addLayout(buttons_layout)
        
        # Play button
        self.play_button = QPushButton()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_button.setToolTip("Start Simulation")
        self.play_button.clicked.connect(self.play_simulation)
        buttons_layout.addWidget(self.play_button)
        
        # Pause button
        self.pause_button = QPushButton()
        self.pause_button.setIcon(QIcon.fromTheme("media-playback-pause"))
        self.pause_button.setToolTip("Pause Simulation")
        self.pause_button.clicked.connect(self.pause_simulation)
        self.pause_button.setEnabled(False)
        buttons_layout.addWidget(self.pause_button)
        
        # Stop button
        self.stop_button = QPushButton()
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_button.setToolTip("Stop Simulation")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        # Step button
        self.step_button = QPushButton()
        self.step_button.setIcon(QIcon.fromTheme("media-skip-forward"))
        self.step_button.setToolTip("Step Simulation")
        self.step_button.clicked.connect(self.step_simulation)
        buttons_layout.addWidget(self.step_button)
        
        buttons_layout.addSpacing(20)
        
        # Simulation speed controls
        buttons_layout.addWidget(QLabel("Speed:"))
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItem("0.1x", 0.1)
        self.speed_combo.addItem("0.5x", 0.5)
        self.speed_combo.addItem("1x", 1.0)
        self.speed_combo.addItem("2x", 2.0)
        self.speed_combo.addItem("5x", 5.0)
        self.speed_combo.addItem("10x", 10.0)
        self.speed_combo.addItem("50x", 50.0)
        self.speed_combo.addItem("100x", 100.0)
        self.speed_combo.setCurrentText("10x")  # Default 10x
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        buttons_layout.addWidget(self.speed_combo)
        
        buttons_layout.addSpacing(20)
        
        # Step size control
        buttons_layout.addWidget(QLabel("Step Size:"))
        
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 60.0)
        self.step_spin.setValue(1.0)
        self.step_spin.setSingleStep(0.1)
        self.step_spin.setSuffix(" sec")
        buttons_layout.addWidget(self.step_spin)
        
        # Add separator
        layout.addSpacing(5)
        
        # Create time control layout
        time_control_layout = QHBoxLayout()
        layout.addLayout(time_control_layout)
        
        # Add run until time control
        time_control_layout.addWidget(QLabel("Run Until:"))
        
        self.until_time = QDateTimeEdit()
        self.until_time.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.until_time.setDateTime(QDateTime.currentDateTime().addSecs(3600))  # 1 hour from now
        time_control_layout.addWidget(self.until_time)
        
        self.run_until_button = QPushButton("Run")
        self.run_until_button.clicked.connect(self.run_until_time)
        time_control_layout.addWidget(self.run_until_button)
        
        time_control_layout.addSpacing(20)
        
        # Add run for duration control
        time_control_layout.addWidget(QLabel("Run For:"))
        
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 86400)  # 1 second to 24 hours
        self.duration_spin.setValue(60)  # Default 60 seconds
        self.duration_spin.setSingleStep(10)
        time_control_layout.addWidget(self.duration_spin)
        
        self.duration_unit = QComboBox()
        self.duration_unit.addItem("Seconds", 1)
        self.duration_unit.addItem("Minutes", 60)
        self.duration_unit.addItem("Hours", 3600)
        time_control_layout.addWidget(self.duration_unit)
        
        self.run_for_button = QPushButton("Run")
        self.run_for_button.clicked.connect(self.run_for_duration)
        time_control_layout.addWidget(self.run_for_button)
    
    def set_simulator(self, simulator):
        """
        Set the simulator instance.
        
        Args:
            simulator: Simulator instance
        """
        self.simulator = simulator
        
        # Update controls
        self.update_controls()
    
    @pyqtSlot()
    def update_controls(self):
        """Update controls with current simulation state."""
        if not self.simulator:
            return
            
        try:
            # Get current time
            current_time = self.simulator.clock.current_time
            time_str = current_time.strftime("%H:%M:%S")
            self.time_display.setText(time_str)
            
            # Get elapsed time
            elapsed_time = self.simulator.clock.elapsed_time
            elapsed_str = f"{int(elapsed_time.total_seconds() // 3600):02d}:{int((elapsed_time.total_seconds() % 3600) // 60):02d}:{int(elapsed_time.total_seconds() % 60):02d}"
            self.elapsed_display.setText(elapsed_str)
            
            # Get simulation status
            status = self.simulator.status.name if hasattr(self.simulator.status, 'name') else str(self.simulator.status)
            self.status_display.setText(status)
            
            # Set button states based on status
            if status == "RUNNING":
                self.is_running = True
                self.play_button.setEnabled(False)
                self.pause_button.setEnabled(True)
                self.stop_button.setEnabled(True)
                self.step_button.setEnabled(False)
            elif status == "PAUSED":
                self.is_running = False
                self.play_button.setEnabled(True)
                self.pause_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.step_button.setEnabled(True)
            else:  # STOPPED or others
                self.is_running = False
                self.play_button.setEnabled(True)
                self.pause_button.setEnabled(False)
                self.stop_button.setEnabled(False)
                self.step_button.setEnabled(True)
            
            # Update until_time with current time + 1 hour if simulator just started
            if not self.is_running and status == "RUNNING":
                self.until_time.setDateTime(
                    QDateTime.fromString(
                        current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "yyyy-MM-dd HH:mm:ss"
                    ).addSecs(3600)  # 1 hour from now
                )
            
        except Exception as e:
            self.logger.error(f"Error updating controls: {str(e)}", exc_info=True)
    
    def play_simulation(self):
        """Start the simulation."""
        if not self.simulator:
            return
            
        try:
            self.simulator.start()
            self.is_running = True
            
            # Update controls immediately
            self.update_controls()
            
        except Exception as e:
            self.logger.error(f"Error starting simulation: {str(e)}", exc_info=True)
    
    def pause_simulation(self):
        """Pause the simulation."""
        if not self.simulator or not self.is_running:
            return
            
        try:
            self.simulator.pause()
            self.is_running = False
            
            # Update controls immediately
            self.update_controls()
            
        except Exception as e:
            self.logger.error(f"Error pausing simulation: {str(e)}", exc_info=True)
    
    def stop_simulation(self):
        """Stop the simulation."""
        if not self.simulator:
            return
            
        try:
            self.simulator.stop()
            self.is_running = False
            
            # Update controls immediately
            self.update_controls()
            
        except Exception as e:
            self.logger.error(f"Error stopping simulation: {str(e)}", exc_info=True)
    
    def step_simulation(self):
        """Step the simulation."""
        if not self.simulator:
            return
            
        try:
            # Get step size
            step_size = self.step_spin.value()
            
            # Step the simulation
            self.simulator.step(step_size)
            
            # Update controls immediately
            self.update_controls()
            
        except Exception as e:
            self.logger.error(f"Error stepping simulation: {str(e)}", exc_info=True)
    
    def change_speed(self):
        """Change simulation speed."""
        if not self.simulator:
            return
            
        try:
            # Get selected speed factor
            speed_factor = self.speed_combo.currentData()
            
            # Set speed factor if clock is available
            if hasattr(self.simulator, 'clock'):
                self.simulator.clock.speed_factor = speed_factor
                self.logger.info(f"Simulation speed set to {speed_factor}x")
            
        except Exception as e:
            self.logger.error(f"Error changing simulation speed: {str(e)}", exc_info=True)
    
    def run_until_time(self):
        """Run simulation until specified time."""
        if not self.simulator:
            return
            
        try:
            # Get target time
            target_time = self.until_time.dateTime().toPyDateTime()
            
            # Convert to sim time if needed (PyQt might return local time)
            # This depends on how your simulation clock works
            
            # Run until target time
            self.logger.info(f"Running simulation until {target_time}")
            
            # Start simulation if not running
            if not self.is_running:
                self.simulator.start()
                self.is_running = True
            
            # Run until target time
            self.simulator.run_until(target_time)
            
            # Update controls immediately
            self.update_controls()
            
        except Exception as e:
            self.logger.error(f"Error running until time: {str(e)}", exc_info=True)
    
    def run_for_duration(self):
        """Run simulation for specified duration."""
        if not self.simulator:
            return
            
        try:
            # Get duration
            duration_value = self.duration_spin.value()
            duration_unit = self.duration_unit.currentData()
            duration_seconds = duration_value * duration_unit
            
            # Create timedelta
            duration = timedelta(seconds=duration_seconds)
            
            self.logger.info(f"Running simulation for {duration}")
            
            # Start simulation if not running
            if not self.is_running:
                self.simulator.start()
                self.is_running = True
            
            # Run for duration
            self.simulator.run_for(duration)
            
            # Update controls immediately
            self.update_controls()
            
        except Exception as e:
            self.logger.error(f"Error running for duration: {str(e)}", exc_info=True)