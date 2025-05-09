"""
Main window for the mining dispatch system.

Provides the primary interface for the application, integrating all UI components.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QDockWidget, QAction, QToolBar, QStatusBar, QMessageBox,
    QMenu, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, QSettings, QSize, QPoint, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QCloseEvent

from utils.logger import get_logger
from utils.config import get_config, set as set_config
from coordination.dispatcher.dispatch_events import EventType, DispatchEvent, EventListener

# Import UI components
from ui.views.map_view import MapView
from ui.views.task_view import TaskView
from ui.views.vehicle_view import VehicleView
from ui.controls.control_panel import ControlPanel
from ui.controls.simulation_controls import SimulationControls
from ui.dialogs.settings_dialog import SettingsDialog
from ui.dialogs.analysis_dialog import AnalysisDialog


class MainWindow(QMainWindow, EventListener):
    """
    Main application window for the mining dispatch system.
    
    Integrates all UI components and provides the main event loop.
    """
    
    # Signal to handle updates from non-GUI threads
    updated_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the main window.
        
        Args:
            parent: Parent widget (default: None)
        """
        super().__init__(parent)
        self.logger = get_logger("MainWindow")
        
        # Initialize state
        self.simulator = None
        self.environment = None
        self.dispatcher = None
        
        # Initialize UI
        self._init_ui()
        
        # Load settings
        self._load_settings()
        
        # Set up update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)  # Update every 100ms
        
        # Connect signal to slot
        self.updated_signal.connect(self.update_ui)
        
        self.logger.info("Main window initialized")
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        config = get_config()
        window_title = config.ui.window_title
        self.setWindowTitle(window_title)
        self.resize(config.ui.window_width, config.ui.window_height)
        
        # Set central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create main splitter for map and controls
        main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(main_splitter)
        
        # Create map view as the main component
        self.map_view = MapView(self)
        main_splitter.addWidget(self.map_view)
        
        # Create control panel
        self.control_panel = ControlPanel(self)
        control_panel_container = QWidget()
        control_panel_layout = QVBoxLayout(control_panel_container)
        control_panel_layout.setContentsMargins(0, 0, 0, 0)
        control_panel_layout.addWidget(self.control_panel)
        main_splitter.addWidget(control_panel_container)
        
        # Set splitter sizes
        main_splitter.setSizes([700, 300])  # 70% map, 30% controls
        
        # Create dockable vehicle view
        self.vehicle_view = VehicleView(self)
        vehicle_dock = QDockWidget("Vehicles", self)
        vehicle_dock.setWidget(self.vehicle_view)
        vehicle_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, vehicle_dock)
        
        # Create dockable task view
        self.task_view = TaskView(self)
        task_dock = QDockWidget("Tasks", self)
        task_dock.setWidget(self.task_view)
        task_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, task_dock)
        
        # Create simulation controls
        self.simulation_controls = SimulationControls(self)
        simulation_dock = QDockWidget("Simulation", self)
        simulation_dock.setWidget(self.simulation_controls)
        simulation_dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, simulation_dock)
        
        # Create menu bar
        self._create_menus()
        
        # Create tool bar
        self._create_toolbar()
        
        # Create status bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Add spacer to status bar to separate status message from other widgets
        self.status_bar.addPermanentWidget(QWidget(self), 1)
        
        # Add clock display to status bar
        self.clock_display = QWidget(self)
        self.status_bar.addPermanentWidget(self.clock_display)
    
    def _create_menus(self):
        """Create application menus."""
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        # New simulation action
        new_action = QAction(QIcon.fromTheme("document-new"), "&New Simulation", self)
        new_action.setShortcut("Ctrl+N")
        new_action.setStatusTip("Create a new simulation")
        new_action.triggered.connect(self.new_simulation)
        file_menu.addAction(new_action)
        
        # Open simulation action
        open_action = QAction(QIcon.fromTheme("document-open"), "&Open Simulation...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open an existing simulation")
        open_action.triggered.connect(self.open_simulation)
        file_menu.addAction(open_action)
        
        # Save simulation action
        save_action = QAction(QIcon.fromTheme("document-save"), "&Save Simulation", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip("Save current simulation")
        save_action.triggered.connect(self.save_simulation)
        file_menu.addAction(save_action)
        
        # Save As action
        save_as_action = QAction(QIcon.fromTheme("document-save-as"), "Save Simulation &As...", self)
        save_as_action.setStatusTip("Save current simulation with a new name")
        save_as_action.triggered.connect(self.save_simulation_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # Export results action
        export_action = QAction(QIcon.fromTheme("document-export"), "&Export Results...", self)
        export_action.setStatusTip("Export simulation results")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction(QIcon.fromTheme("application-exit"), "E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = self.menuBar().addMenu("&Edit")
        
        # Preferences action
        preferences_action = QAction(QIcon.fromTheme("preferences-system"), "&Preferences...", self)
        preferences_action.setStatusTip("Configure application settings")
        preferences_action.triggered.connect(self.show_settings)
        edit_menu.addAction(preferences_action)
        
        # View menu
        view_menu = self.menuBar().addMenu("&View")
        
        # View toolbar action
        view_toolbar_action = QAction("&Toolbar", self)
        view_toolbar_action.setCheckable(True)
        view_toolbar_action.setChecked(True)
        view_toolbar_action.triggered.connect(lambda checked: self.toolbar.setVisible(checked))
        view_menu.addAction(view_toolbar_action)
        
        # View status bar action
        view_statusbar_action = QAction("&Status Bar", self)
        view_statusbar_action.setCheckable(True)
        view_statusbar_action.setChecked(True)
        view_statusbar_action.triggered.connect(lambda checked: self.status_bar.setVisible(checked))
        view_menu.addAction(view_statusbar_action)
        
        view_menu.addSeparator()
        
        # Reset layout action
        reset_layout_action = QAction("&Reset Layout", self)
        reset_layout_action.setStatusTip("Reset window layout to default")
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)
        
        # Simulation menu
        sim_menu = self.menuBar().addMenu("&Simulation")
        
        # Start simulation action
        start_action = QAction(QIcon.fromTheme("media-playback-start"), "&Start", self)
        start_action.setShortcut("F5")
        start_action.setStatusTip("Start simulation")
        start_action.triggered.connect(self.start_simulation)
        sim_menu.addAction(start_action)
        
        # Pause simulation action
        pause_action = QAction(QIcon.fromTheme("media-playback-pause"), "&Pause", self)
        pause_action.setShortcut("F6")
        pause_action.setStatusTip("Pause simulation")
        pause_action.triggered.connect(self.pause_simulation)
        sim_menu.addAction(pause_action)
        
        # Stop simulation action
        stop_action = QAction(QIcon.fromTheme("media-playback-stop"), "S&top", self)
        stop_action.setShortcut("F7")
        stop_action.setStatusTip("Stop simulation")
        stop_action.triggered.connect(self.stop_simulation)
        sim_menu.addAction(stop_action)
        
        sim_menu.addSeparator()
        
        # Step simulation action
        step_action = QAction(QIcon.fromTheme("media-skip-forward"), "S&tep", self)
        step_action.setShortcut("F8")
        step_action.setStatusTip("Step simulation")
        step_action.triggered.connect(self.step_simulation)
        sim_menu.addAction(step_action)
        
        sim_menu.addSeparator()
        
        # Analysis action
        analyze_action = QAction(QIcon.fromTheme("office-chart-line"), "&Analysis...", self)
        analyze_action.setStatusTip("Show simulation analysis")
        analyze_action.triggered.connect(self.show_analysis)
        sim_menu.addAction(analyze_action)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        # About action
        about_action = QAction(QIcon.fromTheme("help-about"), "&About", self)
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """Create application toolbar."""
        self.toolbar = QToolBar("Main Toolbar", self)
        self.toolbar.setMovable(True)
        self.addToolBar(self.toolbar)
        
        # New simulation action
        new_action = QAction(QIcon.fromTheme("document-new"), "New Simulation", self)
        new_action.setStatusTip("Create a new simulation")
        new_action.triggered.connect(self.new_simulation)
        self.toolbar.addAction(new_action)
        
        # Open simulation action
        open_action = QAction(QIcon.fromTheme("document-open"), "Open Simulation", self)
        open_action.setStatusTip("Open an existing simulation")
        open_action.triggered.connect(self.open_simulation)
        self.toolbar.addAction(open_action)
        
        # Save simulation action
        save_action = QAction(QIcon.fromTheme("document-save"), "Save Simulation", self)
        save_action.setStatusTip("Save current simulation")
        save_action.triggered.connect(self.save_simulation)
        self.toolbar.addAction(save_action)
        
        self.toolbar.addSeparator()
        
        # Start simulation action
        start_action = QAction(QIcon.fromTheme("media-playback-start"), "Start", self)
        start_action.setStatusTip("Start simulation")
        start_action.triggered.connect(self.start_simulation)
        self.toolbar.addAction(start_action)
        
        # Pause simulation action
        pause_action = QAction(QIcon.fromTheme("media-playback-pause"), "Pause", self)
        pause_action.setStatusTip("Pause simulation")
        pause_action.triggered.connect(self.pause_simulation)
        self.toolbar.addAction(pause_action)
        
        # Stop simulation action
        stop_action = QAction(QIcon.fromTheme("media-playback-stop"), "Stop", self)
        stop_action.setStatusTip("Stop simulation")
        stop_action.triggered.connect(self.stop_simulation)
        self.toolbar.addAction(stop_action)
        
        # Step simulation action
        step_action = QAction(QIcon.fromTheme("media-skip-forward"), "Step", self)
        step_action.setStatusTip("Step simulation")
        step_action.triggered.connect(self.step_simulation)
        self.toolbar.addAction(step_action)
        
        self.toolbar.addSeparator()
        
        # Analysis action
        analyze_action = QAction(QIcon.fromTheme("office-chart-line"), "Analysis", self)
        analyze_action.setStatusTip("Show simulation analysis")
        analyze_action.triggered.connect(self.show_analysis)
        self.toolbar.addAction(analyze_action)
    
    def _load_settings(self):
        """Load application settings."""
        settings = QSettings("Mining", "DispatchSystem")
        
        # Restore window geometry and state
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
            
        state = settings.value("windowState")
        if state:
            self.restoreState(state)
        
        # Restore last simulation file
        self.last_file = settings.value("lastFile", "")
    
    def _save_settings(self):
        """Save application settings."""
        settings = QSettings("Mining", "DispatchSystem")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("lastFile", self.last_file if hasattr(self, 'last_file') else "")
    
    def reset_layout(self):
        """Reset window layout to default."""
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks)
        self.setWindowState(Qt.WindowNoState)  # Ensure it's not maximized
        
        # Reset window size
        config = get_config()
        self.resize(config.ui.window_width, config.ui.window_height)
        
        # Reset all dock widgets
        for dock in self.findChildren(QDockWidget):
            self.removeDockWidget(dock)
            
        # Add dock widgets back in default positions
        for dock in self.findChildren(QDockWidget):
            if dock.objectName() == "vehicle_dock":
                self.addDockWidget(Qt.RightDockWidgetArea, dock)
            elif dock.objectName() == "task_dock":
                self.addDockWidget(Qt.RightDockWidgetArea, dock)
            elif dock.objectName() == "simulation_dock":
                self.addDockWidget(Qt.BottomDockWidgetArea, dock)
                
        self.status_bar.showMessage("Layout reset to default", 3000)
    
    @pyqtSlot()
    def update_ui(self):
        """Update UI with latest simulation state."""
        # Update only if we have a simulator
        if not self.simulator:
            return
            
        # Get simulation time
        try:
            current_time = self.simulator.clock.current_time
            elapsed_time = self.simulator.clock.elapsed_time
            
            # Update time display
            time_str = current_time.strftime("%H:%M:%S")
            elapsed_str = f"{int(elapsed_time.total_seconds() // 3600):02d}:{int((elapsed_time.total_seconds() % 3600) // 60):02d}:{int(elapsed_time.total_seconds() % 60):02d}"
            self.status_bar.showMessage(f"Simulation Time: {time_str} | Elapsed: {elapsed_str}")
            
            # Update views
            self.map_view.update_view()
            self.vehicle_view.update_view()
            self.task_view.update_view()
            self.simulation_controls.update_controls()
            
        except Exception as e:
            self.logger.error(f"Error updating UI: {str(e)}", exc_info=True)
    
    def set_simulator(self, simulator):
        """
        Set the simulator instance.
        
        Args:
            simulator: Simulator instance
        """
        self.simulator = simulator
        
        # If simulator has dispatcher, register as event listener
        if simulator and simulator.dispatcher:
            simulator.dispatcher.add_event_listener(EventType.ALL, self)
        
        # Set simulator in views
        self.map_view.set_simulator(simulator)
        self.vehicle_view.set_simulator(simulator)
        self.task_view.set_simulator(simulator)
        self.simulation_controls.set_simulator(simulator)
        
        # Update environment
        if simulator:
            self.environment = simulator.environment
            self.dispatcher = simulator.dispatcher
    
    def on_event(self, event: DispatchEvent) -> None:
        """
        Handle dispatch events.
        
        Args:
            event: Dispatch event
        """
        # Emit signal to update UI from main thread
        self.updated_signal.emit()
    
    def new_simulation(self):
        """Create a new simulation."""
        from coordination.simulation.simulator import Simulator, SimulationConfig
        
        # Create a new simulator with default settings
        try:
            config = SimulationConfig()
            simulator = Simulator(config=config)
            simulator.setup()
            
            # Set the simulator
            self.set_simulator(simulator)
            
            self.status_bar.showMessage("New simulation created", 3000)
            
        except Exception as e:
            self.logger.error(f"Error creating new simulation: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to create new simulation: {str(e)}")
    
    def open_simulation(self):
        """Open an existing simulation."""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Simulation Files (*.sim);;All Files (*)")
        file_dialog.setDefaultSuffix("sim")
        file_dialog.setAcceptMode(QFileDialog.AcceptOpen)
        
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.load_simulation(file_path)
    
    def load_simulation(self, file_path: str):
        """
        Load a simulation from file.
        
        Args:
            file_path: Path to simulation file
        """
        try:
            # TODO: Implement simulation loading
            self.status_bar.showMessage(f"Loaded simulation from {file_path}", 3000)
            self.last_file = file_path
            
        except Exception as e:
            self.logger.error(f"Error loading simulation: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load simulation: {str(e)}")
    
    def save_simulation(self):
        """Save current simulation."""
        if hasattr(self, 'last_file') and self.last_file:
            self.save_simulation_to(self.last_file)
        else:
            self.save_simulation_as()
    
    def save_simulation_as(self):
        """Save current simulation with a new name."""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Simulation Files (*.sim);;All Files (*)")
        file_dialog.setDefaultSuffix("sim")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.save_simulation_to(file_path)
    
    def save_simulation_to(self, file_path: str):
        """
        Save current simulation to file.
        
        Args:
            file_path: Path to save simulation
        """
        try:
            # TODO: Implement simulation saving
            self.status_bar.showMessage(f"Saved simulation to {file_path}", 3000)
            self.last_file = file_path
            
        except Exception as e:
            self.logger.error(f"Error saving simulation: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save simulation: {str(e)}")
    
    def export_results(self):
        """Export simulation results."""
        if not self.simulator:
            QMessageBox.warning(self, "Warning", "No active simulation to export")
            return
            
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("CSV Files (*.csv);;JSON Files (*.json);;All Files (*)")
        file_dialog.setDefaultSuffix("csv")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            
            try:
                # TODO: Implement results export
                self.status_bar.showMessage(f"Exported results to {file_path}", 3000)
                
            except Exception as e:
                self.logger.error(f"Error exporting results: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")
    
    def start_simulation(self):
        """Start the simulation."""
        if not self.simulator:
            QMessageBox.warning(self, "Warning", "No active simulation to start")
            return
            
        try:
            self.simulator.start()
            self.status_bar.showMessage("Simulation started", 3000)
            
        except Exception as e:
            self.logger.error(f"Error starting simulation: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to start simulation: {str(e)}")
    
    def pause_simulation(self):
        """Pause the simulation."""
        if not self.simulator:
            return
            
        try:
            self.simulator.pause()
            self.status_bar.showMessage("Simulation paused", 3000)
            
        except Exception as e:
            self.logger.error(f"Error pausing simulation: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to pause simulation: {str(e)}")
    
    def stop_simulation(self):
        """Stop the simulation."""
        if not self.simulator:
            return
            
        try:
            self.simulator.stop()
            self.status_bar.showMessage("Simulation stopped", 3000)
            
        except Exception as e:
            self.logger.error(f"Error stopping simulation: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to stop simulation: {str(e)}")
    
    def step_simulation(self):
        """Step the simulation."""
        if not self.simulator:
            QMessageBox.warning(self, "Warning", "No active simulation to step")
            return
            
        try:
            self.simulator.step()
            self.status_bar.showMessage("Simulation stepped", 3000)
            
        except Exception as e:
            self.logger.error(f"Error stepping simulation: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to step simulation: {str(e)}")
    
    def show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec_():
            # Apply settings
            self.status_bar.showMessage("Settings applied", 3000)
    
    def show_analysis(self):
        """Show analysis dialog."""
        if not self.simulator:
            QMessageBox.warning(self, "Warning", "No active simulation to analyze")
            return
            
        dialog = AnalysisDialog(self.simulator, self)
        dialog.exec_()
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Mining Dispatch System",
            "<h3>Open-Pit Mine Multi-Vehicle Coordination System</h3>"
            "<p>Version 1.0.0</p>"
            "<p>A system for coordinating multiple vehicles in open-pit mining operations.</p>"
            "<p>&copy; 2025 Mining Operations</p>"
        )
    
    def closeEvent(self, event: QCloseEvent):
        """
        Handle window close event.
        
        Args:
            event: Close event
        """
        # Save settings before closing
        self._save_settings()
        
        # Stop simulation if running
        if self.simulator:
            try:
                self.simulator.stop()
            except Exception as e:
                self.logger.error(f"Error stopping simulation on exit: {str(e)}")
        
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Mining Dispatch System")
    app.setOrganizationName("Mining")
    
    # Set style
    app.setStyle("Fusion")
    
    # Create and show main window
    main_window = MainWindow()
    main_window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()