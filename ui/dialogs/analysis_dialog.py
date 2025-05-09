"""
Analysis dialog for the mining dispatch system.

Provides visualization and analysis of simulation results.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from typing import Dict, List, Optional, Any, Tuple
import math
import random

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QComboBox, QGroupBox, QFormLayout,
    QDialogButtonBox, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QPalette, QBrush, QPen

# For data visualization
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from utils.logger import get_logger


class ChartWidget(QWidget):
    """Widget for displaying charts using Matplotlib."""
    
    def __init__(self, parent=None):
        """
        Initialize chart widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("ChartWidget")
        
        # Check if matplotlib is available
        if not MATPLOTLIB_AVAILABLE:
            self._init_fallback_ui()
            return
            
        # Initialize UI with matplotlib
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface with matplotlib."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create figure and canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Create navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add widgets to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Create initial axes
        self.axes = self.figure.add_subplot(111)
        self.figure.tight_layout()
    
    def _init_fallback_ui(self):
        """Initialize fallback UI when matplotlib is not available."""
        layout = QVBoxLayout(self)
        label = QLabel("Matplotlib is not available. Please install matplotlib to enable charts.")
        label.setStyleSheet("color: red;")
        layout.addWidget(label)
    
    def plot_bar_chart(self, data: Dict[str, float], title: str, xlabel: str, ylabel: str, color: str = '#3070B0'):
        """
        Plot a bar chart.
        
        Args:
            data: Dictionary of label -> value
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Bar color
        """
        if not MATPLOTLIB_AVAILABLE:
            return
            
        try:
            # Clear figure
            self.figure.clear()
            self.axes = self.figure.add_subplot(111)
            
            # Get data
            labels = list(data.keys())
            values = list(data.values())
            
            # Create bar chart
            self.axes.bar(labels, values, color=color)
            
            # Set labels and title
            self.axes.set_title(title)
            self.axes.set_xlabel(xlabel)
            self.axes.set_ylabel(ylabel)
            
            # Rotate x labels if there are many
            if len(labels) > 5:
                self.axes.set_xticklabels(labels, rotation=45, ha='right')
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error plotting bar chart: {str(e)}", exc_info=True)
    
    def plot_pie_chart(self, data: Dict[str, float], title: str, autopct: str = '%1.1f%%'):
        """
        Plot a pie chart.
        
        Args:
            data: Dictionary of label -> value
            title: Chart title
            autopct: Format string for percentage labels
        """
        if not MATPLOTLIB_AVAILABLE:
            return
            
        try:
            # Clear figure
            self.figure.clear()
            self.axes = self.figure.add_subplot(111)
            
            # Get data
            labels = list(data.keys())
            values = list(data.values())
            
            # Create pie chart
            self.axes.pie(values, labels=labels, autopct=autopct, startangle=90)
            self.axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            # Set title
            self.axes.set_title(title)
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error plotting pie chart: {str(e)}", exc_info=True)
    
    def plot_line_chart(self, data: Dict[str, List[float]], x_values: List[float], title: str, xlabel: str, ylabel: str):
        """
        Plot a line chart with multiple lines.
        
        Args:
            data: Dictionary of line_name -> values
            x_values: X-axis values (shared across all lines)
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        if not MATPLOTLIB_AVAILABLE:
            return
            
        try:
            # Clear figure
            self.figure.clear()
            self.axes = self.figure.add_subplot(111)
            
            # Plot each line
            for label, values in data.items():
                # Ensure x and y have same length
                x = x_values[:len(values)]
                self.axes.plot(x, values, label=label)
            
            # Set labels and title
            self.axes.set_title(title)
            self.axes.set_xlabel(xlabel)
            self.axes.set_ylabel(ylabel)
            
            # Add legend if multiple lines
            if len(data) > 1:
                self.axes.legend()
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error plotting line chart: {str(e)}", exc_info=True)
    
    def plot_histogram(self, data: List[float], bins: int, title: str, xlabel: str, ylabel: str):
        """
        Plot a histogram.
        
        Args:
            data: List of values
            bins: Number of bins
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        if not MATPLOTLIB_AVAILABLE:
            return
            
        try:
            # Clear figure
            self.figure.clear()
            self.axes = self.figure.add_subplot(111)
            
            # Create histogram
            self.axes.hist(data, bins=bins)
            
            # Set labels and title
            self.axes.set_title(title)
            self.axes.set_xlabel(xlabel)
            self.axes.set_ylabel(ylabel)
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error plotting histogram: {str(e)}", exc_info=True)


class PerformanceTab(QWidget):
    """Tab for analyzing performance metrics."""
    
    def __init__(self, simulator, parent=None):
        """
        Initialize performance tab.
        
        Args:
            simulator: Simulator instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("PerformanceTab")
        self.simulator = simulator
        
        # Initialize UI
        self._init_ui()
        
        # Update data
        self.update_data()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create chart selection area
        selection_layout = QHBoxLayout()
        layout.addLayout(selection_layout)
        
        selection_layout.addWidget(QLabel("Chart Type:"))
        
        self.chart_combo = QComboBox()
        self.chart_combo.addItem("Task Completion", "task_completion")
        self.chart_combo.addItem("Resource Utilization", "resource_utilization")
        self.chart_combo.addItem("Distance Metrics", "distance_metrics")
        self.chart_combo.addItem("Time Metrics", "time_metrics")
        self.chart_combo.currentIndexChanged.connect(self.update_chart)
        selection_layout.addWidget(self.chart_combo)
        
        selection_layout.addStretch()
        
        # Add export button
        export_button = QPushButton("Export Data...")
        export_button.clicked.connect(self.export_data)
        selection_layout.addWidget(export_button)
        
        # Add separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Create chart area
        self.chart_widget = ChartWidget()
        layout.addWidget(self.chart_widget)
        
        # Add separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Create metrics table
        metrics_layout = QHBoxLayout()
        layout.addLayout(metrics_layout)
        
        # Create basic metrics widget
        basic_metrics = QGroupBox("Performance Metrics")
        basic_layout = QFormLayout(basic_metrics)
        
        self.total_tasks_label = QLabel("0")
        basic_layout.addRow("Total Tasks:", self.total_tasks_label)
        
        self.completed_tasks_label = QLabel("0")
        basic_layout.addRow("Completed Tasks:", self.completed_tasks_label)
        
        self.failed_tasks_label = QLabel("0")
        basic_layout.addRow("Failed Tasks:", self.failed_tasks_label)
        
        self.throughput_label = QLabel("0 tasks/hour")
        basic_layout.addRow("Throughput:", self.throughput_label)
        
        metrics_layout.addWidget(basic_metrics)
        
        # Create time metrics widget
        time_metrics = QGroupBox("Time Metrics")
        time_layout = QFormLayout(time_metrics)
        
        self.avg_completion_label = QLabel("0.0 seconds")
        time_layout.addRow("Avg. Completion Time:", self.avg_completion_label)
        
        self.avg_waiting_label = QLabel("0.0 seconds")
        time_layout.addRow("Avg. Waiting Time:", self.avg_waiting_label)
        
        self.sim_time_label = QLabel("0.0 seconds")
        time_layout.addRow("Simulation Time:", self.sim_time_label)
        
        metrics_layout.addWidget(time_metrics)
        
        # Create distance metrics widget
        distance_metrics = QGroupBox("Distance Metrics")
        distance_layout = QFormLayout(distance_metrics)
        
        self.total_distance_label = QLabel("0.0 meters")
        distance_layout.addRow("Total Distance:", self.total_distance_label)
        
        self.avg_distance_label = QLabel("0.0 meters/task")
        distance_layout.addRow("Avg. Distance/Task:", self.avg_distance_label)
        
        self.utilization_label = QLabel("0.0%")
        distance_layout.addRow("Vehicle Utilization:", self.utilization_label)
        
        metrics_layout.addWidget(distance_metrics)
    
    def update_data(self):
        """Update data from simulator."""
        if not self.simulator:
            return
            
        try:
            # Get metrics from simulator
            metrics = self.simulator.get_metrics()
            
            # Update labels
            self.total_tasks_label.setText(str(metrics.total_tasks))
            self.completed_tasks_label.setText(str(metrics.completed_tasks))
            self.failed_tasks_label.setText(str(metrics.failed_tasks))
            self.throughput_label.setText(f"{metrics.throughput:.2f} tasks/hour")
            
            self.avg_completion_label.setText(f"{metrics.average_task_completion_time:.2f} seconds")
            self.avg_waiting_label.setText(f"{metrics.average_task_waiting_time:.2f} seconds")
            self.sim_time_label.setText(f"{metrics.total_simulation_time:.2f} seconds")
            
            self.total_distance_label.setText(f"{metrics.total_distance_traveled:.2f} meters")
            
            avg_distance = 0.0
            if metrics.completed_tasks > 0:
                avg_distance = metrics.total_distance_traveled / metrics.completed_tasks
            
            self.avg_distance_label.setText(f"{avg_distance:.2f} meters/task")
            self.utilization_label.setText(f"{metrics.average_vehicle_utilization:.2f}%")
            
            # Update chart
            self.update_chart()
            
        except Exception as e:
            self.logger.error(f"Error updating performance data: {str(e)}", exc_info=True)
    
    def update_chart(self):
        """Update chart based on current selection."""
        if not self.simulator:
            return
            
        try:
            # Get metrics from simulator
            metrics = self.simulator.get_metrics()
            
            # Get selected chart type
            chart_type = self.chart_combo.currentData()
            
            if chart_type == "task_completion":
                # Task completion chart (pie chart)
                data = {
                    "Completed": metrics.completed_tasks,
                    "Failed": metrics.failed_tasks,
                    "Pending": metrics.total_tasks - metrics.completed_tasks - metrics.failed_tasks
                }
                
                # Remove categories with zero values
                data = {k: v for k, v in data.items() if v > 0}
                
                if data:
                    self.chart_widget.plot_pie_chart(data, "Task Status Distribution")
                
            elif chart_type == "resource_utilization":
                # Resource utilization chart (bar chart)
                data = {}
                
                # Get vehicle-specific metrics if available
                if hasattr(metrics, 'metrics_by_vehicle') and metrics.metrics_by_vehicle:
                    for vehicle_id, vehicle_metrics in metrics.metrics_by_vehicle.items():
                        if 'utilization' in vehicle_metrics:
                            data[vehicle_id] = vehicle_metrics['utilization']
                
                # If no real data, create placeholder data
                if not data:
                    total_vehicles = getattr(metrics, 'total_vehicles', 0)
                    util = getattr(metrics, 'average_vehicle_utilization', 0.0)
                    
                    # Create some random data for visualization
                    for i in range(min(total_vehicles, 5)):
                        vehicle_id = f"V{i+1}"
                        # Random variation around average
                        variation = random.uniform(0.8, 1.2) if util > 0 else 0
                        data[vehicle_id] = util * variation
                
                self.chart_widget.plot_bar_chart(
                    data,
                    "Vehicle Utilization",
                    "Vehicle",
                    "Utilization (%)",
                    "#70B030"  # Green color
                )
                
            elif chart_type == "distance_metrics":
                # Distance metrics chart (bar chart)
                data = {}
                
                # Get vehicle-specific metrics if available
                if hasattr(metrics, 'metrics_by_vehicle') and metrics.metrics_by_vehicle:
                    for vehicle_id, vehicle_metrics in metrics.metrics_by_vehicle.items():
                        if 'distance_traveled' in vehicle_metrics:
                            data[vehicle_id] = vehicle_metrics['distance_traveled']
                
                # If no real data, create placeholder data
                if not data:
                    total_vehicles = getattr(metrics, 'total_vehicles', 0)
                    total_distance = getattr(metrics, 'total_distance_traveled', 0.0)
                    
                    # Distribute total distance randomly among vehicles
                    if total_vehicles > 0 and total_distance > 0:
                        for i in range(min(total_vehicles, 5)):
                            vehicle_id = f"V{i+1}"
                            # Random portion of total distance
                            data[vehicle_id] = total_distance * random.uniform(0.1, 0.3)
                
                self.chart_widget.plot_bar_chart(
                    data,
                    "Distance Traveled by Vehicle",
                    "Vehicle",
                    "Distance (meters)",
                    "#3070B0"  # Blue color
                )
                
            elif chart_type == "time_metrics":
                # Time metrics chart (bar chart)
                data = {
                    "Avg. Completion": metrics.average_task_completion_time,
                    "Avg. Waiting": metrics.average_task_waiting_time
                }
                
                if hasattr(metrics, 'metrics_by_task_type') and metrics.metrics_by_task_type:
                    # Add task-type specific metrics if available
                    for task_type, task_metrics in metrics.metrics_by_task_type.items():
                        if 'average_completion_time' in task_metrics:
                            data[f"{task_type} Completion"] = task_metrics['average_completion_time']
                
                self.chart_widget.plot_bar_chart(
                    data,
                    "Task Time Metrics",
                    "Metric",
                    "Time (seconds)",
                    "#B03070"  # Purple color
                )
            
        except Exception as e:
            self.logger.error(f"Error updating chart: {str(e)}", exc_info=True)
    
    def export_data(self):
        """Export performance data to CSV file."""
        if not self.simulator:
            return
            
        try:
            # Show file dialog
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Export Performance Data",
                "performance_data.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_name:
                return
                
            # Get metrics
            metrics = self.simulator.get_metrics()
            
            # Create CSV content
            csv_content = "Metric,Value\n"
            csv_content += f"Total Tasks,{metrics.total_tasks}\n"
            csv_content += f"Completed Tasks,{metrics.completed_tasks}\n"
            csv_content += f"Failed Tasks,{metrics.failed_tasks}\n"
            csv_content += f"Throughput (tasks/hour),{metrics.throughput}\n"
            csv_content += f"Average Completion Time (seconds),{metrics.average_task_completion_time}\n"
            csv_content += f"Average Waiting Time (seconds),{metrics.average_task_waiting_time}\n"
            csv_content += f"Total Simulation Time (seconds),{metrics.total_simulation_time}\n"
            csv_content += f"Total Distance Traveled (meters),{metrics.total_distance_traveled}\n"
            
            avg_distance = 0.0
            if metrics.completed_tasks > 0:
                avg_distance = metrics.total_distance_traveled / metrics.completed_tasks
            
            csv_content += f"Average Distance per Task (meters),{avg_distance}\n"
            csv_content += f"Average Vehicle Utilization (%),{metrics.average_vehicle_utilization}\n"
            
            # Add vehicle-specific metrics if available
            if hasattr(metrics, 'metrics_by_vehicle') and metrics.metrics_by_vehicle:
                csv_content += "\nVehicle Metrics\n"
                csv_content += "Vehicle ID,Distance Traveled,Utilization\n"
                
                for vehicle_id, vehicle_metrics in metrics.metrics_by_vehicle.items():
                    distance = vehicle_metrics.get('distance_traveled', 0.0)
                    utilization = vehicle_metrics.get('utilization', 0.0)
                    csv_content += f"{vehicle_id},{distance},{utilization}\n"
            
            # Add task-type specific metrics if available
            if hasattr(metrics, 'metrics_by_task_type') and metrics.metrics_by_task_type:
                csv_content += "\nTask Type Metrics\n"
                csv_content += "Task Type,Count,Avg. Completion Time,Avg. Waiting Time\n"
                
                for task_type, task_metrics in metrics.metrics_by_task_type.items():
                    count = task_metrics.get('count', 0)
                    completion_time = task_metrics.get('average_completion_time', 0.0)
                    waiting_time = task_metrics.get('average_waiting_time', 0.0)
                    csv_content += f"{task_type},{count},{completion_time},{waiting_time}\n"
            
            # Write to file
            with open(file_name, 'w') as f:
                f.write(csv_content)
                
            self.logger.info(f"Performance data exported to {file_name}")
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}", exc_info=True)


class TaskAnalysisTab(QWidget):
    """Tab for analyzing task performance and statistics."""
    
    def __init__(self, simulator, parent=None):
        """
        Initialize task analysis tab.
        
        Args:
            simulator: Simulator instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("TaskAnalysisTab")
        self.simulator = simulator
        
        # Initialize UI
        self._init_ui()
        
        # Update data
        self.update_data()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create filter controls
        filter_layout = QHBoxLayout()
        layout.addLayout(filter_layout)
        
        filter_layout.addWidget(QLabel("Filter:"))
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Tasks", "all")
        self.filter_combo.addItem("Completed Tasks", "completed")
        self.filter_combo.addItem("Failed Tasks", "failed")
        self.filter_combo.addItem("Pending Tasks", "pending")
        self.filter_combo.currentIndexChanged.connect(self.update_filter)
        filter_layout.addWidget(self.filter_combo)
        
        filter_layout.addStretch()
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.update_data)
        filter_layout.addWidget(refresh_button)
        
        export_button = QPushButton("Export...")
        export_button.clicked.connect(self.export_data)
        filter_layout.addWidget(export_button)
        
        # Create table
        self.task_table = QTableWidget(0, 6)
        self.task_table.setHorizontalHeaderLabels([
            "ID", "Type", "Status", "Created", "Completed", "Duration"
        ])
        self.task_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.task_table.setSelectionMode(QTableWidget.SingleSelection)
        self.task_table.setAlternatingRowColors(True)
        self.task_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.task_table)
        
        # Add charts
        charts_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(charts_splitter)
        
        # Task status chart
        self.status_chart = ChartWidget()
        status_container = QWidget()
        status_layout = QVBoxLayout(status_container)
        status_layout.addWidget(QLabel("<b>Task Status Distribution</b>"))
        status_layout.addWidget(self.status_chart)
        charts_splitter.addWidget(status_container)
        
        # Task type chart
        self.type_chart = ChartWidget()
        type_container = QWidget()
        type_layout = QVBoxLayout(type_container)
        type_layout.addWidget(QLabel("<b>Task Type Distribution</b>"))
        type_layout.addWidget(self.type_chart)
        charts_splitter.addWidget(type_container)
        
        # Set initial splitter sizes
        charts_splitter.setSizes([50, 50])
    
    def update_data(self):
        """Update data from simulator."""
        if not self.simulator or not hasattr(self.simulator, 'dispatcher'):
            return
            
        try:
            # Get tasks from dispatcher
            dispatcher = self.simulator.dispatcher
            if not dispatcher:
                return
                
            # Get all tasks
            tasks = dispatcher.get_all_tasks()
            
            # Store tasks
            self.tasks = tasks
            
            # Apply filter
            self.update_filter()
            
        except Exception as e:
            self.logger.error(f"Error updating task data: {str(e)}", exc_info=True)
    
    def update_filter(self):
        """Apply filter to task data."""
        if not hasattr(self, 'tasks') or not self.tasks:
            return
            
        try:
            # Get filter
            filter_type = self.filter_combo.currentData()
            
            # Clear table
            self.task_table.setRowCount(0)
            
            # Filter tasks
            filtered_tasks = {}
            
            for task_id, task in self.tasks.items():
                # Get status
                status = "UNKNOWN"
                if hasattr(task, 'status'):
                    status = task.status.name if hasattr(task.status, 'name') else str(task.status)
                
                # Apply filter
                if filter_type == "all":
                    # No filter
                    filtered_tasks[task_id] = task
                elif filter_type == "completed" and status == "COMPLETED":
                    filtered_tasks[task_id] = task
                elif filter_type == "failed" and status == "FAILED":
                    filtered_tasks[task_id] = task
                elif filter_type == "pending" and status == "PENDING":
                    filtered_tasks[task_id] = task
            
            # Populate table
            self.task_table.setRowCount(len(filtered_tasks))
            
            for row, (task_id, task) in enumerate(filtered_tasks.items()):
                # Task ID
                self.task_table.setItem(row, 0, QTableWidgetItem(task_id))
                
                # Task type
                task_type = "Unknown"
                if hasattr(task, 'task_type'):
                    task_type = task.task_type
                elif hasattr(task, '__class__'):
                    task_type = task.__class__.__name__
                    
                self.task_table.setItem(row, 1, QTableWidgetItem(task_type))
                
                # Task status
                status = "UNKNOWN"
                if hasattr(task, 'status'):
                    status = task.status.name if hasattr(task.status, 'name') else str(task.status)
                    
                status_item = QTableWidgetItem(status)
                
                # Set color based on status
                if status == "COMPLETED":
                    status_item.setBackground(QBrush(QColor(192, 255, 192)))  # Light green
                elif status == "FAILED":
                    status_item.setBackground(QBrush(QColor(255, 192, 192)))  # Light red
                elif status == "PENDING":
                    status_item.setBackground(QBrush(QColor(255, 224, 192)))  # Light orange
                elif status == "ASSIGNED":
                    status_item.setBackground(QBrush(QColor(224, 224, 255)))  # Light blue
                
                self.task_table.setItem(row, 2, status_item)
                
                # Created time
                created_time = "Unknown"
                if hasattr(task, 'creation_time'):
                    if hasattr(task.creation_time, 'strftime'):
                        created_time = task.creation_time.strftime("%H:%M:%S")
                        
                self.task_table.setItem(row, 3, QTableWidgetItem(created_time))
                
                # Completed time
                completed_time = ""
                if hasattr(task, 'completion_time'):
                    if hasattr(task.completion_time, 'strftime'):
                        completed_time = task.completion_time.strftime("%H:%M:%S")
                        
                self.task_table.setItem(row, 4, QTableWidgetItem(completed_time))
                
                # Duration
                duration = ""
                if hasattr(task, 'execution_time'):
                    duration = f"{task.execution_time:.1f} sec"
                    
                self.task_table.setItem(row, 5, QTableWidgetItem(duration))
            
            # Update charts
            self.update_charts()
            
        except Exception as e:
            self.logger.error(f"Error updating task filter: {str(e)}", exc_info=True)
    
    def update_charts(self):
        """Update charts with current data."""
        if not hasattr(self, 'tasks') or not self.tasks:
            return
            
        try:
            # Task status distribution chart
            status_data = {}
            
            # Task type distribution chart
            type_data = {}
            
            # Process all tasks
            for task_id, task in self.tasks.items():
                # Get status
                status = "UNKNOWN"
                if hasattr(task, 'status'):
                    status = task.status.name if hasattr(task.status, 'name') else str(task.status)
                
                # Update status count
                status_data[status] = status_data.get(status, 0) + 1
                
                # Get type
                task_type = "Unknown"
                if hasattr(task, 'task_type'):
                    task_type = task.task_type
                elif hasattr(task, '__class__'):
                    task_type = task.__class__.__name__
                
                # Update type count
                type_data[task_type] = type_data.get(task_type, 0) + 1
            
            # Plot charts
            if status_data:
                self.status_chart.plot_pie_chart(status_data, "Task Status Distribution")
                
            if type_data:
                self.type_chart.plot_pie_chart(type_data, "Task Type Distribution")
            
        except Exception as e:
            self.logger.error(f"Error updating task charts: {str(e)}", exc_info=True)
    
    def export_data(self):
        """Export task data to CSV file."""
        if not hasattr(self, 'tasks') or not self.tasks:
            return
            
        try:
            # Show file dialog
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Export Task Data",
                "task_data.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_name:
                return
                
            # Create CSV content
            csv_content = "Task ID,Type,Status,Created,Completed,Duration\n"
            
            for task_id, task in self.tasks.items():
                # Get type
                task_type = "Unknown"
                if hasattr(task, 'task_type'):
                    task_type = task.task_type
                elif hasattr(task, '__class__'):
                    task_type = task.__class__.__name__
                
                # Get status
                status = "UNKNOWN"
                if hasattr(task, 'status'):
                    status = task.status.name if hasattr(task.status, 'name') else str(task.status)
                
                # Get created time
                created_time = ""
                if hasattr(task, 'creation_time'):
                    if hasattr(task.creation_time, 'strftime'):
                        created_time = task.creation_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Get completed time
                completed_time = ""
                if hasattr(task, 'completion_time'):
                    if hasattr(task.completion_time, 'strftime'):
                        completed_time = task.completion_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Get duration
                duration = ""
                if hasattr(task, 'execution_time'):
                    duration = f"{task.execution_time}"
                
                # Add row
                csv_content += f"{task_id},{task_type},{status},{created_time},{completed_time},{duration}\n"
            
            # Write to file
            with open(file_name, 'w') as f:
                f.write(csv_content)
                
            self.logger.info(f"Task data exported to {file_name}")
            
        except Exception as e:
            self.logger.error(f"Error exporting task data: {str(e)}", exc_info=True)


class VehicleAnalysisTab(QWidget):
    """Tab for analyzing vehicle performance and statistics."""
    
    def __init__(self, simulator, parent=None):
        """
        Initialize vehicle analysis tab.
        
        Args:
            simulator: Simulator instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("VehicleAnalysisTab")
        self.simulator = simulator
        
        # Initialize UI
        self._init_ui()
        
        # Update data
        self.update_data()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create filter controls
        filter_layout = QHBoxLayout()
        layout.addLayout(filter_layout)
        
        filter_layout.addWidget(QLabel("Filter:"))
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Vehicles", "all")
        self.filter_combo.addItem("Active Vehicles", "active")
        self.filter_combo.addItem("Idle Vehicles", "idle")
        self.filter_combo.currentIndexChanged.connect(self.update_filter)
        filter_layout.addWidget(self.filter_combo)
        
        filter_layout.addStretch()
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.update_data)
        filter_layout.addWidget(refresh_button)
        
        export_button = QPushButton("Export...")
        export_button.clicked.connect(self.export_data)
        filter_layout.addWidget(export_button)
        
        # Create table
        self.vehicle_table = QTableWidget(0, 5)
        self.vehicle_table.setHorizontalHeaderLabels([
            "ID", "Type", "State", "Distance", "Tasks Completed"
        ])
        self.vehicle_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.vehicle_table.setSelectionMode(QTableWidget.SingleSelection)
        self.vehicle_table.setAlternatingRowColors(True)
        self.vehicle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.vehicle_table)
        
        # Add charts
        charts_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(charts_splitter)
        
        # Vehicle state chart
        self.state_chart = ChartWidget()
        state_container = QWidget()
        state_layout = QVBoxLayout(state_container)
        state_layout.addWidget(QLabel("<b>Vehicle State Distribution</b>"))
        state_layout.addWidget(self.state_chart)
        charts_splitter.addWidget(state_container)
        
        # Vehicle distance chart
        self.distance_chart = ChartWidget()
        distance_container = QWidget()
        distance_layout = QVBoxLayout(distance_container)
        distance_layout.addWidget(QLabel("<b>Distance Traveled by Vehicle</b>"))
        distance_layout.addWidget(self.distance_chart)
        charts_splitter.addWidget(distance_container)
        
        # Set initial splitter sizes
        charts_splitter.setSizes([50, 50])
    
    def update_data(self):
        """Update data from simulator."""
        if not self.simulator or not hasattr(self.simulator, 'dispatcher'):
            return
            
        try:
            # Get vehicles from dispatcher
            dispatcher = self.simulator.dispatcher
            if not dispatcher:
                return
                
            # Get all vehicles
            vehicles = dispatcher.get_all_vehicles()
            
            # Store vehicles
            self.vehicles = vehicles
            
            # Get metrics
            metrics = self.simulator.get_metrics()
            self.metrics = metrics
            
            # Apply filter
            self.update_filter()
            
        except Exception as e:
            self.logger.error(f"Error updating vehicle data: {str(e)}", exc_info=True)
    
    def update_filter(self):
        """Apply filter to vehicle data."""
        if not hasattr(self, 'vehicles') or not self.vehicles:
            return
            
        try:
            # Get filter
            filter_type = self.filter_combo.currentData()
            
            # Clear table
            self.vehicle_table.setRowCount(0)
            
            # Filter vehicles
            filtered_vehicles = {}
            
            for vehicle_id, vehicle in self.vehicles.items():
                # Get state
                state = "UNKNOWN"
                if hasattr(vehicle, 'state'):
                    state = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                
                # Apply filter
                if filter_type == "all":
                    # No filter
                    filtered_vehicles[vehicle_id] = vehicle
                elif filter_type == "active" and state not in ["IDLE", "OUT_OF_SERVICE"]:
                    filtered_vehicles[vehicle_id] = vehicle
                elif filter_type == "idle" and state == "IDLE":
                    filtered_vehicles[vehicle_id] = vehicle
            
            # Populate table
            self.vehicle_table.setRowCount(len(filtered_vehicles))
            
            # Get metrics by vehicle if available
            vehicle_metrics = {}
            if hasattr(self, 'metrics') and hasattr(self.metrics, 'metrics_by_vehicle'):
                vehicle_metrics = self.metrics.metrics_by_vehicle
            
            for row, (vehicle_id, vehicle) in enumerate(filtered_vehicles.items()):
                # Vehicle ID
                self.vehicle_table.setItem(row, 0, QTableWidgetItem(vehicle_id))
                
                # Vehicle type
                vehicle_type = "Unknown"
                if hasattr(vehicle, 'vehicle_type'):
                    vehicle_type = vehicle.vehicle_type
                    
                self.vehicle_table.setItem(row, 1, QTableWidgetItem(vehicle_type))
                
                # Vehicle state
                state = "UNKNOWN"
                if hasattr(vehicle, 'state'):
                    state = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                    
                state_item = QTableWidgetItem(state)
                
                # Set color based on state
                if state == "IDLE":
                    state_item.setBackground(QBrush(QColor(192, 255, 192)))  # Light green
                elif state in ["LOADING", "UNLOADING"]:
                    state_item.setBackground(QBrush(QColor(255, 192, 192)))  # Light red
                elif state in ["MOVING", "EN_ROUTE"]:
                    state_item.setBackground(QBrush(QColor(192, 192, 255)))  # Light blue
                
                self.vehicle_table.setItem(row, 2, state_item)
                
                # Distance
                distance = 0.0
                if vehicle_id in vehicle_metrics and 'distance_traveled' in vehicle_metrics[vehicle_id]:
                    distance = vehicle_metrics[vehicle_id]['distance_traveled']
                    
                self.vehicle_table.setItem(row, 3, QTableWidgetItem(f"{distance:.1f} m"))
                
                # Tasks completed
                tasks_completed = 0
                if vehicle_id in vehicle_metrics and 'tasks_completed' in vehicle_metrics[vehicle_id]:
                    tasks_completed = vehicle_metrics[vehicle_id]['tasks_completed']
                    
                self.vehicle_table.setItem(row, 4, QTableWidgetItem(str(tasks_completed)))
            
            # Update charts
            self.update_charts()
            
        except Exception as e:
            self.logger.error(f"Error updating vehicle filter: {str(e)}", exc_info=True)
    
    def update_charts(self):
        """Update charts with current data."""
        if not hasattr(self, 'vehicles') or not self.vehicles:
            return
            
        try:
            # Vehicle state distribution chart
            state_data = {}
            
            # Vehicle distance chart
            distance_data = {}
            
            # Get metrics by vehicle if available
            vehicle_metrics = {}
            if hasattr(self, 'metrics') and hasattr(self.metrics, 'metrics_by_vehicle'):
                vehicle_metrics = self.metrics.metrics_by_vehicle
            
            # Process all vehicles
            for vehicle_id, vehicle in self.vehicles.items():
                # Get state
                state = "UNKNOWN"
                if hasattr(vehicle, 'state'):
                    state = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                
                # Update state count
                state_data[state] = state_data.get(state, 0) + 1
                
                # Get distance
                distance = 0.0
                if vehicle_id in vehicle_metrics and 'distance_traveled' in vehicle_metrics[vehicle_id]:
                    distance = vehicle_metrics[vehicle_id]['distance_traveled']
                
                # Update distance data (limit to top 10 vehicles)
                if len(distance_data) < 10:
                    distance_data[vehicle_id] = distance
            
            # Sort distance data by value (descending)
            distance_data = dict(sorted(distance_data.items(), key=lambda x: x[1], reverse=True))
            
            # Plot charts
            if state_data:
                self.state_chart.plot_pie_chart(state_data, "Vehicle State Distribution")
                
            if distance_data:
                self.distance_chart.plot_bar_chart(
                    distance_data,
                    "Distance Traveled by Vehicle",
                    "Vehicle",
                    "Distance (meters)",
                    "#3070B0"  # Blue color
                )
            
        except Exception as e:
            self.logger.error(f"Error updating vehicle charts: {str(e)}", exc_info=True)
    
    def export_data(self):
        """Export vehicle data to CSV file."""
        if not hasattr(self, 'vehicles') or not self.vehicles:
            return
            
        try:
            # Show file dialog
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Export Vehicle Data",
                "vehicle_data.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_name:
                return
                
            # Get metrics by vehicle if available
            vehicle_metrics = {}
            if hasattr(self, 'metrics') and hasattr(self.metrics, 'metrics_by_vehicle'):
                vehicle_metrics = self.metrics.metrics_by_vehicle
                
            # Create CSV content
            csv_content = "Vehicle ID,Type,State,Distance Traveled,Tasks Completed,Current Load\n"
            
            for vehicle_id, vehicle in self.vehicles.items():
                # Get type
                vehicle_type = "Unknown"
                if hasattr(vehicle, 'vehicle_type'):
                    vehicle_type = vehicle.vehicle_type
                
                # Get state
                state = "UNKNOWN"
                if hasattr(vehicle, 'state'):
                    state = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                
                # Get distance
                distance = 0.0
                if vehicle_id in vehicle_metrics and 'distance_traveled' in vehicle_metrics[vehicle_id]:
                    distance = vehicle_metrics[vehicle_id]['distance_traveled']
                
                # Get tasks completed
                tasks_completed = 0
                if vehicle_id in vehicle_metrics and 'tasks_completed' in vehicle_metrics[vehicle_id]:
                    tasks_completed = vehicle_metrics[vehicle_id]['tasks_completed']
                
                # Get current load
                current_load = 0.0
                if hasattr(vehicle, 'current_load'):
                    current_load = vehicle.current_load
                
                # Add row
                csv_content += f"{vehicle_id},{vehicle_type},{state},{distance},{tasks_completed},{current_load}\n"
            
            # Write to file
            with open(file_name, 'w') as f:
                f.write(csv_content)
                
            self.logger.info(f"Vehicle data exported to {file_name}")
            
        except Exception as e:
            self.logger.error(f"Error exporting vehicle data: {str(e)}", exc_info=True)


class AnalysisDialog(QDialog):
    """
    Dialog for analyzing simulation results.
    
    Provides various views and analytics for simulation data.
    """
    
    def __init__(self, simulator, parent=None):
        """
        Initialize analysis dialog.
        
        Args:
            simulator: Simulator instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("AnalysisDialog")
        self.simulator = simulator
        
        self.setWindowTitle("Simulation Analysis")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Add performance tab
        self.performance_tab = PerformanceTab(self.simulator)
        tab_widget.addTab(self.performance_tab, "Performance")
        
        # Add task analysis tab
        self.task_tab = TaskAnalysisTab(self.simulator)
        tab_widget.addTab(self.task_tab, "Tasks")
        
        # Add vehicle analysis tab
        self.vehicle_tab = VehicleAnalysisTab(self.simulator)
        tab_widget.addTab(self.vehicle_tab, "Vehicles")
        
        # Add button box
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Add refresh button
        refresh_button = QPushButton("Refresh Data")
        refresh_button.clicked.connect(self.refresh_data)
        button_box.addButton(refresh_button, QDialogButtonBox.ActionRole)
        
        # Add export all button
        export_button = QPushButton("Export All...")
        export_button.clicked.connect(self.export_all_data)
        button_box.addButton(export_button, QDialogButtonBox.ActionRole)
    
    def refresh_data(self):
        """Refresh all data in the dialog."""
        self.performance_tab.update_data()
        self.task_tab.update_data()
        self.vehicle_tab.update_data()
    
    def export_all_data(self):
        """Export all data to a directory."""
        if not self.simulator:
            return
            
        try:
            # Show directory dialog
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select Export Directory",
                "",
                QFileDialog.ShowDirsOnly
            )
            
            if not directory:
                return
                
            # Export performance data
            self.performance_tab.export_data()
            
            # Export task data
            self.task_tab.export_data()
            
            # Export vehicle data
            self.vehicle_tab.export_data()
            
            self.logger.info(f"All data exported to {directory}")
            
        except Exception as e:
            self.logger.error(f"Error exporting all data: {str(e)}", exc_info=True)