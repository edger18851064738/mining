"""
Mining simulator for the dispatch system.

Provides a simulation environment for testing and evaluating
the mining dispatch system with simulated vehicles and tasks.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import threading
import time
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Any, Optional, Callable, Union
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
from utils.logger import get_logger, timed
from utils.config import SystemConfig, get_config
from utils.geo.coordinates import Point2D
from utils.io.serialization import Serializable

from coordination.simulation.simulation_clock import SimulationClock, TimeUpdateMode, ClockStatus
from coordination.dispatcher.dispatch_events import (
    EventType, DispatchEvent, VehicleEvent, TaskEvent, 
    AssignmentEvent, PathEvent, ConflictEvent, SystemEvent,
    EventListener
)


class SimulationStatus(Enum):
    """Status of the simulation."""
    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    ERROR = auto()


class SimulationConfig:
    """Configuration for the simulation."""
    
    def __init__(self, 
                 duration: Optional[timedelta] = None,
                 seed: Optional[int] = None,
                 log_level: str = "INFO",
                 real_time_factor: float = 10.0,
                 task_generation_rate: float = 1.0,
                 vehicle_count: int = 5,
                 environment_size: Tuple[int, int] = (1000, 1000),
                 **kwargs):
        """
        Initialize simulation configuration.
        
        Args:
            duration: Total duration of simulation (None for unlimited)
            seed: Random seed for reproducibility
            log_level: Logging level
            real_time_factor: How much faster than real time
            task_generation_rate: Tasks per minute
            vehicle_count: Number of vehicles to simulate
            environment_size: Size of environment (width, height)
            **kwargs: Additional configuration parameters
        """
        self.duration = duration
        self.seed = seed
        self.log_level = log_level
        self.real_time_factor = real_time_factor
        self.task_generation_rate = task_generation_rate
        self.vehicle_count = vehicle_count
        self.environment_size = environment_size
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""
    
    # Basic statistics
    total_vehicles: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Time metrics
    total_simulation_time: float = 0.0  # seconds
    total_vehicle_travel_time: float = 0.0  # seconds
    total_vehicle_idle_time: float = 0.0  # seconds
    total_task_waiting_time: float = 0.0  # seconds
    total_task_execution_time: float = 0.0  # seconds
    
    # Distance metrics
    total_distance_traveled: float = 0.0  # meters
    
    # Efficiency metrics
    throughput: float = 0.0  # tasks per hour
    average_task_completion_time: float = 0.0  # seconds
    average_task_waiting_time: float = 0.0  # seconds
    
    # Resource utilization
    average_vehicle_utilization: float = 0.0  # percentage
    
    # Additional metrics
    metrics_by_vehicle: Dict[str, Dict[str, float]] = None
    metrics_by_task_type: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        """Initialize nested dictionaries."""
        self.metrics_by_vehicle = {}
        self.metrics_by_task_type = {}
    
    def update_efficiency_metrics(self) -> None:
        """Update derived efficiency metrics."""
        # Calculate throughput
        if self.total_simulation_time > 0:
            self.throughput = (self.completed_tasks / self.total_simulation_time) * 3600  # tasks per hour
        
        # Calculate average times
        if self.completed_tasks > 0:
            self.average_task_completion_time = self.total_task_execution_time / self.completed_tasks
            self.average_task_waiting_time = self.total_task_waiting_time / self.completed_tasks
        
        # Calculate utilization
        if self.total_vehicles > 0 and self.total_simulation_time > 0:
            total_possible_time = self.total_vehicles * self.total_simulation_time
            if total_possible_time > 0:
                self.average_vehicle_utilization = (
                    (total_possible_time - self.total_vehicle_idle_time) / total_possible_time
                ) * 100


class SimulationEventListener(EventListener):
    """Event listener that collects metrics from dispatch events."""
    
    def __init__(self, simulator):
        """
        Initialize the event listener.
        
        Args:
            simulator: The simulator instance
        """
        self.simulator = simulator
        self.logger = get_logger("SimEventListener")
    
    def on_event(self, event: DispatchEvent) -> None:
        """
        Handle a dispatch event.
        
        Args:
            event: The event to handle
        """
        # Process event based on type
        if event.event_type == EventType.TASK_COMPLETED:
            # Update task completion metrics
            self.simulator.metrics.completed_tasks += 1
            
            # Additional metrics could be collected here
            
        elif event.event_type == EventType.TASK_FAILED:
            # Update task failure metrics
            self.simulator.metrics.failed_tasks += 1
            
        elif event.event_type == EventType.VEHICLE_POSITION_CHANGED:
            # Could update distance traveled
            if isinstance(event, VehicleEvent) and event.details:
                old_pos = event.details.get('old_position')
                new_pos = event.details.get('new_position')
                
                if old_pos and new_pos:
                    # Calculate distance
                    distance = Point2D(*old_pos).distance_to(Point2D(*new_pos))
                    
                    # Update metrics
                    self.simulator.metrics.total_distance_traveled += distance
                    
                    # Update vehicle-specific metrics
                    if event.vehicle_id not in self.simulator.metrics.metrics_by_vehicle:
                        self.simulator.metrics.metrics_by_vehicle[event.vehicle_id] = {
                            'distance_traveled': 0.0
                        }
                    
                    self.simulator.metrics.metrics_by_vehicle[event.vehicle_id]['distance_traveled'] += distance
        
        # Additional event types can be handled here


class Simulator(Serializable):
    """
    Mining simulator for the dispatch system.
    
    Simulates a mining environment with vehicles and tasks
    to evaluate the performance of the dispatch system.
    """
    
    def __init__(self, 
                environment=None,
                dispatcher=None,
                config: Optional[SimulationConfig] = None):
        """
        Initialize the simulator.
        
        Args:
            environment: Mining environment
            dispatcher: Dispatcher instance
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        self.logger = get_logger("Simulator")
        
        # Set random seed if provided
        if self.config.seed is not None:
            random.seed(self.config.seed)
        
        # Store components
        self.environment = environment
        self.dispatcher = dispatcher
        
        # Initialize simulation clock
        self.clock = SimulationClock(
            speed_factor=self.config.real_time_factor,
            update_mode=TimeUpdateMode.SCALED_TIME
        )
        
        # Initialize metrics
        self.metrics = SimulationMetrics()
        
        # Initialize state
        self._status = SimulationStatus.STOPPED
        self._task_generator_timer = None
        self._stop_event = threading.Event()
        
        # Initialize event listener
        self._event_listener = SimulationEventListener(self)
        
        # Initialize vehicle and task collections
        self._vehicles = {}
        self._tasks = {}
        
        # Initialize lock
        self._lock = threading.RLock()
        
        self.logger.info("Simulator initialized")
    
    @property
    def status(self) -> SimulationStatus:
        """Get the current simulation status."""
        return self._status
    
    def setup(self) -> None:
        """
        Set up the simulation environment.
        
        Creates the environment, vehicles, and initial tasks.
        """
        with self._lock:
            # Skip if already running
            if self._status != SimulationStatus.STOPPED:
                return
            
            try:
                # Create environment if needed
                if self.environment is None:
                    self._create_environment()
                
                # Create dispatcher if needed
                if self.dispatcher is None:
                    self._create_dispatcher()
                
                # Add event listener to dispatcher
                if self.dispatcher:
                    self.dispatcher.add_event_listener(EventType.ALL, self._event_listener)
                
                # Create vehicles
                self._create_vehicles()
                
                # Create initial tasks
                self._create_initial_tasks()
                
                self.logger.info("Simulation setup complete")
                
            except Exception as e:
                self.logger.error(f"Error in simulation setup: {str(e)}", exc_info=True)
                self._status = SimulationStatus.ERROR
                raise
    
    def _create_environment(self) -> None:
        """Create the mining environment."""
        # This method should be implemented when the domain models are available
        # For now, we'll use a placeholder
        self.logger.info("Creating simulation environment")
        
        try:
            # Placeholder - replace with actual environment creation
            # self.environment = MiningEnvironment(...)
            self.environment = None  # Placeholder until implementation
            
            self.logger.info("Environment created")
            
        except Exception as e:
            self.logger.error(f"Error creating environment: {str(e)}", exc_info=True)
            raise
    
    def _create_dispatcher(self) -> None:
        """Create the dispatcher."""
        # This method should be implemented when the dispatcher is available
        # For now, we'll use a placeholder
        self.logger.info("Creating dispatcher")
        
        try:
            # Placeholder - replace with actual dispatcher creation
            # from coordination.dispatcher.mining_dispatcher import MiningDispatcher
            # self.dispatcher = MiningDispatcher(self.environment)
            self.dispatcher = None  # Placeholder until implementation
            
            self.logger.info("Dispatcher created")
            
        except Exception as e:
            self.logger.error(f"Error creating dispatcher: {str(e)}", exc_info=True)
            raise
    
    def _create_vehicles(self) -> None:
        """Create simulated vehicles."""
        # This method should be implemented when the vehicle models are available
        # For now, we'll use placeholder logic
        self.logger.info(f"Creating {self.config.vehicle_count} vehicles")
        
        try:
            # Placeholder - replace with actual vehicle creation
            # for i in range(self.config.vehicle_count):
            #     vehicle = MiningVehicle(f"V{i+1}", environment=self.environment)
            #     self._vehicles[vehicle.vehicle_id] = vehicle
            #     self.dispatcher.add_vehicle(vehicle)
            
            self.metrics.total_vehicles = self.config.vehicle_count
            
            self.logger.info(f"Created {self.config.vehicle_count} vehicles")
            
        except Exception as e:
            self.logger.error(f"Error creating vehicles: {str(e)}", exc_info=True)
            raise
    
    def _create_initial_tasks(self) -> None:
        """Create initial tasks for the simulation."""
        # This method should be implemented when the task models are available
        # For now, we'll use placeholder logic
        self.logger.info("Creating initial tasks")
        
        try:
            # Placeholder - replace with actual task creation
            # Create a few initial tasks
            # for i in range(3):
            #     task = self._generate_random_task()
            #     self._tasks[task.task_id] = task
            #     self.dispatcher.add_task(task)
            
            self.logger.info("Initial tasks created")
            
        except Exception as e:
            self.logger.error(f"Error creating initial tasks: {str(e)}", exc_info=True)
            raise
    
    def _generate_random_task(self) -> Any:
        """
        Generate a random task.
        
        Returns:
            Task: A randomly generated task
        """
        # This method should be implemented when the task models are available
        # For now, we'll return None as a placeholder
        return None
    
    def start(self) -> None:
        """
        Start the simulation.
        
        Begins executing the simulation loop and advancing time.
        """
        with self._lock:
            # Skip if already running
            if self._status == SimulationStatus.RUNNING:
                return
            
            # Ensure setup is done
            if self._status == SimulationStatus.STOPPED:
                self.setup()
            
            # Update status
            self._status = SimulationStatus.RUNNING
            
            # Start the clock
            self.clock.start()
            
            # Start the dispatcher if available
            if self.dispatcher:
                self.dispatcher.start()
            
            # Clear stop event
            self._stop_event.clear()
            
            # Schedule task generation
            self._schedule_task_generation()
            
            self.logger.info("Simulation started")
    
    def stop(self) -> None:
        """
        Stop the simulation.
        
        Stops executing the simulation and resets state.
        """
        with self._lock:
            # Skip if already stopped
            if self._status == SimulationStatus.STOPPED:
                return
            
            # Update status
            self._status = SimulationStatus.STOPPED
            
            # Stop the clock
            self.clock.stop()
            
            # Stop the dispatcher if available
            if self.dispatcher:
                self.dispatcher.stop()
            
            # Signal to stop task generation
            self._stop_event.set()
            
            # Cancel task generator timer
            if self._task_generator_timer:
                self.clock.cancel_event(self._task_generator_timer)
                self._task_generator_timer = None
            
            # Finalize metrics
            self._finalize_metrics()
            
            self.logger.info("Simulation stopped")
    
    def pause(self) -> None:
        """
        Pause the simulation.
        
        Temporarily stops the simulation execution.
        """
        with self._lock:
            # Skip if not running
            if self._status != SimulationStatus.RUNNING:
                return
            
            # Update status
            self._status = SimulationStatus.PAUSED
            
            # Pause the clock
            self.clock.pause()
            
            # Pause the dispatcher if available
            if self.dispatcher:
                self.dispatcher.pause()
            
            self.logger.info("Simulation paused")
    
    def resume(self) -> None:
        """
        Resume the simulation.
        
        Continues executing a paused simulation.
        """
        with self._lock:
            # Skip if not paused
            if self._status != SimulationStatus.PAUSED:
                return
            
            # Update status
            self._status = SimulationStatus.RUNNING
            
            # Resume the clock
            self.clock.resume()
            
            # Resume the dispatcher if available
            if self.dispatcher:
                self.dispatcher.resume()
            
            self.logger.info("Simulation resumed")
    
    def step(self, time_step: float = 1.0) -> None:
        """
        Advance the simulation by a single time step.
        
        Args:
            time_step: Time to advance in seconds
        """
        with self._lock:
            # Advance the clock
            self.clock.advance(timedelta(seconds=time_step))
            
            # Run dispatcher cycle
            if self.dispatcher:
                self.dispatcher.dispatch_cycle()
            
            # Update metrics
            self._update_metrics()
            
            self.logger.debug(f"Simulation stepped by {time_step} seconds")
    
    def run_until(self, end_time: datetime) -> None:
        """
        Run the simulation until a specific time.
        
        Args:
            end_time: Time to run until
        """
        with self._lock:
            # Skip if not running
            if self._status != SimulationStatus.RUNNING:
                self.start()
            
            # Check if end_time is in the future
            if end_time <= self.clock.current_time:
                self.logger.warning(f"End time {end_time} is not in the future, nothing to do")
                return
            
            # Set the clock to advance to the end time
            self.clock.set_time(end_time)
            
            # Run dispatcher cycle
            if self.dispatcher:
                self.dispatcher.dispatch_cycle()
            
            # Update metrics
            self._update_metrics()
            
            self.logger.info(f"Simulation run until {end_time}")
    
    def run_for(self, duration: timedelta) -> None:
        """
        Run the simulation for a specific duration.
        
        Args:
            duration: Duration to run
        """
        # Calculate end time
        end_time = self.clock.current_time + duration
        
        # Run until end time
        self.run_until(end_time)
    
    def _schedule_task_generation(self) -> None:
        """Schedule the task generation timer."""
        # Skip if not running
        if self._status != SimulationStatus.RUNNING:
            return
        
        # Calculate interval based on task generation rate
        if self.config.task_generation_rate > 0:
            # Calculate seconds between tasks
            seconds_per_task = 60.0 / self.config.task_generation_rate
            
            # Add randomness to prevent regular patterns
            seconds_per_task *= random.uniform(0.8, 1.2)
            
            # Schedule the next task generation
            self._task_generator_timer = self.clock.schedule_event(
                seconds_per_task, self._generate_task
            )
    
    def _generate_task(self) -> None:
        """Generate a new random task and add it to the system."""
        try:
            # Skip if not running
            if self._status != SimulationStatus.RUNNING:
                return
            
            # Generate a task
            task = self._generate_random_task()
            
            # Add task to tracking
            if task:
                task_id = getattr(task, 'task_id', str(id(task)))
                self._tasks[task_id] = task
                
                # Add to dispatcher
                if self.dispatcher:
                    self.dispatcher.add_task(task)
                
                # Update metrics
                self.metrics.total_tasks += 1
                
                self.logger.debug(f"Generated new task: {task_id}")
            
            # Reschedule task generation
            self._schedule_task_generation()
            
        except Exception as e:
            self.logger.error(f"Error generating task: {str(e)}", exc_info=True)
    
    def _update_metrics(self) -> None:
        """Update simulation metrics."""
        # Update simulation time
        self.metrics.total_simulation_time = self.clock.elapsed_seconds
        
        # Update efficiency metrics
        self.metrics.update_efficiency_metrics()
    
    def _finalize_metrics(self) -> None:
        """Finalize metrics at the end of simulation."""
        # Update metrics one last time
        self._update_metrics()
        
        # Log summary
        self.logger.info(f"Simulation completed: {self.metrics.completed_tasks} tasks completed, "
                        f"{self.metrics.failed_tasks} tasks failed")
        self.logger.info(f"Total distance traveled: {self.metrics.total_distance_traveled:.2f} meters")
        self.logger.info(f"Average vehicle utilization: {self.metrics.average_vehicle_utilization:.2f}%")
        self.logger.info(f"Throughput: {self.metrics.throughput:.2f} tasks/hour")
    
    def get_metrics(self) -> SimulationMetrics:
        """
        Get the current simulation metrics.
        
        Returns:
            SimulationMetrics: Current metrics
        """
        # Update metrics before returning
        self._update_metrics()
        
        return self.metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert simulator state to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "status": self._status.name,
            "current_time": self.clock.current_time.isoformat(),
            "elapsed_time": str(self.clock.elapsed_time),
            "metrics": {
                "total_vehicles": self.metrics.total_vehicles,
                "total_tasks": self.metrics.total_tasks,
                "completed_tasks": self.metrics.completed_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "total_distance_traveled": self.metrics.total_distance_traveled,
                "average_vehicle_utilization": self.metrics.average_vehicle_utilization,
                "throughput": self.metrics.throughput
            },
            "config": {
                "real_time_factor": self.config.real_time_factor,
                "task_generation_rate": self.config.task_generation_rate,
                "vehicle_count": self.config.vehicle_count,
                "environment_size": self.config.environment_size
            }
        }