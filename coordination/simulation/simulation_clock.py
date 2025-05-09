"""
Simulation clock for the mining dispatch system.

Provides time management functionality for simulations.
"""
import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import time
import threading
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Any, Optional, Callable
from datetime import datetime, timedelta
import heapq

from utils.logger import get_logger


class ClockStatus(Enum):
    """Status of the simulation clock."""
    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()


class TimeUpdateMode(Enum):
    """Modes for updating simulation time."""
    REAL_TIME = auto()      # Simulation time advances with real time
    SCALED_TIME = auto()    # Simulation time advances faster/slower than real time
    STEPPED = auto()        # Simulation time advances in discrete steps
    EVENT_BASED = auto()    # Simulation time advances to next scheduled event


class ScheduledEvent:
    """Event scheduled for execution at a specific simulation time."""
    
    def __init__(self, sim_time: float, event_id: int, callback: Callable, data: Any = None):
        """
        Initialize a scheduled event.
        
        Args:
            sim_time: Simulation time when event should execute
            event_id: Unique identifier for this event
            callback: Function to call when event executes
            data: Optional data to pass to callback
        """
        self.sim_time = sim_time
        self.event_id = event_id
        self.callback = callback
        self.data = data
    
    def __lt__(self, other):
        """Compare events for priority queue ordering."""
        if not isinstance(other, ScheduledEvent):
            return NotImplemented
        return self.sim_time < other.sim_time


class SimulationClock:
    """
    Clock for managing time in simulation.
    
    Provides functionality to advance time in various modes,
    schedule events, and synchronize simulation components.
    """
    
    def __init__(self, 
                initial_time: Optional[datetime] = None,
                speed_factor: float = 1.0,
                update_mode: TimeUpdateMode = TimeUpdateMode.SCALED_TIME,
                update_interval: float = 0.1):
        """
        Initialize the simulation clock.
        
        Args:
            initial_time: Starting simulation time (default: current time)
            speed_factor: How much faster/slower than real time (default: 1.0)
            update_mode: Mode for advancing simulation time
            update_interval: Time between updates in seconds (real time)
        """
        self.logger = get_logger("SimulationClock")
        
        # Initialize time parameters
        self._sim_time = initial_time or datetime.now()
        self._start_wall_time = time.time()
        self._start_sim_time = self._sim_time
        self._speed_factor = speed_factor
        self._update_mode = update_mode
        self._update_interval = update_interval
        
        # Initialize status
        self._status = ClockStatus.STOPPED
        
        # Initialize scheduled events
        self._events = []  # Priority queue of scheduled events
        self._next_event_id = 0
        self._event_lock = threading.RLock()
        
        # Initialize update thread
        self._update_thread = None
        self._stop_event = threading.Event()
        
        # Initialize time listeners
        self._time_listeners = set()
    
    @property
    def current_time(self) -> datetime:
        """Get the current simulation time."""
        return self._sim_time
    
    @property
    def current_time_float(self) -> float:
        """Get the current simulation time as float (seconds since epoch)."""
        return self._sim_time.timestamp()
    
    @property
    def elapsed_time(self) -> timedelta:
        """Get the elapsed simulation time since start."""
        return self._sim_time - self._start_sim_time
    
    @property
    def elapsed_seconds(self) -> float:
        """Get the elapsed simulation time in seconds."""
        return self.elapsed_time.total_seconds()
    
    @property
    def status(self) -> ClockStatus:
        """Get the current clock status."""
        return self._status
    
    @property
    def speed_factor(self) -> float:
        """Get the speed factor."""
        return self._speed_factor
    
    @speed_factor.setter
    def speed_factor(self, value: float) -> None:
        """
        Set the speed factor.
        
        Args:
            value: New speed factor (must be positive)
            
        Raises:
            ValueError: If speed factor is not positive
        """
        if value <= 0:
            raise ValueError("Speed factor must be positive")
        
        with self._event_lock:
            # Reset wall time reference when changing speed
            self._start_wall_time = time.time()
            self._start_sim_time = self._sim_time
            self._speed_factor = value
            
            self.logger.info(f"Speed factor set to {value}")
    
    @property
    def update_mode(self) -> TimeUpdateMode:
        """Get the update mode."""
        return self._update_mode
    
    @update_mode.setter
    def update_mode(self, mode: TimeUpdateMode) -> None:
        """Set the update mode."""
        with self._event_lock:
            old_mode = self._update_mode
            self._update_mode = mode
            
            # Reset start times when changing mode
            self._start_wall_time = time.time()
            self._start_sim_time = self._sim_time
            
            self.logger.info(f"Update mode changed from {old_mode.name} to {mode.name}")
    
    def start(self) -> None:
        """
        Start the simulation clock.
        
        Begins advancing simulation time according to the update mode.
        """
        if self._status == ClockStatus.RUNNING:
            return
        
        # Reset start times
        self._start_wall_time = time.time()
        
        # If previously paused, don't reset start_sim_time
        if self._status != ClockStatus.PAUSED:
            self._start_sim_time = self._sim_time
        
        # Set status
        self._status = ClockStatus.RUNNING
        
        # Clear stop event
        self._stop_event.clear()
        
        # Start update thread
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
        
        self.logger.info(f"Clock started at {self._sim_time}")
    
    def stop(self) -> None:
        """
        Stop the simulation clock.
        
        Stops advancing simulation time and resets to initial time.
        """
        if self._status == ClockStatus.STOPPED:
            return
        
        # Set status
        self._status = ClockStatus.STOPPED
        
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
        
        self._update_thread = None
        
        # Reset time
        self._sim_time = self._start_sim_time
        
        # Clear event queue
        with self._event_lock:
            self._events = []
        
        self.logger.info("Clock stopped and reset")
    
    def pause(self) -> None:
        """
        Pause the simulation clock.
        
        Temporarily stops advancing simulation time but preserves current time.
        """
        if self._status != ClockStatus.RUNNING:
            return
        
        # Set status
        self._status = ClockStatus.PAUSED
        
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
        
        self._update_thread = None
        
        self.logger.info(f"Clock paused at {self._sim_time}")
    
    def resume(self) -> None:
        """
        Resume the simulation clock.
        
        Continues advancing simulation time from current time.
        """
        if self._status != ClockStatus.PAUSED:
            return
        
        # Start the clock again
        self.start()
        
        self.logger.info(f"Clock resumed at {self._sim_time}")
    
    def set_time(self, new_time: datetime) -> None:
        """
        Set the simulation time.
        
        Args:
            new_time: New simulation time
        """
        with self._event_lock:
            old_time = self._sim_time
            self._sim_time = new_time
            self._start_wall_time = time.time()
            self._start_sim_time = new_time
            
            # Process events that should have happened
            if new_time > old_time:
                self._process_events_until(new_time)
            
            self.logger.info(f"Time set to {new_time}")
            
            # Notify listeners
            self._notify_time_listeners()
    
    def advance(self, delta: timedelta) -> None:
        """
        Advance the simulation time by a specified amount.
        
        Args:
            delta: Time to advance
        """
        with self._event_lock:
            new_time = self._sim_time + delta
            self._sim_time = new_time
            
            # Process events that should have happened
            self._process_events_until(new_time)
            
            self.logger.debug(f"Time advanced to {new_time}")
            
            # Notify listeners
            self._notify_time_listeners()
    
    def schedule_event(self, delay: float, callback: Callable, data: Any = None) -> int:
        """
        Schedule an event to occur after a delay.
        
        Args:
            delay: Delay in simulation seconds
            callback: Function to call when event occurs
            data: Optional data to pass to callback
            
        Returns:
            int: Event ID for cancellation
        """
        with self._event_lock:
            event_time = self.current_time_float + delay
            event_id = self._next_event_id
            self._next_event_id += 1
            
            event = ScheduledEvent(event_time, event_id, callback, data)
            heapq.heappush(self._events, event)
            
            self.logger.debug(f"Scheduled event {event_id} for {datetime.fromtimestamp(event_time)}")
            
            return event_id
    
    def schedule_at(self, time_point: datetime, callback: Callable, data: Any = None) -> int:
        """
        Schedule an event to occur at a specific time.
        
        Args:
            time_point: Simulation time for event
            callback: Function to call when event occurs
            data: Optional data to pass to callback
            
        Returns:
            int: Event ID for cancellation
        """
        with self._event_lock:
            event_time = time_point.timestamp()
            event_id = self._next_event_id
            self._next_event_id += 1
            
            event = ScheduledEvent(event_time, event_id, callback, data)
            heapq.heappush(self._events, event)
            
            self.logger.debug(f"Scheduled event {event_id} for {time_point}")
            
            return event_id
    
    def cancel_event(self, event_id: int) -> bool:
        """
        Cancel a scheduled event.
        
        Args:
            event_id: ID of event to cancel
            
        Returns:
            bool: True if event was found and canceled
        """
        with self._event_lock:
            # Find the event
            for i, event in enumerate(self._events):
                if event.event_id == event_id:
                    # Remove the event
                    self._events[i] = self._events[-1]
                    self._events.pop()
                    
                    # Restore heap property
                    if i < len(self._events):
                        heapq.heapify(self._events)
                    
                    self.logger.debug(f"Canceled event {event_id}")
                    return True
            
            return False
    
    def add_time_listener(self, listener: Callable[[datetime], None]) -> None:
        """
        Add a listener to be notified of time updates.
        
        Args:
            listener: Function to call with current time
        """
        self._time_listeners.add(listener)
    
    def remove_time_listener(self, listener: Callable[[datetime], None]) -> None:
        """
        Remove a time listener.
        
        Args:
            listener: Listener to remove
        """
        if listener in self._time_listeners:
            self._time_listeners.remove(listener)
    
    def _notify_time_listeners(self) -> None:
        """Notify all time listeners of current time."""
        current_time = self._sim_time
        for listener in self._time_listeners:
            try:
                listener(current_time)
            except Exception as e:
                self.logger.error(f"Error in time listener: {str(e)}")
    
    def _update_loop(self) -> None:
        """Main loop for updating simulation time."""
        while not self._stop_event.is_set():
            try:
                # Update time based on mode
                self._update_time()
                
                # Sleep until next update
                time.sleep(self._update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {str(e)}", exc_info=True)
                time.sleep(1.0)  # Avoid busy loop in case of error
    
    def _update_time(self) -> None:
        """Update simulation time based on update mode."""
        with self._event_lock:
            old_time = self._sim_time
            
            if self._update_mode == TimeUpdateMode.REAL_TIME:
                # Advance time to match real time
                elapsed_real = time.time() - self._start_wall_time
                new_time = self._start_sim_time + timedelta(seconds=elapsed_real)
                self._sim_time = new_time
                
            elif self._update_mode == TimeUpdateMode.SCALED_TIME:
                # Advance time faster/slower than real time
                elapsed_real = time.time() - self._start_wall_time
                elapsed_sim = elapsed_real * self._speed_factor
                new_time = self._start_sim_time + timedelta(seconds=elapsed_sim)
                self._sim_time = new_time
                
            elif self._update_mode == TimeUpdateMode.STEPPED:
                # Time advances only via explicit advance() calls
                return
                
            elif self._update_mode == TimeUpdateMode.EVENT_BASED:
                # Time advances to next event
                if self._events:
                    next_event = self._events[0]
                    new_time = datetime.fromtimestamp(next_event.sim_time)
                    self._sim_time = new_time
                    
                    # Process events at this time
                    self._process_events_until(new_time)
                else:
                    # No events, don't advance time
                    return
            
            # Process events up to current time
            if old_time != self._sim_time:
                self._process_events_until(self._sim_time)
                self._notify_time_listeners()
    
    def _process_events_until(self, until_time: datetime) -> None:
        """
        Process all events scheduled up to the given time.
        
        Args:
            until_time: Process events up to this time
        """
        until_timestamp = until_time.timestamp()
        
        # Process events that should have occurred
        while self._events and self._events[0].sim_time <= until_timestamp:
            event = heapq.heappop(self._events)
            
            try:
                # Call event callback
                if event.data is not None:
                    event.callback(event.data)
                else:
                    event.callback()
            except Exception as e:
                self.logger.error(f"Error executing event {event.event_id}: {str(e)}", exc_info=True)