"""
Vehicle state definitions for the mining dispatch system.

Defines the possible states and transitions for vehicles in the system.
"""

from enum import Enum, auto
from typing import Dict, List, Set, Optional


class VehicleState(Enum):
    """
    Enumeration of possible vehicle states.
    
    Each state represents a discrete operational mode of a vehicle.
    """
    
    IDLE = auto()             # Vehicle is not assigned to any task
    PREPARING = auto()        # Vehicle is preparing for a task (loading, etc.)
    EN_ROUTE = auto()         # Vehicle is moving to a destination
    LOADING = auto()          # Vehicle is being loaded
    UNLOADING = auto()        # Vehicle is being unloaded
    WAITING = auto()          # Vehicle is waiting (e.g., at an intersection)
    EMERGENCY_STOP = auto()   # Vehicle is stopped due to emergency
    MAINTENANCE = auto()      # Vehicle is undergoing maintenance
    OUT_OF_SERVICE = auto()   # Vehicle is out of service
    CHARGING = auto()         # Vehicle is charging (for electric vehicles)
    FAULT = auto()            # Vehicle has a fault condition

    @classmethod
    def get_valid_transitions(cls, current_state: 'VehicleState') -> Set['VehicleState']:
        """
        Get the set of valid state transitions from a given state.
        
        Args:
            current_state: The current vehicle state
            
        Returns:
            Set[VehicleState]: Set of valid next states
        """
        # Define valid state transitions
        transitions = {
            cls.IDLE: {cls.PREPARING, cls.EN_ROUTE, cls.MAINTENANCE, cls.OUT_OF_SERVICE, cls.CHARGING},
            cls.PREPARING: {cls.EN_ROUTE, cls.IDLE, cls.EMERGENCY_STOP, cls.FAULT},
            cls.EN_ROUTE: {cls.LOADING, cls.UNLOADING, cls.WAITING, cls.IDLE, cls.EMERGENCY_STOP, cls.FAULT},
            cls.LOADING: {cls.EN_ROUTE, cls.EMERGENCY_STOP, cls.FAULT},
            cls.UNLOADING: {cls.EN_ROUTE, cls.IDLE, cls.EMERGENCY_STOP, cls.FAULT},
            cls.WAITING: {cls.EN_ROUTE, cls.EMERGENCY_STOP, cls.FAULT},
            cls.EMERGENCY_STOP: {cls.IDLE, cls.EN_ROUTE, cls.FAULT, cls.MAINTENANCE},
            cls.MAINTENANCE: {cls.IDLE, cls.OUT_OF_SERVICE},
            cls.OUT_OF_SERVICE: {cls.MAINTENANCE, cls.IDLE},
            cls.CHARGING: {cls.IDLE, cls.FAULT},
            cls.FAULT: {cls.MAINTENANCE, cls.OUT_OF_SERVICE, cls.EMERGENCY_STOP}
        }
        
        return transitions.get(current_state, set())
    
    @classmethod
    def can_transition(cls, current_state: 'VehicleState', 
                    next_state: 'VehicleState') -> bool:
        """
        Check if a transition from current_state to next_state is valid.
        
        Args:
            current_state: The current vehicle state
            next_state: The target vehicle state
            
        Returns:
            bool: True if transition is valid, False otherwise
        """
        valid_transitions = cls.get_valid_transitions(current_state)
        return next_state in valid_transitions


class TransportStage(Enum):
    """
    Enumeration of transport stages for vehicles.
    
    Represents the logical stage of a transport operation, independent of the
    physical vehicle state.
    """
    
    NONE = auto()              # Not in a transport operation
    APPROACHING = auto()       # Approaching pickup location
    LOADING = auto()           # At pickup location, being loaded
    TRANSPORTING = auto()      # Carrying cargo to destination
    UNLOADING = auto()         # At destination, being unloaded
    RETURNING = auto()         # Returning to base/idle position


class VehicleStateError(Exception):
    """Exception raised for invalid state transitions."""
    
    def __init__(self, current_state: VehicleState, target_state: VehicleState):
        """
        Initialize a state transition error.
        
        Args:
            current_state: Current vehicle state
            target_state: Target vehicle state
        """
        self.current_state = current_state
        self.target_state = target_state
        super().__init__(
            f"Invalid state transition: {current_state.name} -> {target_state.name}"
        )


class VehicleStateManager:
    """
    Manages the state and state transitions of a vehicle.
    
    Ensures that state transitions are valid and maintains history of states.
    """
    
    def __init__(self, initial_state: VehicleState = VehicleState.IDLE):
        """
        Initialize the state manager.
        
        Args:
            initial_state: Initial vehicle state
        """
        self._current_state = initial_state
        self._state_history = [(initial_state, None)]  # (state, timestamp)
        self._transport_stage = TransportStage.NONE
    
    @property
    def current_state(self) -> VehicleState:
        """Get the current vehicle state."""
        return self._current_state
    
    @property
    def transport_stage(self) -> TransportStage:
        """Get the current transport stage."""
        return self._transport_stage
    
    @transport_stage.setter
    def transport_stage(self, stage: TransportStage) -> None:
        """
        Set the transport stage.
        
        Args:
            stage: New transport stage
        """
        self._transport_stage = stage
    
    def transition_to(self, target_state: VehicleState, 
                     force: bool = False) -> bool:
        """
        Transition to a new state.
        
        Args:
            target_state: Target state
            force: If True, allow invalid transitions
            
        Returns:
            bool: True if transition was successful
            
        Raises:
            VehicleStateError: If transition is invalid and force=False
        """
        # Check if transition is valid
        if not force and not VehicleState.can_transition(self._current_state, target_state):
            raise VehicleStateError(self._current_state, target_state)
        
        # Record previous state
        previous_state = self._current_state
        
        # Update current state
        self._current_state = target_state
        
        # Update history with timestamp
        import time
        self._state_history.append((target_state, time.time()))
        
        # Update transport stage based on state if applicable
        self._update_transport_stage(previous_state, target_state)
        
        return True
    
    def _update_transport_stage(self, 
                              previous_state: VehicleState,
                              current_state: VehicleState) -> None:
        """
        Update transport stage based on state transition.
        
        Args:
            previous_state: Previous vehicle state
            current_state: Current vehicle state
        """
        # State transitions that affect transport stage
        if current_state == VehicleState.LOADING:
            self._transport_stage = TransportStage.LOADING
        elif current_state == VehicleState.UNLOADING:
            self._transport_stage = TransportStage.UNLOADING
        elif current_state == VehicleState.EN_ROUTE:
            # Keep current transport stage or set to APPROACHING if none
            if self._transport_stage == TransportStage.NONE:
                self._transport_stage = TransportStage.APPROACHING
            elif self._transport_stage == TransportStage.LOADING:
                self._transport_stage = TransportStage.TRANSPORTING
            elif self._transport_stage == TransportStage.UNLOADING:
                self._transport_stage = TransportStage.RETURNING
        elif current_state == VehicleState.IDLE:
            self._transport_stage = TransportStage.NONE
    
    def get_state_history(self) -> List[tuple]:
        """
        Get the history of state transitions.
        
        Returns:
            List[tuple]: List of (state, timestamp) tuples
        """
        return self._state_history.copy()
    
    def get_time_in_current_state(self) -> float:
        """
        Get the time spent in the current state (seconds).
        
        Returns:
            float: Time in seconds
        """
        if len(self._state_history) < 1:
            return 0.0
            
        import time
        last_transition_time = self._state_history[-1][1]
        
        if last_transition_time is None:
            return 0.0
            
        return time.time() - last_transition_time
    
    def reset(self, state: VehicleState = VehicleState.IDLE) -> None:
        """
        Reset the state manager to a specified state.
        
        Args:
            state: State to reset to
        """
        import time
        self._current_state = state
        self._state_history = [(state, time.time())]
        self._transport_stage = TransportStage.NONE
    
    def __str__(self) -> str:
        """String representation of the current state."""
        return f"VehicleState: {self._current_state.name}, Stage: {self._transport_stage.name}"