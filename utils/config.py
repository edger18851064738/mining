"""
Configuration management for the mining dispatch system.

Provides a centralized configuration system that supports:
- Loading from config files
- Environment variable overrides
- Runtime configuration changes
- Configuration validation
"""

import os
import sys
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import yaml
import logging
import configparser
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Type, Set
from dataclasses import dataclass, field, asdict
import copy
import threading

# Default configuration paths
DEFAULT_CONFIG_PATHS = [
    "./config.yaml",
    "./config.json",
    "./config.ini",
    "./config/config.yaml",
    "./config/config.json",
    "./config/config.ini",
]

# Type alias for configuration sections
ConfigSection = Dict[str, Any]


@dataclass
class MapConfig:
    """Map and environment configuration."""
    
    # Map dimensions
    grid_size: int = 200
    grid_nodes: int = 50
    
    # Safety parameters
    safe_radius: int = 30
    obstacle_density: float = 0.15
    
    # Key locations (percentages of grid size)
    key_locations: Dict[str, List[float]] = field(default_factory=lambda: {
        'parking': [0.75, 0.5],
        'unload': [0.95, 0.95],
        'load1': [0.05, 0.95],
        'load2': [0.05, 0.05],
        'load3': [0.95, 0.05]
    })
    
    # Map visualization
    show_grid: bool = True
    grid_color: str = "#cccccc"
    obstacle_color: str = "#555555"
    
    def get_key_location(self, key: str) -> List[float]:
        """Get absolute coordinates for a key location."""
        if key not in self.key_locations:
            raise KeyError(f"Key location '{key}' not found in map configuration")
            
        percentage = self.key_locations[key]
        return [percentage[0] * self.grid_size, percentage[1] * self.grid_size]


@dataclass
class VehicleConfig:
    """Vehicle configuration parameters."""
    
    # Basic parameters
    default_max_speed: float = 5.0
    default_turning_radius: float = 10.0
    default_max_capacity: float = 50000.0
    
    # Vehicle dimensions
    length: float = 5.0
    width: float = 2.0
    
    # Vehicle types with specifications
    vehicle_types: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'small': {
            'max_speed': 7.0,
            'turning_radius': 8.0,
            'max_capacity': 30000.0,
            'length': 4.0,
            'width': 1.8
        },
        'standard': {
            'max_speed': 5.0,
            'turning_radius': 10.0,
            'max_capacity': 50000.0, 
            'length': 5.0,
            'width': 2.0
        },
        'heavy': {
            'max_speed': 3.5,
            'turning_radius': 15.0,
            'max_capacity': 80000.0,
            'length': 6.5,
            'width': 2.5
        }
    })
    
    # Visualization
    default_color: str = "#3070B0"
    vehicle_colors: Dict[str, str] = field(default_factory=lambda: {
        'small': '#70B030',
        'standard': '#3070B0',
        'heavy': '#B03070'
    })
    
    def get_vehicle_spec(self, vehicle_type: str) -> Dict[str, Any]:
        """Get specifications for a vehicle type."""
        if vehicle_type not in self.vehicle_types:
            logging.warning(f"Unknown vehicle type '{vehicle_type}', using 'standard'")
            vehicle_type = 'standard'
            
        return self.vehicle_types[vehicle_type]


@dataclass
class AlgorithmConfig:
    """Algorithm configuration parameters."""
    
    # Path planning parameters
    path_planner: str = "hybrid_astar"  # hybrid_astar, rrt, etc.
    step_size: float = 0.8
    grid_resolution: float = 0.3
    max_iterations: int = 5000
    
    # Conflict resolution parameters
    conflict_resolution: str = "cbs"  # cbs, priority, etc.
    collision_threshold: float = 3.0
    max_replanning_attempts: int = 3
    
    # CBS specific parameters
    cbs_max_nodes: int = 1000
    cbs_time_limit: float = 5.0  # seconds
    
    # Task allocation parameters
    task_allocator: str = "priority"  # priority, auction, miqp, etc.
    
    # Smoothing parameters
    smoothing_enabled: bool = True
    smoothing_factor: float = 0.5
    smoothing_iterations: int = 10
    
    # Advanced options
    use_rs_curves: bool = True
    analytic_expansion_step: int = 5


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    
    # Basic settings
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Output settings
    console_output: bool = True
    file_output: bool = True
    file_path: str = "logs/dispatch.log"
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    
    # Component levels (override default level)
    component_levels: Dict[str, str] = field(default_factory=lambda: {
        "dispatch": "INFO",
        "planning": "INFO",
        "conflict": "INFO",
        "vehicle": "INFO",
        "task": "INFO"
    })


@dataclass
class UIConfig:
    """User interface configuration parameters."""
    
    # Window settings
    window_title: str = "露天矿多车协同调度系统"
    window_width: int = 1280
    window_height: int = 800
    window_maximized: bool = False
    
    # Theme settings
    theme: str = "light"  # light, dark
    primary_color: str = "#3070B0"
    secondary_color: str = "#70B030"
    accent_color: str = "#B03070"
    
    # Map display
    map_background_color: str = "#FFFFFF"
    map_grid_color: str = "#CCCCCC"
    map_obstacle_color: str = "#555555"
    
    # Display options
    show_path: bool = True
    show_vehicle_state: bool = True
    show_conflicts: bool = True
    show_heat_map: bool = False
    
    # Animation settings
    animation_enabled: bool = True
    default_simulation_speed: float = 1.0


@dataclass
class SystemConfig:
    """
    Overall system configuration that contains all other configuration sections.
    """
    
    # Component configurations
    map: MapConfig = field(default_factory=MapConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # System-level settings
    system_name: str = "MiningDispatchSystem"
    version: str = "1.0.0"
    debug_mode: bool = False
    multithreading: bool = True
    thread_pool_size: int = 4
    
    # Simulation parameters
    simulation_step_time: float = 0.05  # seconds
    environment_type: str = "mine"  # mine, warehouse, etc.


class ConfigManager:
    """
    Configuration manager for the mining dispatch system.
    
    This class provides access to configuration settings and ensures
    that settings are loaded from files and can be modified at runtime.
    It implements the Singleton pattern to ensure one global configuration.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        with self._lock:
            if self._initialized:
                return
                
            self._logger = logging.getLogger("config")
            self._config_path = config_path
            self._config = SystemConfig()
            self._initialized = True
            self._listeners = set()
            
            # Try to load configuration file
            if config_path:
                self.load_config(config_path)
            else:
                self._auto_discover_and_load()
    
    def _auto_discover_and_load(self) -> bool:
        """
        Automatically discover and load configuration from default paths.
        
        Returns:
            bool: True if a configuration was loaded, False otherwise
        """
        for path in DEFAULT_CONFIG_PATHS:
            if os.path.exists(path):
                try:
                    self.load_config(path)
                    self._logger.info(f"Configuration loaded from {path}")
                    return True
                except Exception as e:
                    self._logger.warning(f"Failed to load config from {path}: {str(e)}")
        
        self._logger.warning("No configuration file found, using defaults")
        return False
    
    def load_config(self, path: str) -> None:
        """
        Load configuration from a file.
        
        Args:
            path: Path to configuration file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file has invalid format
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        file_ext = os.path.splitext(path)[1].lower()
        
        try:
            if file_ext in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            elif file_ext == '.json':
                with open(path, 'r') as f:
                    config_dict = json.load(f)
            elif file_ext == '.ini':
                config_parser = configparser.ConfigParser()
                config_parser.read(path)
                config_dict = {section: dict(config_parser[section]) 
                               for section in config_parser.sections()}
            else:
                raise ValueError(f"Unsupported configuration file type: {file_ext}")
            
            # Update the configuration
            self.update_from_dict(config_dict)
            self._config_path = path
            
        except Exception as e:
            self._logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to a file.
        
        Args:
            path: Path to save configuration (uses current path if None)
            
        Raises:
            ValueError: If path is None and no current path is set
        """
        save_path = path or self._config_path
        
        if save_path is None:
            raise ValueError("No configuration path specified")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        file_ext = os.path.splitext(save_path)[1].lower()
        config_dict = self.as_dict()
        
        try:
            if file_ext in ['.yaml', '.yml']:
                with open(save_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            elif file_ext == '.json':
                with open(save_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            elif file_ext == '.ini':
                config_parser = configparser.ConfigParser()
                for section, values in config_dict.items():
                    if isinstance(values, dict):
                        config_parser[section] = {
                            k: str(v) for k, v in values.items()
                        }
                
                with open(save_path, 'w') as f:
                    config_parser.write(f)
            else:
                raise ValueError(f"Unsupported configuration file type: {file_ext}")
                
            self._logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self._logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return asdict(self._config)
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
        """
        with self._lock:
            modified_sections = set()
            
            # Update each section
            for section_name, section_config in config_dict.items():
                if hasattr(self._config, section_name) and isinstance(section_config, dict):
                    section = getattr(self._config, section_name)
                    
                    # Update all fields that exist in the section
                    for key, value in section_config.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
                            modified_sections.add(section_name)
                elif hasattr(self._config, section_name):
                    # Direct attribute of main config
                    setattr(self._config, section_name, section_config)
                    modified_sections.add('system')
            
            # Notify listeners about modified sections
            for section in modified_sections:
                self._notify_listeners(section)
    
    def add_listener(self, listener: Callable[[str], None]) -> None:
        """
        Add a configuration change listener.
        
        Args:
            listener: Callback function that takes a section name parameter
        """
        with self._lock:
            self._listeners.add(listener)
    
    def remove_listener(self, listener: Callable[[str], None]) -> None:
        """
        Remove a configuration change listener.
        
        Args:
            listener: Listener to remove
        """
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
    
    def _notify_listeners(self, section: str) -> None:
        """
        Notify all listeners about a configuration change.
        
        Args:
            section: Name of the section that changed
        """
        for listener in self._listeners:
            try:
                listener(section)
            except Exception as e:
                self._logger.warning(f"Error in configuration listener: {str(e)}")
    
    def get_config(self) -> SystemConfig:
        """
        Get the complete configuration object.
        
        Returns:
            SystemConfig: Complete configuration
        """
        return copy.deepcopy(self._config)
    
    def get_map_config(self) -> MapConfig:
        """
        Get map configuration.
        
        Returns:
            MapConfig: Map configuration section
        """
        return copy.deepcopy(self._config.map)
    
    def get_vehicle_config(self) -> VehicleConfig:
        """
        Get vehicle configuration.
        
        Returns:
            VehicleConfig: Vehicle configuration section
        """
        return copy.deepcopy(self._config.vehicle)
    
    def get_algorithm_config(self) -> AlgorithmConfig:
        """
        Get algorithm configuration.
        
        Returns:
            AlgorithmConfig: Algorithm configuration section
        """
        return copy.deepcopy(self._config.algorithm)
    
    def get_logging_config(self) -> LoggingConfig:
        """
        Get logging configuration.
        
        Returns:
            LoggingConfig: Logging configuration section
        """
        return copy.deepcopy(self._config.logging)
    
    def get_ui_config(self) -> UIConfig:
        """
        Get UI configuration.
        
        Returns:
            UIConfig: UI configuration section
        """
        return copy.deepcopy(self._config.ui)
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            section: Configuration section name
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Any: Configuration value or default
        """
        try:
            section_obj = getattr(self._config, section)
            return getattr(section_obj, key)
        except AttributeError:
            return default
    
    def set(self, section: str, key: str, value: Any) -> bool:
        """
        Set a specific configuration value.
        
        Args:
            section: Configuration section name
            key: Configuration key
            value: New value
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self._lock:
            try:
                section_obj = getattr(self._config, section)
                setattr(section_obj, key, value)
                self._notify_listeners(section)
                return True
            except AttributeError:
                return False
    
    def override_from_env(self, prefix: str = "DISPATCH_") -> None:
        """
        Override configuration from environment variables.
        
        Format: PREFIX_SECTION_KEY=value
        
        Args:
            prefix: Environment variable prefix
        """
        for env_name, env_value in os.environ.items():
            if env_name.startswith(prefix):
                # Remove prefix and split into section and key
                parts = env_name[len(prefix):].lower().split('_', 1)
                
                if len(parts) != 2:
                    continue
                    
                section, key = parts
                
                # Try to set the value
                try:
                    # Convert string value to appropriate type
                    orig_value = self.get(section, key)
                    
                    if orig_value is None:
                        continue
                        
                    if isinstance(orig_value, bool):
                        value = env_value.lower() in ('true', 'yes', '1', 'y')
                    elif isinstance(orig_value, int):
                        value = int(env_value)
                    elif isinstance(orig_value, float):
                        value = float(env_value)
                    else:
                        value = env_value
                    
                    self.set(section, key, value)
                    self._logger.info(f"Configuration override from environment: {section}.{key} = {value}")
                    
                except (AttributeError, ValueError) as e:
                    self._logger.warning(f"Failed to override {section}.{key} from environment: {str(e)}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> SystemConfig:
    """
    Get the global configuration.
    
    Returns:
        SystemConfig: Complete configuration
    """
    return config_manager.get_config()


def load_config(path: str) -> None:
    """
    Load configuration from a file.
    
    Args:
        path: Path to configuration file
    """
    config_manager.load_config(path)


def save_config(path: Optional[str] = None) -> None:
    """
    Save current configuration to a file.
    
    Args:
        path: Path to save configuration (uses current path if None)
    """
    config_manager.save_config(path)


def get(section: str, key: str, default: Any = None) -> Any:
    """
    Get a specific configuration value.
    
    Args:
        section: Configuration section name
        key: Configuration key
        default: Default value if key doesn't exist
        
    Returns:
        Any: Configuration value or default
    """
    return config_manager.get(section, key, default)


def set(section: str, key: str, value: Any) -> bool:
    """
    Set a specific configuration value.
    
    Args:
        section: Configuration section name
        key: Configuration key
        value: New value
        
    Returns:
        bool: True if successful, False otherwise
    """
    return config_manager.set(section, key, value)


def add_listener(listener: Callable[[str], None]) -> None:
    """
    Add a configuration change listener.
    
    Args:
        listener: Callback function that takes a section name parameter
    """
    config_manager.add_listener(listener)


def remove_listener(listener: Callable[[str], None]) -> None:
    """
    Remove a configuration change listener.
    
    Args:
        listener: Listener to remove
    """
    config_manager.remove_listener(listener)