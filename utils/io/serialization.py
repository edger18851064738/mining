"""
Serialization utilities for the mining dispatch system.

Provides functions for serializing and deserializing objects:
- JSON serialization
- YAML serialization
- Custom serialization for complex objects
- Type conversion helpers
"""

import json
import yaml
import pickle
import base64
import datetime
import uuid
import enum
import inspect
import dataclasses
from typing import Dict, List, Any, Optional, Union, Callable, Type, TypeVar, Generic
from pathlib import Path
import numpy as np

from utils.logger import get_logger
from utils.geo.coordinates import Point2D, Point3D, normalize_to_point2d
from utils.math.vectors import Vector2D, Vector3D
from utils.math.trajectories import Path, Trajectory

# Type variable for generic functions
T = TypeVar('T')

# Get logger
logger = get_logger("serialization")


class SerializationError(Exception):
    """Base exception for serialization errors."""
    pass


class DeserializationError(Exception):
    """Base exception for deserialization errors."""
    pass


class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for complex objects.
    
    Handles types like:
    - datetime, date, time
    - UUID
    - enums
    - paths
    - numpy arrays
    - custom objects with to_dict method
    - dataclasses
    """
    
    def default(self, obj):
        """Encode custom objects to JSON serializable types."""
        # Handle datetimes
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return {
                '__type__': obj.__class__.__name__,
                'iso': obj.isoformat()
            }
        
        # Handle UUID
        elif isinstance(obj, uuid.UUID):
            return {
                '__type__': 'UUID',
                'hex': obj.hex
            }
        
        # Handle enums
        elif isinstance(obj, enum.Enum):
            return {
                '__type__': 'Enum',
                'class': obj.__class__.__name__,
                'name': obj.name,
                'value': obj.value
            }
        
        # Handle paths
        elif isinstance(obj, Path):
            return {
                '__type__': 'Path',
                'path': str(obj)
            }
        
        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            return {
                '__type__': 'ndarray',
                'data': obj.tolist(),
                'dtype': str(obj.dtype),
                'shape': obj.shape
            }
        
        # Handle Point2D
        elif isinstance(obj, Point2D):
            return {
                '__type__': 'Point2D',
                'x': obj.x,
                'y': obj.y
            }
        
        # Handle Point3D
        elif isinstance(obj, Point3D):
            return {
                '__type__': 'Point3D',
                'x': obj.x,
                'y': obj.y,
                'z': obj.z
            }
        
        # Handle Vector2D
        elif isinstance(obj, Vector2D):
            return {
                '__type__': 'Vector2D',
                'x': obj.x,
                'y': obj.y
            }
        
        # Handle Vector3D
        elif isinstance(obj, Vector3D):
            return {
                '__type__': 'Vector3D',
                'x': obj.x,
                'y': obj.y,
                'z': obj.z
            }
        
        # Handle Path
        elif isinstance(obj, Path):
            return {
                '__type__': 'Path',
                'points': [{'x': p.x, 'y': p.y} for p in obj.points],
                'path_type': obj.path_type.name if hasattr(obj.path_type, 'name') else str(obj.path_type),
                'metadata': obj.metadata
            }
        
        # Handle Trajectory
        elif isinstance(obj, Trajectory):
            return {
                '__type__': 'Trajectory',
                'points': [{'x': p.x, 'y': p.y} for p in obj.points],
                'timestamps': obj.timestamps,
                'velocities': obj.velocities,
                'path_type': obj.path_type.name if hasattr(obj.path_type, 'name') else str(obj.path_type),
                'metadata': obj.metadata
            }
        
        # Handle objects with to_dict method
        elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
            result = obj.to_dict()
            result['__type__'] = obj.__class__.__name__
            return result
        
        # Handle dataclasses
        elif dataclasses.is_dataclass(obj):
            result = dataclasses.asdict(obj)
            result['__type__'] = obj.__class__.__name__
            return result
        
        # Handle sets (convert to list)
        elif isinstance(obj, set):
            return list(obj)
        
        # Let parent class raise TypeError
        return super().default(obj)


class JSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for complex objects.
    
    Reconstructs objects that were encoded with JSONEncoder.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the decoder with object hook."""
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
        
        # Store custom object types for reconstruction
        self.custom_types = {}
    
    def register_type(self, type_name: str, cls: Type[T]) -> None:
        """
        Register a custom type for deserialization.
        
        Args:
            type_name: Type name as stored in JSON
            cls: Class to instantiate
        """
        self.custom_types[type_name] = cls
    
    def object_hook(self, obj: Dict[str, Any]) -> Any:
        """Decode custom objects from their JSON representation."""
        # Check if this is a typed object
        if '__type__' in obj:
            type_name = obj['__type__']
            
            # Handle datetimes
            if type_name == 'datetime':
                return datetime.datetime.fromisoformat(obj['iso'])
            elif type_name == 'date':
                return datetime.date.fromisoformat(obj['iso'])
            elif type_name == 'time':
                return datetime.time.fromisoformat(obj['iso'])
            
            # Handle UUID
            elif type_name == 'UUID':
                return uuid.UUID(obj['hex'])
            
            # Handle enums
            elif type_name == 'Enum':
                # This requires the enum class to be available
                try:
                    enum_class = globals().get(obj['class'])
                    if enum_class and issubclass(enum_class, enum.Enum):
                        return enum_class[obj['name']]
                except (KeyError, AttributeError):
                    logger.warning(f"Could not deserialize enum {obj['class']}.{obj['name']}")
            
            # Handle Path
            elif type_name == 'Path':
                return Path(obj['path'])
            
            # Handle numpy arrays
            elif type_name == 'ndarray':
                return np.array(obj['data'], dtype=obj['dtype'])
            
            # Handle Point2D
            elif type_name == 'Point2D':
                return Point2D(obj['x'], obj['y'])
            
            # Handle Point3D
            elif type_name == 'Point3D':
                return Point3D(obj['x'], obj['y'], obj['z'])
            
            # Handle Vector2D
            elif type_name == 'Vector2D':
                return Vector2D(obj['x'], obj['y'])
            
            # Handle Vector3D
            elif type_name == 'Vector3D':
                return Vector3D(obj['x'], obj['y'], obj['z'])
            
            # Handle Path
            elif type_name == 'Path':
                from utils.math.trajectories import PathType
                
                # Convert points
                points = [Point2D(p['x'], p['y']) for p in obj['points']]
                
                # Parse path type
                try:
                    path_type = PathType[obj['path_type']]
                except (KeyError, ValueError):
                    path_type = PathType.STRAIGHT_LINE
                
                return Path(points, path_type, obj.get('metadata', {}))
            
            # Handle Trajectory
            elif type_name == 'Trajectory':
                from utils.math.trajectories import PathType
                
                # Convert points
                points = [Point2D(p['x'], p['y']) for p in obj['points']]
                
                # Parse path type
                try:
                    path_type = PathType[obj['path_type']]
                except (KeyError, ValueError):
                    path_type = PathType.STRAIGHT_LINE
                
                return Trajectory(
                    points, 
                    obj.get('timestamps', []), 
                    obj.get('velocities', []),
                    path_type, 
                    obj.get('metadata', {})
                )
            
            # Check registered custom types
            elif type_name in self.custom_types:
                cls = self.custom_types[type_name]
                
                # Handle dataclasses
                if dataclasses.is_dataclass(cls):
                    # Remove __type__ key
                    obj_copy = obj.copy()
                    obj_copy.pop('__type__', None)
                    
                    # Create instance from dict
                    return cls(**obj_copy)
                
                # Handle objects with from_dict method
                elif hasattr(cls, 'from_dict') and callable(getattr(cls, 'from_dict')):
                    # Remove __type__ key
                    obj_copy = obj.copy()
                    obj_copy.pop('__type__', None)
                    
                    # Create instance from dict
                    return cls.from_dict(obj_copy)
                
                # Try to create a new instance and set attributes
                try:
                    instance = cls()
                    for key, value in obj.items():
                        if key != '__type__' and hasattr(instance, key):
                            setattr(instance, key, value)
                    return instance
                except Exception as e:
                    logger.warning(f"Could not deserialize {type_name}: {str(e)}")
        
        # Return unmodified object
        return obj


def to_json(obj: Any, pretty: bool = False, ensure_ascii: bool = False) -> str:
    """
    Serialize an object to JSON string.
    
    Args:
        obj: Object to serialize
        pretty: Whether to format with indentation
        ensure_ascii: Whether to escape non-ASCII characters
        
    Returns:
        str: JSON string
        
    Raises:
        SerializationError: If object can't be serialized
    """
    try:
        indent = 4 if pretty else None
        return json.dumps(obj, cls=JSONEncoder, indent=indent, ensure_ascii=ensure_ascii)
    except Exception as e:
        raise SerializationError(f"Failed to serialize to JSON: {str(e)}") from e


def from_json(json_str: str, custom_types: Dict[str, Type] = None) -> Any:
    """
    Deserialize an object from JSON string.
    
    Args:
        json_str: JSON string
        custom_types: Dictionary mapping type names to classes
        
    Returns:
        Any: Deserialized object
        
    Raises:
        DeserializationError: If JSON can't be deserialized
    """
    try:
        decoder = JSONDecoder()
        
        # Register custom types
        if custom_types:
            for type_name, cls in custom_types.items():
                decoder.register_type(type_name, cls)
        
        return decoder.decode(json_str)
    except Exception as e:
        raise DeserializationError(f"Failed to deserialize from JSON: {str(e)}") from e


def to_yaml(obj: Any, flow_style: bool = False) -> str:
    """
    Serialize an object to YAML string.
    
    Args:
        obj: Object to serialize
        flow_style: Whether to use flow style for collections
        
    Returns:
        str: YAML string
        
    Raises:
        SerializationError: If object can't be serialized
    """
    class CustomYAMLDumper(yaml.SafeDumper):
        pass
    
    def represent_complex_objects(dumper, data):
        """Represent complex objects as mapping."""
        if hasattr(data, 'to_dict') and callable(data.to_dict):
            result = data.to_dict()
            result['__type__'] = data.__class__.__name__
            return dumper.represent_mapping('tag:yaml.org,2002:map', result)
        elif dataclasses.is_dataclass(data):
            result = dataclasses.asdict(data)
            result['__type__'] = data.__class__.__name__
            return dumper.represent_mapping('tag:yaml.org,2002:map', result)
        
        # Default representation for other objects
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))
    
    # Register representers
    CustomYAMLDumper.add_representer(object, represent_complex_objects)
    
    try:
        # Convert to JSON-compatible structure first
        json_str = to_json(obj)
        json_obj = json.loads(json_str)
        
        # Convert to YAML
        return yaml.dump(json_obj, Dumper=CustomYAMLDumper, default_flow_style=flow_style)
    except Exception as e:
        raise SerializationError(f"Failed to serialize to YAML: {str(e)}") from e


def from_yaml(yaml_str: str, custom_types: Dict[str, Type] = None) -> Any:
    """
    Deserialize an object from YAML string.
    
    Args:
        yaml_str: YAML string
        custom_types: Dictionary mapping type names to classes
        
    Returns:
        Any: Deserialized object
        
    Raises:
        DeserializationError: If YAML can't be deserialized
    """
    try:
        # Parse YAML to dict
        data = yaml.safe_load(yaml_str)
        
        # Convert to JSON string
        json_str = json.dumps(data)
        
        # Use JSON decoder
        return from_json(json_str, custom_types)
    except Exception as e:
        raise DeserializationError(f"Failed to deserialize from YAML: {str(e)}") from e


def to_pickle(obj: Any) -> bytes:
    """
    Serialize an object to pickle format.
    
    Args:
        obj: Object to serialize
        
    Returns:
        bytes: Pickled data
        
    Raises:
        SerializationError: If object can't be pickled
    """
    try:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise SerializationError(f"Failed to serialize to pickle: {str(e)}") from e


def from_pickle(data: bytes) -> Any:
    """
    Deserialize an object from pickle data.
    
    Args:
        data: Pickled data
        
    Returns:
        Any: Deserialized object
        
    Raises:
        DeserializationError: If data can't be unpickled
    """
    try:
        return pickle.loads(data)
    except Exception as e:
        raise DeserializationError(f"Failed to deserialize from pickle: {str(e)}") from e


def to_base64(obj: Any) -> str:
    """
    Serialize an object to base64-encoded string.
    
    Args:
        obj: Object to serialize
        
    Returns:
        str: Base64-encoded string
        
    Raises:
        SerializationError: If object can't be serialized
    """
    try:
        pickle_data = to_pickle(obj)
        return base64.b64encode(pickle_data).decode('ascii')
    except Exception as e:
        raise SerializationError(f"Failed to serialize to base64: {str(e)}") from e


def from_base64(data: str) -> Any:
    """
    Deserialize an object from base64-encoded string.
    
    Args:
        data: Base64-encoded string
        
    Returns:
        Any: Deserialized object
        
    Raises:
        DeserializationError: If data can't be deserialized
    """
    try:
        pickle_data = base64.b64decode(data.encode('ascii'))
        return from_pickle(pickle_data)
    except Exception as e:
        raise DeserializationError(f"Failed to deserialize from base64: {str(e)}") from e


def serialize_numpy_array(arr: np.ndarray) -> Dict[str, Any]:
    """
    Serialize a numpy array to a dictionary.
    
    Args:
        arr: Numpy array
        
    Returns:
        Dict[str, Any]: Serialized representation
    """
    return {
        'data': arr.tolist(),
        'dtype': str(arr.dtype),
        'shape': arr.shape
    }


def deserialize_numpy_array(data: Dict[str, Any]) -> np.ndarray:
    """
    Deserialize a numpy array from a dictionary.
    
    Args:
        data: Serialized representation
        
    Returns:
        np.ndarray: Numpy array
    """
    return np.array(data['data'], dtype=data['dtype'])


def is_serializable(obj: Any) -> bool:
    """
    Check if an object is JSON serializable.
    
    Args:
        obj: Object to check
        
    Returns:
        bool: True if object is serializable, False otherwise
    """
    try:
        json.dumps(obj, cls=JSONEncoder)
        return True
    except (TypeError, OverflowError):
        return False


def make_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON serializable form.
    
    Args:
        obj: Object to convert
        
    Returns:
        Any: JSON serializable representation
    """
    if is_serializable(obj):
        return obj
    
    # Try to convert to dictionary
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        return obj.to_dict()
    elif dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    
    # Convert common types
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, enum.Enum):
        return obj.name
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode('ascii')
    
    # Last resort: convert to string
    return str(obj)


def to_serializable_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert an object to a serializable dictionary.
    
    Args:
        obj: Object to convert
        
    Returns:
        Dict[str, Any]: Serializable dictionary
    """
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    
    # Try to convert to dictionary
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        result = obj.to_dict()
    elif dataclasses.is_dataclass(obj):
        result = dataclasses.asdict(obj)
    else:
        # Get all public attributes
        result = {}
        for key, value in inspect.getmembers(obj):
            # Skip methods, private attributes, and built-ins
            if (not key.startswith('_') and
                not inspect.ismethod(value) and
                not inspect.isfunction(value) and
                not inspect.isbuiltin(value)):
                result[key] = value
    
    # Make all values serializable
    return {key: make_serializable(value) for key, value in result.items()}


class Serializable:
    """
    Mixin class for serializable objects.
    
    Provides methods for serialization and deserialization.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert object to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return to_serializable_dict(self)
    
    def to_json(self, pretty: bool = False) -> str:
        """
        Serialize object to JSON string.
        
        Args:
            pretty: Whether to format with indentation
            
        Returns:
            str: JSON string
        """
        return to_json(self, pretty)
    
    def to_yaml(self) -> str:
        """
        Serialize object to YAML string.
        
        Returns:
            str: YAML string
        """
        return to_yaml(self)
    
    def to_pickle(self) -> bytes:
        """
        Serialize object to pickle format.
        
        Returns:
            bytes: Pickled data
        """
        return to_pickle(self)
    
    def to_base64(self) -> str:
        """
        Serialize object to base64-encoded string.
        
        Returns:
            str: Base64-encoded string
        """
        return to_base64(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """
        Create object from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Serializable: New object
        """
        instance = cls()
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Serializable':
        """
        Deserialize object from JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            Serializable: New object
        """
        data = from_json(json_str, {cls.__name__: cls})
        return data
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Serializable':
        """
        Deserialize object from YAML string.
        
        Args:
            yaml_str: YAML string
            
        Returns:
            Serializable: New object
        """
        data = from_yaml(yaml_str, {cls.__name__: cls})
        return data
    
    @classmethod
    def from_pickle(cls, data: bytes) -> 'Serializable':
        """
        Deserialize object from pickle data.
        
        Args:
            data: Pickled data
            
        Returns:
            Serializable: New object
        """
        return from_pickle(data)
    
    @classmethod
    def from_base64(cls, data: str) -> 'Serializable':
        """
        Deserialize object from base64-encoded string.
        
        Args:
            data: Base64-encoded string
            
        Returns:
            Serializable: New object
        """
        return from_base64(data)


# Register custom types for deserialization
def register_custom_types():
    """Register custom types for the default JSONDecoder."""
    from utils.math.trajectories import PathType, Path, Trajectory
    
    custom_types = {
        'Point2D': Point2D,
        'Point3D': Point3D,
        'Vector2D': Vector2D,
        'Vector3D': Vector3D,
        'Path': Path,
        'Trajectory': Trajectory,
        'PathType': PathType
    }
    
    return custom_types