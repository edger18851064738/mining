"""
File utility functions for the mining dispatch system.

Provides functions for common file operations:
- Safe file reading and writing
- File format detection and validation
- Path manipulation
- Directory operations
"""

import os
import sys
import csv
import json
import yaml
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, BinaryIO, TextIO, Callable, Iterator, TypeVar, Generic
import io
import time
import hashlib
import zipfile
import threading
from contextlib import contextmanager

from utils.logger import get_logger

# Type variable for generic functions
T = TypeVar('T')

# Get logger
logger = get_logger("file_utils")


class FileError(Exception):
    """Base exception for file operation errors."""
    pass


class FileNotFoundError(FileError):
    """Exception raised when a file doesn't exist."""
    pass


class FileAccessError(FileError):
    """Exception raised when a file can't be accessed."""
    pass


class FileFormatError(FileError):
    """Exception raised when a file has an invalid format."""
    pass


class FileWriteError(FileError):
    """Exception raised when a file can't be written."""
    pass


class FileLockError(FileError):
    """Exception raised when a file can't be locked."""
    pass


@contextmanager
def safe_open(file_path: Union[str, Path], mode: str = 'r', encoding: Optional[str] = None, 
            **kwargs) -> Iterator[Union[TextIO, BinaryIO]]:
    """
    Context manager for safely opening files with error handling.
    
    Args:
        file_path: Path to the file
        mode: File open mode
        encoding: File encoding (for text modes)
        **kwargs: Additional arguments for open()
        
    Yields:
        Union[TextIO, BinaryIO]: File object
        
    Raises:
        FileNotFoundError: If file doesn't exist and 'r' mode
        FileAccessError: If file can't be accessed
    """
    file_obj = None
    try:
        # Ensure directory exists for write modes
        if 'w' in mode or 'a' in mode or 'x' in mode:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        # Open the file
        file_obj = open(file_path, mode, encoding=encoding, **kwargs)
        yield file_obj
    except PermissionError as e:
        raise FileAccessError(f"Permission denied: {file_path}") from e
    except IOError as e:
        if not os.path.exists(file_path) and 'r' in mode:
            raise FileNotFoundError(f"File not found: {file_path}") from e
        raise FileAccessError(f"I/O error: {file_path} - {str(e)}") from e
    finally:
        if file_obj:
            file_obj.close()


def read_text_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    Read text from a file.
    
    Args:
        file_path: Path to the file
        encoding: File encoding
        
    Returns:
        str: File contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
        FileAccessError: If file can't be accessed
    """
    with safe_open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def write_text_file(file_path: Union[str, Path], content: str, 
                   encoding: str = 'utf-8', create_dirs: bool = True) -> None:
    """
    Write text to a file safely.
    
    Args:
        file_path: Path to the file
        content: Text content to write
        encoding: File encoding
        create_dirs: Create parent directories if they don't exist
        
    Raises:
        FileWriteError: If file can't be written
    """
    try:
        # Ensure directory exists
        if create_dirs:
            directory = os.path.dirname(os.path.abspath(file_path))
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        # Write to a temporary file first
        temp_file = tempfile.NamedTemporaryFile(mode='w', encoding=encoding, 
                                               dir=os.path.dirname(os.path.abspath(file_path)), 
                                               delete=False)
        try:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        finally:
            temp_file.close()
        
        # Rename the temporary file (atomic operation)
        shutil.move(temp_file.name, file_path)
        
    except Exception as e:
        # Clean up temporary file if it exists
        try:
            if 'temp_file' in locals() and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception:
            pass
            
        raise FileWriteError(f"Failed to write to {file_path}: {str(e)}") from e


def read_binary_file(file_path: Union[str, Path]) -> bytes:
    """
    Read binary data from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bytes: File contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
        FileAccessError: If file can't be accessed
    """
    with safe_open(file_path, 'rb') as f:
        return f.read()


def write_binary_file(file_path: Union[str, Path], data: bytes, 
                     create_dirs: bool = True) -> None:
    """
    Write binary data to a file safely.
    
    Args:
        file_path: Path to the file
        data: Binary data to write
        create_dirs: Create parent directories if they don't exist
        
    Raises:
        FileWriteError: If file can't be written
    """
    try:
        # Ensure directory exists
        if create_dirs:
            directory = os.path.dirname(os.path.abspath(file_path))
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        # Write to a temporary file first
        temp_file = tempfile.NamedTemporaryFile(mode='wb', 
                                               dir=os.path.dirname(os.path.abspath(file_path)), 
                                               delete=False)
        try:
            temp_file.write(data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        finally:
            temp_file.close()
        
        # Rename the temporary file (atomic operation)
        shutil.move(temp_file.name, file_path)
        
    except Exception as e:
        # Clean up temporary file if it exists
        try:
            if 'temp_file' in locals() and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception:
            pass
            
        raise FileWriteError(f"Failed to write to {file_path}: {str(e)}") from e


def read_json_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Read JSON from a file.
    
    Args:
        file_path: Path to the file
        encoding: File encoding
        
    Returns:
        Dict[str, Any]: Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        FileAccessError: If file can't be accessed
        FileFormatError: If JSON is invalid
    """
    try:
        with safe_open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise FileFormatError(f"Invalid JSON format in {file_path}: {str(e)}") from e


def write_json_file(file_path: Union[str, Path], data: Any, 
                   indent: int = 4, encoding: str = 'utf-8') -> None:
    """
    Write data to a JSON file safely.
    
    Args:
        file_path: Path to the file
        data: Data to write (must be JSON serializable)
        indent: Indentation spaces
        encoding: File encoding
        
    Raises:
        FileWriteError: If file can't be written
    """
    try:
        # Convert to JSON
        json_content = json.dumps(data, indent=indent, ensure_ascii=False)
        
        # Write to file
        write_text_file(file_path, json_content, encoding=encoding)
        
    except TypeError as e:
        raise FileWriteError(f"Cannot serialize to JSON: {str(e)}") from e
    except Exception as e:
        raise FileWriteError(f"Failed to write JSON to {file_path}: {str(e)}") from e


def read_yaml_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Read YAML from a file.
    
    Args:
        file_path: Path to the file
        encoding: File encoding
        
    Returns:
        Dict[str, Any]: Parsed YAML data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        FileAccessError: If file can't be accessed
        FileFormatError: If YAML is invalid
    """
    try:
        with safe_open(file_path, 'r', encoding=encoding) as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise FileFormatError(f"Invalid YAML format in {file_path}: {str(e)}") from e


def write_yaml_file(file_path: Union[str, Path], data: Any, 
                    encoding: str = 'utf-8') -> None:
    """
    Write data to a YAML file safely.
    
    Args:
        file_path: Path to the file
        data: Data to write (must be YAML serializable)
        encoding: File encoding
        
    Raises:
        FileWriteError: If file can't be written
    """
    try:
        # Convert to YAML
        yaml_content = yaml.safe_dump(data, default_flow_style=False)
        
        # Write to file
        write_text_file(file_path, yaml_content, encoding=encoding)
        
    except yaml.YAMLError as e:
        raise FileWriteError(f"Cannot serialize to YAML: {str(e)}") from e
    except Exception as e:
        raise FileWriteError(f"Failed to write YAML to {file_path}: {str(e)}") from e


def read_csv_file(file_path: Union[str, Path], delimiter: str = ',', 
                 has_header: bool = True, encoding: str = 'utf-8') -> List[Dict[str, str]]:
    """
    Read CSV data from a file.
    
    Args:
        file_path: Path to the file
        delimiter: CSV delimiter
        has_header: Whether the CSV has a header row
        encoding: File encoding
        
    Returns:
        List[Dict[str, str]]: Parsed CSV data as list of dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
        FileAccessError: If file can't be accessed
        FileFormatError: If CSV is invalid
    """
    try:
        with safe_open(file_path, 'r', encoding=encoding, newline='') as f:
            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
                return list(reader)
            else:
                reader = csv.reader(f, delimiter=delimiter)
                data = list(reader)
                
                # Create dictionaries with numeric column names
                return [
                    {str(i): cell for i, cell in enumerate(row)}
                    for row in data
                ]
    except csv.Error as e:
        raise FileFormatError(f"Invalid CSV format in {file_path}: {str(e)}") from e


def write_csv_file(file_path: Union[str, Path], data: List[Dict[str, Any]], 
                  fieldnames: Optional[List[str]] = None, delimiter: str = ',', 
                  encoding: str = 'utf-8') -> None:
    """
    Write data to a CSV file safely.
    
    Args:
        file_path: Path to the file
        data: List of row dictionaries
        fieldnames: List of field names (columns)
        delimiter: CSV delimiter
        encoding: File encoding
        
    Raises:
        FileWriteError: If file can't be written
    """
    try:
        # Determine field names if not provided
        if fieldnames is None and data:
            fieldnames = list(data[0].keys())
        
        # Create temporary file for writing
        directory = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(directory, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(mode='w', encoding=encoding, 
                                               newline='', dir=directory, 
                                               delete=False)
        
        try:
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        finally:
            temp_file.close()
        
        # Rename the temporary file (atomic operation)
        shutil.move(temp_file.name, file_path)
        
    except Exception as e:
        # Clean up temporary file if it exists
        try:
            if 'temp_file' in locals() and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception:
            pass
            
        raise FileWriteError(f"Failed to write CSV to {file_path}: {str(e)}") from e


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the file extension without the dot.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: File extension in lowercase
    """
    return os.path.splitext(str(file_path))[1].lower().lstrip('.')


def get_mime_type(file_path: Union[str, Path]) -> str:
    """
    Get the MIME type of a file based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: MIME type
    """
    extension = get_file_extension(file_path)
    
    # Map of common extensions to MIME types
    mime_map = {
        'txt': 'text/plain',
        'html': 'text/html',
        'css': 'text/css',
        'js': 'application/javascript',
        'json': 'application/json',
        'xml': 'application/xml',
        'yaml': 'application/x-yaml',
        'yml': 'application/x-yaml',
        'csv': 'text/csv',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'svg': 'image/svg+xml',
        'pdf': 'application/pdf',
        'zip': 'application/zip',
        'gz': 'application/gzip',
        'tar': 'application/x-tar',
        'mp3': 'audio/mpeg',
        'mp4': 'video/mp4',
        'wav': 'audio/wav',
        'py': 'text/x-python',
        'cpp': 'text/x-c++',
        'c': 'text/x-c',
        'h': 'text/x-c',
        'hpp': 'text/x-c++',
        'md': 'text/markdown',
    }
    
    return mime_map.get(extension, 'application/octet-stream')


def detect_file_format(file_path: Union[str, Path]) -> str:
    """
    Detect the format of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: File format type ('json', 'yaml', 'csv', 'text', 'binary', 'unknown')
    """
    extension = get_file_extension(file_path)
    
    # Check by extension first
    if extension in ['json']:
        return 'json'
    elif extension in ['yaml', 'yml']:
        return 'yaml'
    elif extension in ['csv', 'tsv']:
        return 'csv'
    elif extension in ['txt', 'md', 'py', 'js', 'html', 'css', 'cpp', 'c', 'h', 'hpp', 'xml']:
        return 'text'
    
    # If extension is ambiguous, check content
    try:
        with open(file_path, 'rb') as f:
            # Read a small sample of the file
            sample = f.read(min(1024, os.path.getsize(file_path)))
            
            # Check if it's text
            try:
                sample.decode('utf-8')
                
                # Check for JSON format
                if sample.strip().startswith(b'{') or sample.strip().startswith(b'['):
                    try:
                        json.loads(sample.decode('utf-8'))
                        return 'json'
                    except json.JSONDecodeError:
                        pass
                
                # Check for YAML format
                if (b'---' in sample or b':' in sample) and not b',' in sample:
                    try:
                        yaml.safe_load(sample.decode('utf-8'))
                        return 'yaml'
                    except yaml.YAMLError:
                        pass
                
                # Check for CSV format
                if b',' in sample and b'\n' in sample:
                    lines = sample.split(b'\n')
                    if len(lines) > 1:
                        # Check if all rows have the same number of columns
                        comma_counts = [line.count(b',') for line in lines if line.strip()]
                        if len(set(comma_counts)) <= 1 and comma_counts[0] > 0:
                            return 'csv'
                
                return 'text'
            except UnicodeDecodeError:
                return 'binary'
    except Exception:
        return 'unknown'


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Calculate the hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('sha256', 'sha1', 'md5')
        
    Returns:
        str: Hex digest of the hash
        
    Raises:
        FileNotFoundError: If file doesn't exist
        FileAccessError: If file can't be accessed
    """
    # Choose hash function
    if algorithm == 'md5':
        hash_func = hashlib.md5()
    elif algorithm == 'sha1':
        hash_func = hashlib.sha1()
    else:
        hash_func = hashlib.sha256()
    
    try:
        with safe_open(file_path, 'rb') as f:
            # Update hash in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        
        # Return hex digest
        return hash_func.hexdigest()
    except Exception as e:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}") from e
        raise FileAccessError(f"Cannot access file {file_path}: {str(e)}") from e


def create_directory(directory_path: Union[str, Path], exists_ok: bool = True) -> None:
    """
    Create a directory, including parent directories.
    
    Args:
        directory_path: Path to the directory
        exists_ok: Don't error if directory already exists
        
    Raises:
        FileAccessError: If directory can't be created
    """
    try:
        os.makedirs(directory_path, exist_ok=exists_ok)
    except PermissionError as e:
        raise FileAccessError(f"Permission denied: {directory_path}") from e
    except OSError as e:
        raise FileAccessError(f"Failed to create directory {directory_path}: {str(e)}") from e


def list_files(directory_path: Union[str, Path], pattern: str = '*', 
              recursive: bool = False) -> List[str]:
    """
    List files matching a pattern.
    
    Args:
        directory_path: Path to the directory
        pattern: Glob pattern
        recursive: Whether to search recursively
        
    Returns:
        List[str]: List of file paths
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        FileAccessError: If directory can't be accessed
    """
    try:
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if recursive:
            glob_pattern = f"**/{pattern}"
        else:
            glob_pattern = pattern
        
        return [str(path) for path in directory.glob(glob_pattern) if path.is_file()]
    except PermissionError as e:
        raise FileAccessError(f"Permission denied: {directory_path}") from e
    except OSError as e:
        raise FileAccessError(f"Failed to list files in {directory_path}: {str(e)}") from e


def copy_file(source_path: Union[str, Path], target_path: Union[str, Path], 
             overwrite: bool = True) -> None:
    """
    Copy a file safely.
    
    Args:
        source_path: Path to the source file
        target_path: Path to the target file
        overwrite: Whether to overwrite existing target
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        FileAccessError: If copy operation fails
    """
    try:
        # Check if source exists
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Check if target exists and overwrite is False
        if os.path.exists(target_path) and not overwrite:
            return
        
        # Ensure target directory exists
        target_dir = os.path.dirname(os.path.abspath(target_path))
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy to a temporary file first
        temp_file = tempfile.NamedTemporaryFile(dir=target_dir, delete=False)
        temp_file.close()
        
        # Copy the file
        shutil.copy2(source_path, temp_file.name)
        
        # Rename to target (atomic operation)
        shutil.move(temp_file.name, target_path)
        
    except PermissionError as e:
        try:
            if 'temp_file' in locals() and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception:
            pass
        raise FileAccessError(f"Permission denied: {source_path} -> {target_path}") from e
    except OSError as e:
        try:
            if 'temp_file' in locals() and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception:
            pass
        raise FileAccessError(f"Failed to copy {source_path} to {target_path}: {str(e)}") from e


def move_file(source_path: Union[str, Path], target_path: Union[str, Path], 
             overwrite: bool = True) -> None:
    """
    Move a file safely.
    
    Args:
        source_path: Path to the source file
        target_path: Path to the target file
        overwrite: Whether to overwrite existing target
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        FileAccessError: If move operation fails
    """
    try:
        # Check if source exists
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Check if target exists and overwrite is False
        if os.path.exists(target_path) and not overwrite:
            return
        
        # Ensure target directory exists
        target_dir = os.path.dirname(os.path.abspath(target_path))
        os.makedirs(target_dir, exist_ok=True)
        
        # If target exists and overwrite is True, remove it
        if os.path.exists(target_path) and overwrite:
            os.unlink(target_path)
        
        # Move the file
        shutil.move(source_path, target_path)
        
    except PermissionError as e:
        raise FileAccessError(f"Permission denied: {source_path} -> {target_path}") from e
    except OSError as e:
        raise FileAccessError(f"Failed to move {source_path} to {target_path}: {str(e)}") from e


def delete_file(file_path: Union[str, Path], ignore_missing: bool = True) -> bool:
    """
    Delete a file.
    
    Args:
        file_path: Path to the file
        ignore_missing: Don't error if file doesn't exist
        
    Returns:
        bool: True if file was deleted, False if it didn't exist
        
    Raises:
        FileNotFoundError: If file doesn't exist and ignore_missing is False
        FileAccessError: If file can't be deleted
    """
    try:
        if not os.path.exists(file_path):
            if ignore_missing:
                return False
            raise FileNotFoundError(f"File not found: {file_path}")
        
        os.unlink(file_path)
        return True
        
    except PermissionError as e:
        raise FileAccessError(f"Permission denied: {file_path}") from e
    except OSError as e:
        raise FileAccessError(f"Failed to delete {file_path}: {str(e)}") from e


def create_zip_archive(file_paths: List[Union[str, Path]], 
                      archive_path: Union[str, Path]) -> None:
    """
    Create a ZIP archive containing specified files.
    
    Args:
        file_paths: List of files to include
        archive_path: Path to the output ZIP file
        
    Raises:
        FileNotFoundError: If a source file doesn't exist
        FileAccessError: If archive can't be created
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(archive_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a temporary file for the ZIP
        temp_file = tempfile.NamedTemporaryFile(dir=output_dir, suffix='.zip', delete=False)
        temp_file.close()
        
        # Create the ZIP file
        with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                # Add the file to the ZIP
                zipf.write(file_path, os.path.basename(file_path))
        
        # Move the temporary file to the target path
        shutil.move(temp_file.name, archive_path)
        
    except PermissionError as e:
        try:
            if 'temp_file' in locals() and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception:
            pass
        raise FileAccessError(f"Permission denied: {archive_path}") from e
    except OSError as e:
        try:
            if 'temp_file' in locals() and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception:
            pass
        raise FileAccessError(f"Failed to create ZIP archive {archive_path}: {str(e)}") from e


def extract_zip_archive(archive_path: Union[str, Path], 
                       target_dir: Union[str, Path]) -> List[str]:
    """
    Extract a ZIP archive.
    
    Args:
        archive_path: Path to the ZIP file
        target_dir: Directory to extract to
        
    Returns:
        List[str]: List of extracted file paths
        
    Raises:
        FileNotFoundError: If archive doesn't exist
        FileAccessError: If archive can't be extracted
        FileFormatError: If archive is invalid
    """
    try:
        # Check if archive exists
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        
        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        # Extract the ZIP file
        extracted_files = []
        
        with zipfile.ZipFile(archive_path, 'r') as zipf:
            zipf.extractall(target_dir)
            extracted_files = zipf.namelist()
        
        # Return full paths to extracted files
        return [os.path.join(target_dir, filename) for filename in extracted_files]
        
    except zipfile.BadZipFile as e:
        raise FileFormatError(f"Invalid ZIP archive: {archive_path}") from e
    except PermissionError as e:
        raise FileAccessError(f"Permission denied: {archive_path} -> {target_dir}") from e
    except OSError as e:
        raise FileAccessError(f"Failed to extract ZIP archive {archive_path}: {str(e)}") from e


class FileLock:
    """
    File-based locking mechanism for coordinating access between processes.
    """
    
    def __init__(self, lock_file: Union[str, Path], timeout: float = 10.0, 
                retry_interval: float = 0.1):
        """
        Initialize a file lock.
        
        Args:
            lock_file: Path to the lock file
            timeout: Maximum wait time in seconds
            retry_interval: Time between retries in seconds
        """
        self.lock_file = str(lock_file)
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.locked = False
    
    def acquire(self) -> bool:
        """
        Acquire the lock.
        
        Returns:
            bool: True if lock was acquired, False on timeout
            
        Raises:
            FileLockError: If lock can't be acquired
        """
        start_time = time.time()
        
        while True:
            try:
                # Try to create the lock file
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                
                # Write PID to lock file
                with os.fdopen(fd, 'w') as f:
                    f.write(f"{os.getpid()}")
                
                self.locked = True
                return True
                
            except OSError:
                # Lock file already exists
                # Check if process that created it is still alive
                try:
                    with open(self.lock_file, 'r') as f:
                        pid = int(f.read().strip())
                    
                    # Check if process is still running
                    try:
                        # For Windows, this raises an exception if process doesn't exist
                        if sys.platform == 'win32':
                            import ctypes
                            kernel32 = ctypes.windll.kernel32
                            handle = kernel32.OpenProcess(1, 0, pid)
                            if handle == 0:
                                # Process doesn't exist
                                os.unlink(self.lock_file)
                                continue
                            kernel32.CloseHandle(handle)
                        else:
                            # For Unix, sending signal 0 checks process existence
                            os.kill(pid, 0)
                    except (OSError, PermissionError, ProcessLookupError):
                        # Process doesn't exist, remove stale lock
                        try:
                            os.unlink(self.lock_file)
                            continue
                        except OSError:
                            pass
                except (IOError, ValueError):
                    # Lock file exists but is invalid, try to remove it
                    try:
                        os.unlink(self.lock_file)
                        continue
                    except OSError:
                        pass
                
                # Check timeout
                if time.time() - start_time > self.timeout:
                    return False
                
                # Wait before retry
                time.sleep(self.retry_interval)
        
        return False
    
    def release(self) -> None:
        """
        Release the lock.
        
        Raises:
            FileLockError: If lock can't be released
        """
        if self.locked:
            try:
                os.unlink(self.lock_file)
                self.locked = False
            except OSError as e:
                raise FileLockError(f"Failed to release lock {self.lock_file}: {str(e)}") from e
    
    def __enter__(self) -> 'FileLock':
        """Context manager entry."""
        if not self.acquire():
            raise FileLockError(f"Timeout acquiring lock: {self.lock_file}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()


@contextmanager
def file_lock(lock_file: Union[str, Path], timeout: float = 10.0, 
             retry_interval: float = 0.1) -> Iterator[None]:
    """
    Context manager for file locking.
    
    Args:
        lock_file: Path to the lock file
        timeout: Maximum wait time in seconds
        retry_interval: Time between retries in seconds
        
    Yields:
        None
        
    Raises:
        FileLockError: If lock can't be acquired
    """
    lock = FileLock(lock_file, timeout, retry_interval)
    try:
        if not lock.acquire():
            raise FileLockError(f"Timeout acquiring lock: {lock_file}")
        yield
    finally:
        lock.release()