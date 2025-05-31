import logging
import sys
from pathlib import Path
from typing import Optional, Union

def setup_logging(log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file. If None, logs only to console
        level: Logging level (default: INFO)
    """
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler) 