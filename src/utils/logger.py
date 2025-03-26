from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"
    

def log(message: str, level: LogLevel = LogLevel.INFO):
    """Log message with different level and timestamp

    Args:
        message (str): Log message
        level (str, optional): Log level. Defaults to "INFO".
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level.name}] >>> {message}")
