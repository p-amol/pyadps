"""
pyadps.processing package

Core processing module for ADCP data quality control and output generation.

This package contains:
- ProcessedDataset: Main class for handling processed data
- ProcessingConfig: Configuration bridge between Streamlit INI and Python
- Core processing functions: sensor_health, signal_quality, velocity_test, profile_test
- Utilities: autoprocess, multifile, writenc
- Optional: xarray accessors for advanced usage
"""

from .core import ProcessedDataset
from .config import ProcessingConfig
from .signal_quality import SignalQualityRunner
from .utility import QCCheckStats, DataModificationStats, QCPipelineReport
from .utility import create_default_mask, replace_data

# Core processing modules (expose at package level for direct access if needed)
from . import sensor_health
from . import signal_quality
from . import velocity_check
from . import profile_operation
from . import autoprocess

# Optional: Accessors for xarray integration
try:
    from . import accessors
except ImportError:
    accessors = None

__version__ = "1.0.0"

__all__ = [
    # Main classes (primary API)
    "ProcessedDataset",
    "ProcessingConfig",
    "SignalQualityRunner",
    # Utilities
    "QCCheckStats",
    "DataModificationStats",
    "QCPipelineReport",
    "create_default_mask",
    "replace_data",
    # Core processing modules
    "sensor_health",
    "signal_quality",
    "profile_operation",
    "velocity_check",
    "autoprocess",
    # Optional
    "accessors",
]
