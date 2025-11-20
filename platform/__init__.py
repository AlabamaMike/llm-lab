"""LLM Experiment Platform package.

IMPORTANT: This package shadows Python's built-in 'platform' module.
We delegate to the real module for stdlib functions only.
"""
import sys as _sys
import importlib.util as _importlib_util

# Load the real built-in platform module directly from stdlib
_real_platform = None

# Find the stdlib platform module file
_stdlib_platform_file = "/usr/lib/python3.11/platform.py"

try:
    # Load it directly using importlib
    _spec = _importlib_util.spec_from_file_location("_stdlib_platform", _stdlib_platform_file)
    if _spec and _spec.loader:
        _real_platform = _importlib_util.module_from_spec(_spec)
        _spec.loader.exec_module(_real_platform)
except Exception:
    pass  # If we can't load it, just continue without delegation

__version__ = "0.1.0"

# Our own submodules/subpackages
_own_modules = {'api', 'core', 'models', 'providers', 'services', 'workers', 'ui', 'migrations'}

# Delegate to stdlib platform module for attributes it provides
def __getattr__(name):
    """Delegate to built-in platform module for stdlib functions, but not our own modules."""
    # First check if it's one of our own modules
    if name in _own_modules:
        # Let Python's normal import system handle it
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # Otherwise, try to delegate to the real platform module
    if _real_platform and hasattr(_real_platform, name):
        return getattr(_real_platform, name)

    # Not found anywhere
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
