"""Pytest configuration file."""
import sys
import os

# Ensure the project root is at the front of sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
elif sys.path[0] != project_root:
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)

# If the stdlib platform module was already imported, remove it
# so pytest will import our local platform package instead
if "platform" in sys.modules:
    platform_module = sys.modules["platform"]
    # Check if it's the stdlib platform (not our package)
    if hasattr(platform_module, "__file__") and "llm-lab" not in str(platform_module.__file__):
        del sys.modules["platform"]
