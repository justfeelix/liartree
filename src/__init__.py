import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
BASEFRAMEWORK_DIR = os.path.join(PROJECT_ROOT, "baseframework")


def setup_baseframework_path():
    """Make the baseframework submodule importable.
    """
    for sub in ("", "metrics", "offline_clustering"):
        path = os.path.join(BASEFRAMEWORK_DIR, sub) if sub else BASEFRAMEWORK_DIR
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


setup_baseframework_path()
