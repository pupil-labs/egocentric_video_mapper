"""Top-level entry-point for the action_cam_mapper package"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.action_cam_mapper")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["__version__"]
