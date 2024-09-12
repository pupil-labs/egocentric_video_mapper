"""Top-level entry-point for the egocentric_video_mapper package"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.egocentric_video_mapper")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["__version__"]
