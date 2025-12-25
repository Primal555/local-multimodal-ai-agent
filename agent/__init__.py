"""Core package for the Local Multimodal AI Agent."""

from importlib import metadata


def get_version() -> str:
    """Return package version if available."""
    try:
        return metadata.version("local_multimodal_agent")
    except metadata.PackageNotFoundError:
        return "0.1.0"
