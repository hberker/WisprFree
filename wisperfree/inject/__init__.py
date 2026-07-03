"""System-wide text injection into the focused field (no clipboard hacks)."""

import sys

from wisperfree.config import InjectionConfig
from wisperfree.inject.base import TextInjector


def create_injector(config: InjectionConfig) -> TextInjector:
    method = config.method
    if method == "auto":
        method = {
            "darwin": "macos",
            "win32": "windows",
        }.get(sys.platform, "linux")
    if method == "macos":
        from wisperfree.inject.macos import MacOSInjector

        return MacOSInjector(config)
    if method == "windows":
        from wisperfree.inject.windows import WindowsInjector

        return WindowsInjector(config)
    if method == "linux":
        from wisperfree.inject.linux import LinuxInjector

        return LinuxInjector(config)
    raise ValueError(f"unknown injection method: {method!r}")


__all__ = ["TextInjector", "create_injector"]
