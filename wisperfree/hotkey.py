"""Configurable global hotkey (default Ctrl+Space) toggling dictation.

pynput's GlobalHotKeys covers macOS/Windows/X11. The Fn key is not
observable via portable APIs, hence the Ctrl+Space default; macOS users
who want Fn can map it to the dictation shortcut in System Settings.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)


class HotkeyListener:
    def __init__(self, combo: str, on_toggle: Callable[[], None]):
        self.combo = combo
        self.on_toggle = on_toggle
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard  # optional dependency

        self._listener = keyboard.GlobalHotKeys({self.combo: self.on_toggle})
        self._listener.start()
        log.info("global hotkey registered: %s", self.combo)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def rebind(self, combo: str) -> None:
        self.stop()
        self.combo = combo
        self.start()
