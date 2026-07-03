"""Windows text injection via SendInput with KEYEVENTF_UNICODE.

Posts unicode key events directly to the focused window's input queue —
the canonical way to type arbitrary text system-wide without the
clipboard. Implemented with ctypes; no extra dependencies.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from wisperfree.inject.base import TextInjector

INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUT_UNION)]


class WindowsInjector(TextInjector):
    def _type_text(self, text: str) -> None:
        delay = self.config.inter_key_delay_ms / 1000.0
        send_input = ctypes.windll.user32.SendInput
        for char in text:
            for code_unit in _utf16_units(char):
                events = []
                for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
                    inp = INPUT()
                    inp.type = INPUT_KEYBOARD
                    inp.union.ki = KEYBDINPUT(0, code_unit, flags, 0, None)
                    events.append(inp)
                array = (INPUT * len(events))(*events)
                send_input(len(events), array, ctypes.sizeof(INPUT))
            if delay:
                time.sleep(delay)


def _utf16_units(char: str) -> list[int]:
    raw = char.encode("utf-16-le")
    return [int.from_bytes(raw[i : i + 2], "little") for i in range(0, len(raw), 2)]
