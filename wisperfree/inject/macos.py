"""macOS text injection via Quartz CGEvent keyboard events.

CGEventKeyboardSetUnicodeString posts real keyboard events carrying
arbitrary unicode, so it works in any app without touching the
clipboard. Requires the Accessibility permission (System Settings ->
Privacy & Security -> Accessibility).
"""

from __future__ import annotations

import time

from wisperfree.inject.base import TextInjector

# CGEventKeyboardSetUnicodeString accepts up to 20 UTF-16 code units per event
_UNITS_PER_EVENT = 20


class MacOSInjector(TextInjector):
    def _type_text(self, text: str) -> None:
        import Quartz  # pyobjc-framework-Quartz

        delay = self.config.inter_key_delay_ms / 1000.0
        for chunk in _utf16_chunks(text, _UNITS_PER_EVENT):
            # The length argument counts UTF-16 code units, not Python code
            # points; passing len(chunk) would truncate non-BMP characters
            # (e.g. emoji, which are two units each).
            unit_len = len(chunk.encode("utf-16-le")) // 2
            for key_down in (True, False):
                event = Quartz.CGEventCreateKeyboardEvent(None, 0, key_down)
                Quartz.CGEventKeyboardSetUnicodeString(event, unit_len, chunk)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            if delay:
                time.sleep(delay)


def _utf16_chunks(text: str, max_units: int):
    """Split text so each chunk is at most ``max_units`` UTF-16 code units,
    never splitting a surrogate pair (a single non-BMP code point)."""
    chunk: list[str] = []
    units = 0
    for ch in text:
        ch_units = len(ch.encode("utf-16-le")) // 2  # 1 for BMP, 2 for non-BMP
        if chunk and units + ch_units > max_units:
            yield "".join(chunk)
            chunk, units = [], 0
        chunk.append(ch)
        units += ch_units
    if chunk:
        yield "".join(chunk)
