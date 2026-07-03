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
_CHARS_PER_EVENT = 20


class MacOSInjector(TextInjector):
    def _type_text(self, text: str) -> None:
        import Quartz  # pyobjc-framework-Quartz

        delay = self.config.inter_key_delay_ms / 1000.0
        for i in range(0, len(text), _CHARS_PER_EVENT):
            chunk = text[i : i + _CHARS_PER_EVENT]
            for key_down in (True, False):
                event = Quartz.CGEventCreateKeyboardEvent(None, 0, key_down)
                Quartz.CGEventKeyboardSetUnicodeString(
                    event, len(chunk), chunk
                )
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            if delay:
                time.sleep(delay)
