"""Linux text injection.

X11: ``xdotool type`` synthesizes real key events via XTEST.
Wayland: ``wtype`` uses the virtual-keyboard protocol.
Auto-detected from the session type; both are direct key synthesis, not
clipboard pastes.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from wisperfree.inject.base import TextInjector


class LinuxInjector(TextInjector):
    def _type_text(self, text: str) -> None:
        tool = self._pick_tool()
        delay_ms = self.config.inter_key_delay_ms
        if tool == "wtype":
            cmd = ["wtype"]
            if delay_ms:
                cmd += ["-d", str(delay_ms)]
            cmd += ["--", text]
        else:
            cmd = ["xdotool", "type", "--delay", str(delay_ms), "--", text]
        subprocess.run(cmd, check=True, timeout=30)

    @staticmethod
    def _pick_tool() -> str:
        session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        wayland = session == "wayland" or os.environ.get("WAYLAND_DISPLAY")
        if wayland and shutil.which("wtype"):
            return "wtype"
        if shutil.which("xdotool"):
            return "xdotool"
        if shutil.which("wtype"):
            return "wtype"
        raise RuntimeError(
            "No injection tool found: install xdotool (X11) or wtype (Wayland)"
        )
