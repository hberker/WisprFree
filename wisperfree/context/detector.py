"""Active application / window detection.

macOS: NSWorkspace (AppKit) frontmost application.
Windows: GetForegroundWindow + process name via Win32 (ctypes).
Linux: xdotool/xprop on X11; swaymsg / hyprctl on Wayland compositors
that expose it (a fundamental Wayland limitation otherwise).

The result feeds the LLM prompt so tone/formatting adapt per app.
Detection failures are non-fatal — dictation proceeds with no context.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class ActiveWindow:
    app_name: str | None = None
    window_title: str | None = None

    @property
    def label(self) -> str | None:
        if self.app_name and self.window_title:
            return f"{self.app_name} — {self.window_title}"
        return self.app_name or self.window_title


class ContextDetector:
    def detect(self) -> ActiveWindow:
        try:
            if sys.platform == "darwin":
                return self._detect_macos()
            if sys.platform == "win32":
                return self._detect_windows()
            return self._detect_linux()
        except Exception:
            return ActiveWindow()

    @staticmethod
    def _detect_macos() -> ActiveWindow:
        try:
            from AppKit import NSWorkspace  # pyobjc

            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            return ActiveWindow(app_name=str(app.localizedName()))
        except ImportError:
            script = (
                'tell application "System Events" to get name of first '
                "application process whose frontmost is true"
            )
            out = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=3,
            )
            return ActiveWindow(app_name=out.stdout.strip() or None)

    @staticmethod
    def _detect_windows() -> ActiveWindow:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value or None

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        app_name = None
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED
        if handle:
            try:
                size = wintypes.DWORD(260)
                path_buf = ctypes.create_unicode_buffer(size.value)
                if kernel32.QueryFullProcessImageNameW(
                    handle, 0, path_buf, ctypes.byref(size)
                ):
                    app_name = path_buf.value.rsplit("\\", 1)[-1].removesuffix(".exe")
            finally:
                kernel32.CloseHandle(handle)
        return ActiveWindow(app_name=app_name, window_title=title)

    @staticmethod
    def _detect_linux() -> ActiveWindow:
        # Wayland compositors with query interfaces
        if shutil.which("hyprctl"):
            out = subprocess.run(
                ["hyprctl", "activewindow", "-j"],
                capture_output=True, text=True, timeout=3,
            )
            if out.returncode == 0 and out.stdout.strip():
                data = json.loads(out.stdout)
                return ActiveWindow(
                    app_name=data.get("class"), window_title=data.get("title")
                )
        if shutil.which("swaymsg"):
            out = subprocess.run(
                ["swaymsg", "-t", "get_tree"],
                capture_output=True, text=True, timeout=3,
            )
            if out.returncode == 0 and out.stdout.strip():
                node = _find_focused(json.loads(out.stdout))
                if node:
                    app = node.get("app_id") or node.get(
                        "window_properties", {}
                    ).get("class")
                    return ActiveWindow(
                        app_name=app, window_title=node.get("name")
                    )
        # X11
        if shutil.which("xdotool"):
            win = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=3,
            )
            if win.returncode == 0 and win.stdout.strip():
                wid = win.stdout.strip()
                title = subprocess.run(
                    ["xdotool", "getwindowname", wid],
                    capture_output=True, text=True, timeout=3,
                ).stdout.strip() or None
                cls = subprocess.run(
                    ["xdotool", "getwindowclassname", wid],
                    capture_output=True, text=True, timeout=3,
                ).stdout.strip() or None
                return ActiveWindow(app_name=cls, window_title=title)
        return ActiveWindow()


def _find_focused(node: dict) -> dict | None:
    if node.get("focused"):
        return node
    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        found = _find_focused(child)
        if found:
            return found
    return None
