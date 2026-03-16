from __future__ import annotations

import ctypes
import random
import time
from typing import Literal


Point = tuple[int, int]
ActionType = Literal["pick", "drop"]

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_uint),
        ("dwFlags", ctypes.c_uint),
        ("time", ctypes.c_uint),
        ("dwExtraInfo", ULONG_PTR),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_uint), ("u", INPUT_UNION)]


class QTEClickBot:
    def __init__(
        self,
        pick: Point,
        drop: Point,
        type: ActionType,
    ) -> None:
        self.pick = pick
        self.drop = drop
        self.type = type.lower()
        if self.type not in ("pick", "drop"):
            raise ValueError("type must be 'pick' or 'drop'")

    def _target(self) -> Point:
        if self.type == "pick":
            return self.pick
        return self.drop

    def _click(self, point: Point) -> None:
        x, y = point
        if not self._click_with_interception(x, y):
            self._click_with_sendinput(x, y)

    def _click_with_interception(self, x: int, y: int) -> bool:
        try:
            import interception
        except Exception:
            return False

        try:
            # Support different interception wrappers.
            if hasattr(interception, "move_to"):
                interception.move_to(int(x), int(y))
            elif hasattr(interception, "mouse_move"):
                interception.mouse_move(int(x), int(y))
            else:
                return False

            time.sleep(random.uniform(0.02, 0.06))

            if hasattr(interception, "mouse_down") and hasattr(interception, "mouse_up"):
                try:
                    interception.mouse_down("left")
                    time.sleep(random.uniform(0.02, 0.06))
                    interception.mouse_up("left")
                except TypeError:
                    interception.mouse_down()
                    time.sleep(random.uniform(0.02, 0.06))
                    interception.mouse_up()
                return True
        except Exception:
            return False

        return False

    def _send_input(self, flags: int, dx: int = 0, dy: int = 0) -> None:
        inp = INPUT(
            type=INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=dx,
                dy=dy,
                mouseData=0,
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            ),
        )
        result = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if result != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def _click_with_sendinput(self, x: int, y: int) -> None:
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        if screen_w <= 1 or screen_h <= 1:
            raise RuntimeError("Invalid screen size")

        abs_x = int(round(x * 65535 / (screen_w - 1)))
        abs_y = int(round(y * 65535 / (screen_h - 1)))
        self._send_input(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, abs_x, abs_y)
        time.sleep(random.uniform(0.02, 0.06))
        self._send_input(MOUSEEVENTF_LEFTDOWN)
        time.sleep(random.uniform(0.02, 0.06))
        self._send_input(MOUSEEVENTF_LEFTUP)

    def run(self) -> bool:
        try:
            time.sleep(random.uniform(1.0, 3.0))
            self._click(self._target())
            return True
        except Exception as exc:
            print(f"[QTEClickBot][WARN] click failed: {exc}")
            return False
