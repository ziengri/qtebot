from __future__ import annotations

import ctypes
import random
import time
from typing import Literal


Point = tuple[int, int]
ActionType = Literal["pick", "drop"]

user32 = ctypes.WinDLL("user32", use_last_error=True)

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


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
        if not user32.SetCursorPos(int(x), int(y)):
            raise ctypes.WinError(ctypes.get_last_error())
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(random.uniform(0.03, 0.08))
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def run(self) -> bool:
        try:
            time.sleep(random.uniform(1.0, 3.0))
            self._click(self._target())
            return True
        except Exception as exc:
            print(f"[QTEClickBot][WARN] click failed: {exc}")
            return False
