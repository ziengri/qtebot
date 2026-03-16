from __future__ import annotations

import random
import threading
import time
from typing import Literal, Optional

import interception


Point = tuple[int, int]
ActionType = Literal["pick", "drop"]


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

    @staticmethod
    def _sleep_interruptible(
        duration: float,
        stop_event: Optional[threading.Event],
        step: float = 0.05,
    ) -> bool:
        end_ts = time.monotonic() + duration
        while time.monotonic() < end_ts:
            if stop_event is not None and stop_event.is_set():
                return False
            time.sleep(step)
        return True

    def _click(self, point: Point) -> None:
        x, y = point

        if hasattr(interception, "move_to"):
            interception.move_to(int(x), int(y))
        elif hasattr(interception, "mouse_move"):
            interception.mouse_move(int(x), int(y))
        else:
            raise RuntimeError("interception mouse move API is unavailable")

        time.sleep(random.uniform(0.02, 0.06))

        if not hasattr(interception, "mouse_down") or not hasattr(interception, "mouse_up"):
            raise RuntimeError("interception mouse click API is unavailable")

        try:
            interception.mouse_down("left")
            time.sleep(random.uniform(0.02, 0.06))
            interception.mouse_up("left")
        except TypeError:
            interception.mouse_down()
            time.sleep(random.uniform(0.02, 0.06))
            interception.mouse_up()

    def run(self, stop_event: Optional[threading.Event] = None) -> bool:
        try:
            if not self._sleep_interruptible(random.uniform(1.0, 3.0), stop_event):
                return False

            if stop_event is not None and stop_event.is_set():
                return False

            self._click(self._target())
            return True
        except Exception as exc:
            print(f"[QTEClickBot][WARN] click failed: {exc}")
            return False
