from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Literal, Optional, Protocol

import cv2
import numpy as np


Direction = Literal["left", "right"]


user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_KEYBOARD = 1
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002

SCANCODE_BY_KEY: dict[str, int] = {
    "a": 0x1E,
    "d": 0x20,
}

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_uint),
        ("time", ctypes.c_uint),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_uint),
        ("dwFlags", ctypes.c_uint),
        ("time", ctypes.c_uint),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_uint),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("type", ctypes.c_uint),
        ("u", INPUT_UNION),
    ]


def send_key(scan_code: int, key_up: bool = False) -> None:
    flags = KEYEVENTF_SCANCODE
    if key_up:
        flags |= KEYEVENTF_KEYUP

    inp = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(
            wVk=0,
            wScan=scan_code,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        ),
    )

    result = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if result != 1:
        raise ctypes.WinError(ctypes.get_last_error())


class CameraLike(Protocol):
    def start(self, region: Optional[tuple[int, int, int, int]] = None) -> None:
        ...

    def stop(self) -> None:
        ...

    def get_frame(self):
        ...


@dataclass(frozen=True)
class MotionResult:
    shift_x: float
    direction: Direction
    score: float


class MotionDetector:
    def __init__(
        self,
        max_shift: int = 15,
        move_threshold: float = 2.0,
        blur_kernel: Optional[int] = 5,
        resize_factor: Optional[float] = None,
    ) -> None:
        if max_shift < 1:
            raise ValueError("max_shift must be >= 1")
        if move_threshold < 0:
            raise ValueError("move_threshold must be >= 0")
        if blur_kernel is not None and blur_kernel > 0 and blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be odd")
        if resize_factor is not None and not (0 < resize_factor <= 1.0):
            raise ValueError("resize_factor must be in (0, 1]")

        self.max_shift = int(max_shift)
        self.move_threshold = float(move_threshold)
        self.blur_kernel = blur_kernel
        self.resize_factor = resize_factor
        self._prev_gray: Optional[np.ndarray] = None
        self._flow_levels = 3
        self._flow_winsize = 21
        self._flow_iterations = 3

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 1:
            gray = frame[:, :, 0]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(gray)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = self._to_gray(frame)

        if self.resize_factor is not None and self.resize_factor != 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=self.resize_factor,
                fy=self.resize_factor,
                interpolation=cv2.INTER_AREA,
            )

        if self.blur_kernel is not None and self.blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        return gray

    def _direction_from_shift(self, shift_x: float) -> Direction:
        if shift_x < 0:
            return "left"
        return "right"

    def update(self, frame: np.ndarray) -> MotionResult:
        curr_gray = self._preprocess(frame)

        if self._prev_gray is None:
            self._prev_gray = curr_gray
            return MotionResult(shift_x=0.0, direction="right", score=0.0)

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=self._flow_levels,
            winsize=self._flow_winsize,
            iterations=self._flow_iterations,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flow_x = flow[..., 0]
        raw_shift_x = float(np.median(flow_x))
        clamped_shift_x = float(np.clip(raw_shift_x, -self.max_shift, self.max_shift))

        self._prev_gray = curr_gray

        if self.resize_factor is not None and self.resize_factor != 1.0:
            shift_x = clamped_shift_x / self.resize_factor
        else:
            shift_x = clamped_shift_x

        direction = self._direction_from_shift(shift_x)
        # confidence proxy: larger coherent horizontal flow => higher confidence
        score = float(np.mean(np.abs(flow_x)))
        return MotionResult(shift_x=shift_x, direction=direction, score=score)


class InputController:
    def __init__(self, key_left: str = "d", key_right: str = "a") -> None:
        self.key_left = key_right
        self.key_right = key_left
        self._held_left = False
        self._held_right = False
        self._backend_error_reported = False

    def _safe_key_down(self, key: str) -> None:
        try:
            scan = SCANCODE_BY_KEY[key.lower()]
            send_key(scan, key_up=False)
            print(f"Key down:{key}")
        except Exception as exc:
            if not self._backend_error_reported:
                print(f"[InputController][WARN] key_down failed: {exc}")
                self._backend_error_reported = True

    def _safe_key_up(self, key: str) -> None:
        try:
            scan = SCANCODE_BY_KEY[key.lower()]
            send_key(scan, key_up=True)
            print(f"Key up:{key}")
        except Exception as exc:
            if not self._backend_error_reported:
                print(f"[InputController][WARN] key_up failed: {exc}")
                self._backend_error_reported = True

    def _hold_left(self) -> None:
        if not self._held_left:
            self._safe_key_down(self.key_left)
            self._held_left = True

    def _release_left(self) -> None:
        if self._held_left:
            self._safe_key_up(self.key_left)
            self._held_left = False

    def _hold_right(self) -> None:
        if not self._held_right:
            self._safe_key_down(self.key_right)
            self._held_right = True

    def _release_right(self) -> None:
        if self._held_right:
            self._safe_key_up(self.key_right)
            self._held_right = False

    def set_direction(self, direction: Direction) -> None:
        print(f"Direction:{direction}")
        if direction == "left":
            self._release_right()
            self._hold_left()
            return
        self._release_left()
        self._hold_right()

    def release_all(self) -> None:
        self._release_left()
        self._release_right()


class QTEBotMotion:
    def __init__(
        self,
        camera: CameraLike,
        detector: MotionDetector,
        controller: InputController,
        consecutive_frames_required: int = 2,
        switch_cooldown: float = 0.05,
        loop_sleep: float = 0.001,
        debug: bool = False,
        debug_window_name: str = "QTE Motion Debug",
        log_interval: float = 0.2,
        initial_direction: Direction = "right",
    ) -> None:
        if consecutive_frames_required < 1:
            raise ValueError("consecutive_frames_required must be >= 1")
        if switch_cooldown < 0:
            raise ValueError("switch_cooldown must be >= 0")

        self.camera = camera
        self.detector = detector
        self.controller = controller
        self.consecutive_frames_required = int(consecutive_frames_required)
        self.switch_cooldown = float(switch_cooldown)
        self.loop_sleep = float(loop_sleep)
        self.debug = debug
        self.debug_window_name = debug_window_name
        self.log_interval = log_interval

        self._pending_direction: Direction = initial_direction
        self._pending_count = 0
        self._committed_direction: Direction = initial_direction
        self._last_switch_ts = 0.0
        self._last_log_ts = 0.0

    def _update_pending(self, direction: Direction) -> None:
        if direction == self._pending_direction:
            self._pending_count += 1
        else:
            self._pending_direction = direction
            self._pending_count = 1

    def _try_commit(self, now: float) -> None:
        if self._pending_count < self.consecutive_frames_required:
            return
        if self._pending_direction == self._committed_direction:
            return
        if now - self._last_switch_ts < self.switch_cooldown:
            return

        self._committed_direction = self._pending_direction
        self._last_switch_ts = now
        self.controller.set_direction(self._committed_direction)

    def _debug_log(self, result: MotionResult) -> None:
        now = time.monotonic()
        if now - self._last_log_ts < self.log_interval:
            return
        print(
            "[motion] "
            f"shift_x={result.shift_x:.2f} "
            f"dir={result.direction} "
            f"score(flow_mag)={result.score:.4f} "
            f"commit={self._committed_direction} "
            f"pending={self._pending_direction}:{self._pending_count}"
        )
        self._last_log_ts = now

    def _draw_debug(self, frame: np.ndarray, result: MotionResult) -> np.ndarray:
        view = frame.copy()
        h, w = view.shape[:2]
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), (255, 255, 0), 1)
        text = (
            f"shift={result.shift_x:.2f} "
            f"dir={result.direction} "
            f"score={result.score:.3f} "
            f"commit={self._committed_direction}"
        )
        cv2.putText(
            view,
            text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return view

    def run(self) -> None:
        try:
            self.camera.start()
            self.controller.set_direction(self._committed_direction)
            print("Motion QTE bot started. Press Esc in debug window to stop.")

            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(self.loop_sleep)
                    continue

                try:
                    result = self.detector.update(frame)
                except Exception as exc:
                    if self.debug:
                        print(f"[motion][WARN] detector failed: {exc}")
                    time.sleep(self.loop_sleep)
                    continue

                # If movement is too small, keep current pressed direction.
                if abs(result.shift_x) < self.detector.move_threshold:
                    observed_direction = self._committed_direction
                else:
                    observed_direction = result.direction

                self._update_pending(observed_direction)
                now = time.monotonic()
                self._try_commit(now)

                if self.debug:
                    self._debug_log(result)
                    debug_frame = self._draw_debug(frame, result)
                    cv2.imshow(self.debug_window_name, debug_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                time.sleep(self.loop_sleep)
        except KeyboardInterrupt:
            pass
        finally:
            self.controller.release_all()
            self.camera.stop()
            if self.debug:
                cv2.destroyAllWindows()
