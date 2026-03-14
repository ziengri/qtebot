import gc
import threading
import time
from typing import Optional, Tuple

import mss
import numpy as np


ScreenRegion = Tuple[int, int, int, int]


class CameraManager:
    def __init__(
        self,
        region: ScreenRegion,
        target_fps: int = 60,
        video_mode: bool = True,
        max_buffer_len: int = 64,
        output_color: str = "BGR",
    ) -> None:
        self.camera = None
        self._lock = threading.RLock()
        self._active = False
        self._next_frame_ts = 0.0

        self.target_fps = target_fps
        self.video_mode = video_mode
        self.max_buffer_len = max_buffer_len
        self.output_color = output_color.upper()

        self.current_region: ScreenRegion = region
        self.expected_shape = self._calc_expected_shape(region, self.output_color)

    @staticmethod
    def _calc_expected_shape(
        region: ScreenRegion,
        output_color: str,
    ) -> tuple[int, int, int]:
        channels = 1 if output_color == "GRAY" else 3
        return (region[3], region[2], channels)

    @staticmethod
    def _to_monitor(region: ScreenRegion) -> dict:
        left, top, width, height = region
        return {"left": left, "top": top, "width": width, "height": height}

    def _create_camera(self) -> None:
        self.camera = mss.mss()

    def _start_camera(self, region: ScreenRegion) -> None:
        if self.camera is None:
            self._create_camera()

        self.current_region = region
        self.expected_shape = self._calc_expected_shape(region, self.output_color)
        self._active = True
        self._next_frame_ts = 0.0

    def start(self, region: Optional[ScreenRegion] = None) -> None:
        if region is None:
            region = self.current_region
        with self._lock:
            self._start_camera(region)

    def stop(self) -> None:
        with self._lock:
            self._active = False
            if self.camera is not None:
                try:
                    self.camera.close()
                except Exception:
                    pass
                self.camera = None

    def restart(self, region: Optional[ScreenRegion] = None) -> None:
        if region is None:
            region = self.current_region

        self.stop()
        gc.collect()
        time.sleep(0.05)
        self.start(region)

    def set_region(self, region: ScreenRegion, restart: bool = True) -> None:
        if restart:
            self.restart(region)
        else:
            self.current_region = region
            self.expected_shape = self._calc_expected_shape(region, self.output_color)

    def _convert_frame(self, frame_bgra: np.ndarray) -> np.ndarray:
        if self.output_color == "BGRA":
            return frame_bgra
        if self.output_color == "BGR":
            return frame_bgra[:, :, :3]
        if self.output_color == "RGB":
            return frame_bgra[:, :, :3][:, :, ::-1]
        if self.output_color == "GRAY":
            bgr = frame_bgra[:, :, :3]
            gray = np.dot(bgr[..., ::-1], [0.299, 0.587, 0.114]).astype(np.uint8)
            return gray[:, :, None]
        return frame_bgra[:, :, :3]

    def get_frame(self):
        with self._lock:
            if not self._active:
                return None

            if self.camera is None:
                self._create_camera()

            if self.target_fps > 0:
                now = time.perf_counter()
                if now < self._next_frame_ts:
                    return None
                self._next_frame_ts = now + (1.0 / self.target_fps)

            monitor = self._to_monitor(self.current_region)
            raw = np.asarray(self.camera.grab(monitor), dtype=np.uint8)
            return self._convert_frame(raw)

    def is_valid_frame_shape(self, frame) -> bool:
        return frame is not None and frame.shape == self.expected_shape

    def get_region(self) -> ScreenRegion:
        return self.current_region

    def get_expected_shape(self) -> tuple[int, int, int]:
        return self.expected_shape
