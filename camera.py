import gc
import time
from typing import Optional, Tuple

import dxcam


ScreenRegion = Tuple[int, int, int, int]
DxcamRegion = Tuple[int, int, int, int]


class CameraManager:
    def __init__(
        self,
        region: ScreenRegion,
        target_fps: int = 60,
        video_mode: bool = True,
        max_buffer_len: int = 64,
        output_color: str = "BGR",
        recovery_cooldown: float = 0.5,
    ) -> None:
        self.camera = None

        self.target_fps = target_fps
        self.video_mode = video_mode
        self.max_buffer_len = max_buffer_len
        self.output_color = output_color
        self.recovery_cooldown = recovery_cooldown

        self.current_region: ScreenRegion = region
        self.expected_shape = (region[3], region[2], 3)

        self.last_recovery_time = 0.0

    @staticmethod
    def _to_dxcam_region(region: ScreenRegion) -> DxcamRegion:
        left, top, width, height = region
        return (left, top, left + width, top + height)

    def _create_camera(self) -> None:
        self.camera = dxcam.create(
            output_color=self.output_color,
            max_buffer_len=self.max_buffer_len,
        )

    def _start_camera(self, region: ScreenRegion) -> None:
        if self.camera is None:
            self._create_camera()

        self.camera.start(
            region=self._to_dxcam_region(region),
            target_fps=self.target_fps,
            video_mode=self.video_mode,
        )

        self.current_region = region
        self.expected_shape = (region[3], region[2], 3)

    def start(self, region: Optional[ScreenRegion] = None) -> None:
        if region is None:
            region = self.current_region
        self._start_camera(region)

    def stop(self) -> None:
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass

    def restart(self, region: Optional[ScreenRegion] = None) -> None:
        if region is None:
            region = self.current_region

        self.stop()

        try:
            del self.camera
        except Exception:
            pass

        self.camera = None
        gc.collect()
        time.sleep(0.2)

        self._start_camera(region)

    def recover(self) -> None:
        now = time.time()
        if now - self.last_recovery_time < self.recovery_cooldown:
            return

        self.last_recovery_time = now
        print("[CameraManager] Recovering dxcam after access loss/output change...")
        self.restart()

    def get_frame(self):
        if self.camera is None:
            return None

        try:
            frame = self.camera.get_latest_frame()
        except Exception as exc:
            print(f"[CameraManager] get_latest_frame failed: {exc}")
            self.recover()
            return None

        if frame is None:
            return None

        # Иногда после system transition может прийти кадр не того размера
        if not self.is_valid_frame_shape(frame):
            print(
                f"[CameraManager] Invalid frame shape: got={getattr(frame, 'shape', None)} "
                f"expected={self.expected_shape}"
            )
            self.recover()
            return None

        return frame

    def is_valid_frame_shape(self, frame) -> bool:
        return frame is not None and hasattr(frame, "shape") and frame.shape == self.expected_shape

    def set_region(self, region: ScreenRegion, restart: bool = True) -> None:
        if restart:
            self.restart(region)
        else:
            self.current_region = region
            self.expected_shape = (region[3], region[2], 3)

    def get_region(self) -> ScreenRegion:
        return self.current_region