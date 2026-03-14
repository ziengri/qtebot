import gc
import threading
import traceback
import time
from typing import Optional, Tuple

import dxcam


# ВНЕШНИЙ формат, удобный для тебя:
# (left, top, width, height)
ScreenRegion = Tuple[int, int, int, int]

# ВНУТРЕННИЙ формат dxcam:
# (left, top, right, bottom)
DxcamRegion = Tuple[int, int, int, int]


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
        self._state_cv = threading.Condition(self._lock)
        self._is_restarting = False
        self._active = False

        self.target_fps = target_fps
        self.video_mode = video_mode
        self.max_buffer_len = max_buffer_len
        self.output_color = output_color

        self.current_region: ScreenRegion = region
        self.expected_shape = (region[3], region[2], 3)
        self.restart_delay_sec = 0.2
        self.restart_retries = 8
        self.restart_retry_sleep_sec = 0.35
        self.frame_stall_timeout_sec = 1.0
        self.recover_cooldown_sec = 0.5
        now = time.monotonic()
        self._last_frame_ts = now
        self._last_recover_ts = now
        self.trace_dxcam_output_change = True

    @staticmethod
    def _to_dxcam_region(region: ScreenRegion) -> DxcamRegion:
        left, top, width, height = region
        right = left + width
        bottom = top + height
        return (left, top, right, bottom)

    def _create_camera(self) -> None:
        self.camera = dxcam.create(
            output_color=self.output_color,
            max_buffer_len=self.max_buffer_len,
        )
        self._install_dxcam_trace_hooks()

    def _install_dxcam_trace_hooks(self) -> None:
        if not self.trace_dxcam_output_change or self.camera is None:
            return

        if getattr(self.camera, "_camera_manager_trace_hooks_installed", False):
            return

        # dxcam marks DXGI access-loss in Duplicator.update_frame() by returning False.
        # Hook both points to print exactly where recovery starts.
        duplicator = getattr(self.camera, "_duplicator", None)
        if duplicator is not None and hasattr(duplicator, "update_frame"):
            original_update_frame = duplicator.update_frame

            def traced_update_frame():
                ok = original_update_frame()
                if ok is False:
                    print(
                        "[CameraManager][TRACE] "
                        "Duplicator.update_frame -> False (DXGI access/output loss)"
                    )
                return ok

            duplicator.update_frame = traced_update_frame

        if hasattr(self.camera, "_on_output_change"):
            original_on_output_change = self.camera._on_output_change

            def traced_on_output_change():
                print(
                    "[CameraManager][TRACE] DXCamera._on_output_change() called "
                    "(duplicator rebuild path)"
                )
                stack = "".join(traceback.format_stack(limit=8))
                print("[CameraManager][TRACE] call stack:\n" + stack)
                return original_on_output_change()

            self.camera._on_output_change = traced_on_output_change

        self.camera._camera_manager_trace_hooks_installed = True

    def _start_camera(self, region: ScreenRegion) -> None:
        if self.camera is None:
            self._create_camera()

        dx_region = self._to_dxcam_region(region)

        self.camera.start(
            region=dx_region,
            target_fps=self.target_fps,
            video_mode=self.video_mode,
        )

        self._active = True
        self.current_region = region
        self.expected_shape = (region[3], region[2], 3)

    def _stop_camera_safely(self) -> None:
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass

    def _rebuild_camera(self, region: ScreenRegion) -> None:
        self._stop_camera_safely()

        try:
            del self.camera
        except Exception:
            pass

        self.camera = None
        gc.collect()
        time.sleep(self.restart_delay_sec)

        self._start_camera(region)

    def _wait_if_restarting(self) -> None:
        with self._state_cv:
            while self._is_restarting:
                self._state_cv.wait()

    def _begin_restart(self) -> bool:
        with self._state_cv:
            if self._is_restarting:
                return False
            self._is_restarting = True
            return True

    def _end_restart(self) -> None:
        with self._state_cv:
            self._is_restarting = False
            self._state_cv.notify_all()

    def _recover_with_retries(
        self,
        region: Optional[ScreenRegion] = None,
        reason: str = "",
    ) -> bool:
        if region is None:
            region = self.current_region

        is_leader = self._begin_restart()
        if not is_leader:
            self._wait_if_restarting()
            return self.camera is not None

        ok = False
        try:
            if reason:
                print(f"[CameraManager] recovery started: {reason}")

            for attempt in range(1, self.restart_retries + 1):
                try:
                    self._rebuild_camera(region)
                    ok = True
                    if attempt > 1:
                        print(f"[CameraManager] recovery success on attempt {attempt}")
                    break
                except Exception as exc:
                    print(f"[CameraManager] recovery attempt {attempt} failed: {exc}")
                    if attempt < self.restart_retries:
                        time.sleep(self.restart_retry_sleep_sec)
        finally:
            self._end_restart()

        if not ok:
            print("[CameraManager] recovery failed: camera unavailable")
        else:
            self._last_frame_ts = time.monotonic()
            self._last_recover_ts = time.monotonic()
        return ok

    def start(self, region: Optional[ScreenRegion] = None) -> None:
        self._wait_if_restarting()
        if region is None:
            region = self.current_region
        self._active = True
        self._recover_with_retries(region=region, reason="start requested")

    def stop(self) -> None:
        self._wait_if_restarting()
        with self._lock:
            self._active = False
            self._stop_camera_safely()

    def restart(self, region: Optional[ScreenRegion] = None) -> None:
        if region is None:
            region = self.current_region
        self._active = True
        self._recover_with_retries(region=region, reason="manual restart")

    def set_region(self, region: ScreenRegion, restart: bool = True) -> None:
        if restart:
            self.restart(region)
        else:
            self.current_region = region
            self.expected_shape = (region[3], region[2], 3)

    def get_frame(self):
        if not self._active:
            return None

        now = time.monotonic()
        self._wait_if_restarting()
        if self.camera is None:
            if not self._active:
                return None
            if now - self._last_recover_ts >= self.recover_cooldown_sec:
                self._recover_with_retries(reason="camera was None during get_frame")
            return None
        try:
            frame = self.camera.get_latest_frame()
        except Exception as exc:
            if not self._active:
                return None
            if now - self._last_recover_ts >= self.recover_cooldown_sec:
                self._recover_with_retries(reason=f"get_latest_frame error: {exc}")
            return None

        if frame is None:
            if now - self._last_frame_ts > self.frame_stall_timeout_sec:
                if now - self._last_recover_ts >= self.recover_cooldown_sec:
                    self._recover_with_retries(
                        reason=(
                            f"frame stream stalled for "
                            f"{now - self._last_frame_ts:.2f}s"
                        )
                    )
            return None

        if frame.shape != self.expected_shape:
            if now - self._last_recover_ts >= self.recover_cooldown_sec:
                self._recover_with_retries(
                    reason=(
                        f"unexpected frame shape {frame.shape}, "
                        f"expected {self.expected_shape}"
                    )
                )
            return None

        self._last_frame_ts = now
        return frame

    def is_valid_frame_shape(self, frame) -> bool:
        return frame is not None and frame.shape == self.expected_shape

    def get_region(self) -> ScreenRegion:
        return self.current_region

    def get_expected_shape(self) -> tuple[int, int, int]:
        return self.expected_shape
