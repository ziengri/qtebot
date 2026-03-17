from __future__ import annotations

import threading
import time
from typing import Optional

from camera_mss import CameraManager
from qte_click_bot import QTEClickBot
from qtebot import QTEBot
from qtebot_motion import InputController, MotionDetector, QTEBotMotion
from qtebot_template import TemplateDetector, TemplateQTEBot


class QTESequenceRunner:
    def __init__(self) -> None:
        self.enabled_event = threading.Event()
        self.cancel_event = threading.Event()
        self._cycle_idx = 0

    def _log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        thread_name = threading.current_thread().name
        print(f"[{ts}][{thread_name}] {message}")

    def _build_stage1_bot(self) -> QTEBot:
        return QTEBot(
            region=(650, 880, 600, 50),
            key_to_press="space",
            target_fps=120,
            video_mode=True,
            max_buffer_len=64,
            output_color="BGR",
            press_cooldown=0.50,
            green_lost_timeout=0.40,
            reset_lock_after_press=True,
            show_debug=False,
            show_masks=False,
            min_green_area=30,
            min_white_area=10,
            max_white_width=25,
            min_white_height=8,
            max_tracking_jump=80,
            use_center_band_filter=True,
            center_band_half_height=25,
        )

    def _build_stage2_bot(self) -> TemplateQTEBot:
        camera = CameraManager(
            region=(1354, 843, 160, 160),
            target_fps=120,
            output_color="BGR",
        )
        detector = TemplateDetector(
            [
                "templates/templateP1.png",
                "templates/templateP2.png",
                "templates/templateS1.png",
                "templates/templateS2.png",
                "templates/templateS3.png",
            ]
        )
        return TemplateQTEBot(
            camera=camera,
            detector=detector,
            roi=None,
            threshold=0.98,
            cooldown=0.35,
            min_confirmations=2,
            key="space",
            show_debug=False,
        )

    def _build_stage3_bot(self) -> QTEBotMotion:
        camera = CameraManager(
            region=(1400, 1, 200, 250),
            target_fps=120,
            output_color="BGR",
        )
        detector = MotionDetector(
            max_shift=20,
            move_threshold=0.2,
            blur_kernel=5,
            resize_factor=1.0,
        )
        controller = InputController(
            key_left="d",
            key_right="a",
        )
        return QTEBotMotion(
            camera=camera,
            detector=detector,
            controller=controller,
            consecutive_frames_required=2,
            switch_cooldown=0.05,
            loop_sleep=0.001,
            debug=False,
            log_interval=0.2,
            initial_direction="right",
        )

    def _build_stage4_template_bot(self) -> TemplateQTEBot:
        camera = CameraManager(
            region=(744, 678, 430, 100),
            target_fps=120,
            output_color="BGR",
        )
        detector = TemplateDetector(["templates/finish_template.png"])
        return TemplateQTEBot(
            camera=camera,
            detector=detector,
            roi=None,
            threshold=0.98,
            cooldown=0.35,
            min_confirmations=2,
            key=None,
            show_debug=False,
        )

    def _build_stage4_click_bot(self) -> QTEClickBot:
        return QTEClickBot(
            pick=(850, 727),
            drop=(1050, 725),
            type="drop",
        )

    def _sleep_interruptible(self, duration: float) -> bool:
        end_ts = time.monotonic() + duration
        while time.monotonic() < end_ts:
            if not self.enabled_event.is_set():
                self._log("[sleep] interrupted: disabled")
                return False
            if self.cancel_event.is_set():
                self._log("[sleep] interrupted: cancel")
                return False
            time.sleep(0.05)
        return True

    def _on_hotkey_start(self) -> None:
        self._log("[HOTKEY] F8 START/RESTART")
        self.enabled_event.set()
        self.cancel_event.set()  # interrupt current stage and start cycle from stage1

    def _on_hotkey_stop(self) -> None:
        self._log("[HOTKEY] F9 STOP")
        self.enabled_event.clear()
        self.cancel_event.set()

    def _wait_until_enabled(self) -> None:
        while not self.enabled_event.is_set():
            time.sleep(0.1)

    def _handle_stage_false(self, stage_name: str) -> None:
        if not self.enabled_event.is_set():
            self._log(f"[{stage_name}] stopped by F9")
            return
        if self.cancel_event.is_set():
            self._log(f"[{stage_name}] cancelled/restarted")
            return
        self._log(f"[{stage_name}] returned False, restarting cycle")

    def _run_stage4(self) -> bool:
        self._log("[stage4] template start")
        template_ok = self._build_stage4_template_bot().run(stop_event=self.cancel_event)
        self._log(f"[stage4] template result={template_ok}")
        if not template_ok:
            return False

        if not self._sleep_interruptible(1.0):
            return False

        self._log("[stage4] click start")
        click_ok = self._build_stage4_click_bot().run(stop_event=self.cancel_event)
        self._log(f"[stage4] click result={click_ok}")
        return click_ok

    def run_forever(self) -> None:
        try:
            import keyboard
        except Exception as exc:
            self._log(f"[FATAL] keyboard module is required: {exc}")
            return

        keyboard.add_hotkey("f8", self._on_hotkey_start, suppress=False)
        keyboard.add_hotkey("f9", self._on_hotkey_stop, suppress=False)
        self._log("Runner started. F8=start/restart, F9=stop.")

        motion_thread: Optional[threading.Thread] = None
        motion_result = {"value": False}

        try:
            while True:
                self._wait_until_enabled()

                self._cycle_idx += 1
                cycle_id = self._cycle_idx
                self.cancel_event.clear()
                self._log(f"[CYCLE {cycle_id}] start")

                stage1_ok = self._build_stage1_bot().run(stop_event=self.cancel_event)
                self._log(f"[CYCLE {cycle_id}] stage1={stage1_ok}")
                if not stage1_ok:
                    self._handle_stage_false("stage1")
                    continue

                if not self.enabled_event.is_set() or self.cancel_event.is_set():
                    self._log(f"[CYCLE {cycle_id}] interrupted after stage1")
                    continue

                stage2_ok = self._build_stage2_bot().run(stop_event=self.cancel_event)
                self._log(f"[CYCLE {cycle_id}] stage2={stage2_ok}")
                if not stage2_ok:
                    self._handle_stage_false("stage2")
                    continue

                if not self.enabled_event.is_set() or self.cancel_event.is_set():
                    self._log(f"[CYCLE {cycle_id}] interrupted after stage2")
                    continue

                motion_result["value"] = False

                def motion_worker() -> None:
                    self._log(f"[CYCLE {cycle_id}] stage3 thread start")
                    bot = self._build_stage3_bot()
                    motion_result["value"] = bot.run(stop_event=self.cancel_event)
                    self._log(f"[CYCLE {cycle_id}] stage3 thread end result={motion_result['value']}")

                motion_thread = threading.Thread(target=motion_worker, name="stage3-motion", daemon=True)
                motion_thread.start()

                stage4_ok = self._run_stage4()
                self._log(f"[CYCLE {cycle_id}] stage4={stage4_ok}")

                self.cancel_event.set()
                motion_thread.join(timeout=10.0)
                self._log(
                    f"[CYCLE {cycle_id}] stage3 joined "
                    f"(alive={motion_thread.is_alive()} result={motion_result['value']})"
                )
                motion_thread = None

                if not stage4_ok:
                    self._handle_stage_false("stage4")
                    continue

                self._log(f"[CYCLE {cycle_id}] success -> sleep 5s")
                self.cancel_event.clear()
                if not self._sleep_interruptible(5.0):
                    continue
        finally:
            self.cancel_event.set()
            if motion_thread is not None and motion_thread.is_alive():
                motion_thread.join(timeout=2.0)
            keyboard.unhook_all_hotkeys()
            self._log("[runner] stopped")


def main() -> None:
    runner = QTESequenceRunner()
    runner.run_forever()


if __name__ == "__main__":
    main()
