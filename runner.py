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
        self.restart_event = threading.Event()

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
            show_debug=True,
            show_masks=True,
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
            show_debug=True,
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
            debug=True,
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
            show_debug=True,
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
                return False
            if self.restart_event.is_set():
                return False
            if self.cancel_event.is_set():
                return False
            time.sleep(0.05)
        return True

    def _handle_stage_false(self, stage_name: str) -> None:
        if not self.enabled_event.is_set():
            print(f"[{stage_name}] stopped by F9")
            return
        if self.restart_event.is_set():
            print(f"[{stage_name}] restart requested by F8")
            return
        print(f"[{stage_name}] finished without success, restarting cycle from stage 1")

    def _run_stage4(self) -> bool:
        template_bot = self._build_stage4_template_bot()
        if not template_bot.run(stop_event=self.cancel_event):
            return False

        if self.cancel_event.is_set():
            return False

        if not self._sleep_interruptible(1.0):
            return False

        click_bot = self._build_stage4_click_bot()
        return click_bot.run(stop_event=self.cancel_event)

    def _on_hotkey_start(self) -> None:
        print("[HOTKEY] F8 -> start/restart from stage 1")
        self.enabled_event.set()
        self.restart_event.set()
        self.cancel_event.set()

    def _on_hotkey_stop(self) -> None:
        print("[HOTKEY] F9 -> stop all and go idle")
        self.enabled_event.clear()
        self.cancel_event.set()

    def run_forever(self) -> None:
        try:
            import keyboard
        except Exception as exc:
            print(f"[FATAL] keyboard module is required for hotkeys: {exc}")
            return

        keyboard.add_hotkey("f8", self._on_hotkey_start, suppress=False)
        keyboard.add_hotkey("f9", self._on_hotkey_stop, suppress=False)
        print("Runner started. F8=start/restart, F9=stop.")

        motion_thread: Optional[threading.Thread] = None

        try:
            while True:
                while not self.enabled_event.is_set():
                    time.sleep(0.1)

                self.cancel_event.clear()
                self.restart_event.clear()
                print("[CYCLE] start from stage 1")

                stage1_ok = self._build_stage1_bot().run(stop_event=self.cancel_event)
                if not stage1_ok:
                    self._handle_stage_false("stage1")
                    continue

                if self.restart_event.is_set() or not self.enabled_event.is_set():
                    continue

                stage2_ok = self._build_stage2_bot().run(stop_event=self.cancel_event)
                if not stage2_ok:
                    self._handle_stage_false("stage2")
                    continue

                if self.restart_event.is_set() or not self.enabled_event.is_set():
                    continue

                motion_result = {"value": False}

                def motion_worker() -> None:
                    bot = self._build_stage3_bot()
                    motion_result["value"] = bot.run(stop_event=self.cancel_event)

                motion_thread = threading.Thread(target=motion_worker, name="stage3-motion", daemon=True)
                motion_thread.start()

                stage4_ok = self._run_stage4()

                self.cancel_event.set()
                motion_thread.join(timeout=10.0)
                motion_thread = None

                if not stage4_ok:
                    self._handle_stage_false("stage4")
                    continue

                print("[CYCLE] stage4=True, stopping all stages and sleeping 5 seconds")
                self.cancel_event.clear()
                if not self._sleep_interruptible(5.0):
                    continue

                print("[CYCLE] restart after sleep")
        finally:
            self.cancel_event.set()
            if motion_thread is not None and motion_thread.is_alive():
                motion_thread.join(timeout=2.0)
            keyboard.unhook_all_hotkeys()


def main() -> None:
    runner = QTESequenceRunner()
    runner.run_forever()


if __name__ == "__main__":
    main()
