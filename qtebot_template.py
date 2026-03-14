from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np


Rect = tuple[int, int, int, int]  # x, y, w, h


class CameraLike(Protocol):
    def start(self, region: Optional[Rect] = None) -> None:
        ...

    def stop(self) -> None:
        ...

    def get_frame(self):
        ...


@dataclass(frozen=True)
class TemplateMatch:
    template_name: str
    score: float
    top_left: tuple[int, int]
    size: tuple[int, int]  # w, h


class TemplateDetector:
    def __init__(self, template_paths: Sequence[str]) -> None:
        if not template_paths:
            raise ValueError("template_paths must not be empty")

        self.templates: list[tuple[str, np.ndarray]] = []
        for path in template_paths:
            p = Path(path)
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Template not found or unreadable: {p}")
            self.templates.append((p.name, img))

    def match_best(self, frame_gray: np.ndarray, roi: Optional[Rect] = None) -> Optional[TemplateMatch]:
        if frame_gray is None or frame_gray.size == 0:
            return None

        if roi is not None:
            rx, ry, rw, rh = roi
            frame_part = frame_gray[ry : ry + rh, rx : rx + rw]
            origin_x, origin_y = rx, ry
        else:
            frame_part = frame_gray
            origin_x, origin_y = 0, 0

        if frame_part is None or frame_part.size == 0:
            return None

        fh, fw = frame_part.shape[:2]
        best: Optional[TemplateMatch] = None

        for name, tmpl in self.templates:
            th, tw = tmpl.shape[:2]
            if th > fh or tw > fw:
                continue

            res = cv2.matchTemplate(frame_part, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            score = float(max_val)
            top_left = (origin_x + int(max_loc[0]), origin_y + int(max_loc[1]))

            if best is None or score > best.score:
                best = TemplateMatch(
                    template_name=name,
                    score=score,
                    top_left=top_left,
                    size=(tw, th),
                )

        return best


class TemplateQTEBot:
    def __init__(
        self,
        camera: CameraLike,
        detector: TemplateDetector,
        threshold: float = 0.82,
        cooldown: float = 0.35,
        min_confirmations: int = 2,
        roi: Optional[Rect] = None,
        key: str = "space",
        show_debug: bool = False,
        debug_window_name: str = "QTE Template Debug",
        log_interval: float = 0.2,
        idle_sleep: float = 0.001,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.threshold = float(threshold)
        self.cooldown = float(cooldown)
        self.min_confirmations = int(min_confirmations)
        self.roi = roi
        self.key = key
        self.show_debug = show_debug
        self.debug_window_name = debug_window_name
        self.log_interval = log_interval
        self.idle_sleep = idle_sleep

        self._confirmations = 0
        self._last_press_ts = 0.0
        self._last_log_ts = 0.0

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[:, :, 0]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _press_space(self) -> None:
        try:
            import pyautogui

            pyautogui.press(self.key)
        except Exception:
            # Fallback for setups where pyautogui is unavailable.
            import interception

            interception.key_down(self.key)
            time.sleep(0.01)
            interception.key_up(self.key)

    def _can_press(self, now: float) -> bool:
        return (now - self._last_press_ts) >= self.cooldown

    def _draw_debug(
        self,
        frame: np.ndarray,
        match: Optional[TemplateMatch],
        score: float,
        fired: bool,
    ) -> np.ndarray:
        view = frame.copy()

        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            cv2.rectangle(view, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1)

        if match is not None:
            x, y = match.top_left
            w, h = match.size
            color = (0, 255, 0) if match.score >= self.threshold else (0, 165, 255)
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)

        text = (
            f"score={score:.3f} thr={self.threshold:.3f} "
            f"ok={self._confirmations}/{self.min_confirmations}"
        )
        if fired:
            text += " PRESS"
        cv2.putText(view, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return view

    def run(self) -> None:
        try:
            self.camera.start()
            print("Template QTE bot started. Press Q in debug window to stop.")

            while True:
                try:
                    frame = self.camera.get_frame()
                except Exception as exc:
                    print(f"[WARN] camera.get_frame failed: {exc}")
                    time.sleep(self.idle_sleep)
                    continue

                if frame is None:
                    time.sleep(self.idle_sleep)
                    continue

                gray = self._to_gray(frame)
                match = self.detector.match_best(gray, self.roi)
                score = 0.0 if match is None else match.score

                now = time.monotonic()
                if now - self._last_log_ts >= self.log_interval:
                    name = "-" if match is None else match.template_name
                    print(f"[match] best={score:.4f} template={name}")
                    self._last_log_ts = now

                if match is not None and score >= self.threshold:
                    self._confirmations += 1
                else:
                    self._confirmations = 0

                fired = False
                if self._confirmations >= self.min_confirmations and self._can_press(now):
                    try:
                        self._press_space()
                        self._last_press_ts = now
                        fired = True
                        print("[ACTION] SPACE")
                    except Exception as exc:
                        print(f"[WARN] key press failed: {exc}")
                    finally:
                        self._confirmations = 0

                if self.show_debug:
                    debug_frame = self._draw_debug(frame, match, score, fired)
                    cv2.imshow(self.debug_window_name, debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                time.sleep(self.idle_sleep)
        finally:
            self.camera.stop()
            if self.show_debug:
                cv2.destroyAllWindows()
