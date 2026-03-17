import time
import random
import threading
from typing import Optional

import cv2
import numpy as np
import interception

# from camera import CameraManager
from camera_mss import CameraManager


Rect = tuple[int, int, int, int]


class QTEBot:
    def __init__(
        self,
        region: tuple[int, int, int, int],
        key_to_press: str = "space",
        target_fps: int = 120,
        video_mode: bool = True,
        max_buffer_len: int = 64,
        output_color: str = "BGR",
        press_cooldown: float = 0.50,
        green_lost_timeout: float = 0.40,
        reset_lock_after_press: bool = True,
        show_debug: bool = True,
        show_masks: bool = True,
        min_green_area: int = 30,
        min_white_area: int = 10,
        max_white_width: int = 25,
        min_white_height: int = 8,
        max_tracking_jump: int = 80,
        use_center_band_filter: bool = True,
        center_band_half_height: int = 25,
    ) -> None:
        self.region = region
        self.key_to_press = key_to_press.lower()
        self.press_cooldown = press_cooldown
        self.green_lost_timeout = green_lost_timeout
        self.reset_lock_after_press = reset_lock_after_press
        self.show_debug = show_debug
        self.show_masks = show_masks

        self.min_green_area = min_green_area
        self.min_white_area = min_white_area
        self.max_white_width = max_white_width
        self.min_white_height = min_white_height
        self.max_tracking_jump = max_tracking_jump
        self.use_center_band_filter = use_center_band_filter
        self.center_band_half_height = center_band_half_height

        self.camera_manager = CameraManager(
            region=region,
            target_fps=target_fps,
            video_mode=video_mode,
            max_buffer_len=max_buffer_len,
            output_color=output_color,
        )

        self.last_press_time = 0.0
        self.locked_green_rect: Optional[Rect] = None
        self.last_green_seen_time = 0.0
        self.prev_white_rect: Optional[Rect] = None
        self.prev_white_cx = None
        self.lead_pixels = 10
        self.last_green_mask = None
        self.last_white_mask = None

    @staticmethod
    def rect_center(rect: Rect) -> tuple[int, int]:
        x, y, w, h = rect
        return x + w // 2, y + h // 2

    @staticmethod
    def rect_area(rect: Rect) -> int:
        _, _, w, h = rect
        return w * h

    def reset_tracking_state(self) -> None:
        self.locked_green_rect = None
        self.last_green_seen_time = 0.0
        self.prev_white_rect = None

    def find_green_zone(self, frame_bgr) -> tuple[Optional[Rect], np.ndarray]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 80, 80], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        candidates: list[Rect] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_green_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((x, y, w, h))

        if not candidates:
            return None, mask

        best_rect = max(candidates, key=self.rect_area)
        return best_rect, mask

    def build_white_mask(self, frame_bgr) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 150], dtype=np.uint8)
        upper_white = np.array([180, 100, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def find_white_bar(self, frame_bgr) -> tuple[Optional[Rect], np.ndarray]:
        mask = self.build_white_mask(frame_bgr)

        frame_h, frame_w = frame_bgr.shape[:2]
        frame_center_y = frame_h // 2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[dict] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_white_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Палочка узкая и высокая: вертикальная линия, которая ездит по горизонтали
            if h <= w:
                continue
            if w > self.max_white_width:
                continue
            if h < self.min_white_height:
                continue

            cx = x + w // 2
            cy = y + h // 2
            y_penalty = abs(cy - frame_center_y)

            if self.use_center_band_filter and y_penalty > self.center_band_half_height:
                continue

            candidates.append(
                {
                    "rect": (x, y, w, h),
                    "cx": cx,
                    "cy": cy,
                    "h": h,
                    "w": w,
                    "area": area,
                    "y_penalty": y_penalty,
                }
            )

        if not candidates:
            return None, mask

        # 1. Если есть предыдущая позиция, пытаемся продолжить трекинг
        if self.prev_white_rect is not None:
            prev_x, prev_y, prev_w, prev_h = self.prev_white_rect
            prev_cx = prev_x + prev_w // 2

            nearest = min(candidates, key=lambda item: abs(item["cx"] - prev_cx))
            if abs(nearest["cx"] - prev_cx) <= self.max_tracking_jump:
                return nearest["rect"], mask

        # 2. Если предыдущей позиции нет или скачок слишком большой,
        #    берём самый "похожий" объект:
        #    высокий, узкий, ближе к центру по Y
        best = max(
            candidates,
            key=lambda item: (item["h"], -item["w"], -item["y_penalty"], item["area"]),
        )
        return best["rect"], mask

    def update_green_lock(self, green_rect: Optional[Rect], now: float) -> None:
        # Фиксируем цель один раз на раунд
        if self.locked_green_rect is None:
            if green_rect is not None:
                self.locked_green_rect = green_rect
                self.last_green_seen_time = now
        else:
            # Если зелёный снова видно, просто обновляем "живость",
            # но сам прямоугольник не перезаписываем
            if green_rect is not None:
                self.last_green_seen_time = now

        if (
            self.locked_green_rect is not None
            and now - self.last_green_seen_time > self.green_lost_timeout
        ):
            self.locked_green_rect = None

    def should_press(self, white_rect: Rect) -> tuple[bool, int, int, int]:
        if self.locked_green_rect is None:
            return False, 0, 0, 0

        gx, gy, gw, gh = self.locked_green_rect
        white_cx, _ = self.rect_center(white_rect)

        zone_left = gx
        zone_right = gx + gw

        predicted_x = white_cx

        if self.prev_white_cx is not None:
            dx = white_cx - self.prev_white_cx
            if dx > 0:
                predicted_x += self.lead_pixels
            elif dx < 0:
                predicted_x -= self.lead_pixels

        self.prev_white_cx = white_cx

        hit = zone_left <= predicted_x <= zone_right
        return hit, white_cx, zone_left, zone_right

    def press(self, now: float) -> bool:
        if now - self.last_press_time < self.press_cooldown:
            return False

        try:
            interception.key_down(self.key_to_press)
            time.sleep(random.uniform(0.01, 0.02))
            interception.key_up(self.key_to_press)
        except Exception as exc:
            print(f"[ERROR] Не удалось нажать клавишу: {exc}")
            return False

        self.last_press_time = now

        if self.reset_lock_after_press:
            self.locked_green_rect = None

        return True

    def draw_debug(
        self,
        frame,
        green_rect: Optional[Rect],
        white_rect: Optional[Rect],
        white_cx: Optional[int],
        zone_left: Optional[int],
        zone_right: Optional[int],
        pressed: bool,
    ) -> None:
        if green_rect is not None:
            gx, gy, gw, gh = green_rect
            cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)

        if white_rect is not None:
            wx, wy, ww, wh = white_rect
            cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), (255, 255, 255), 2)

        if white_rect is not None and white_cx is not None:
            _, white_cy = self.rect_center(white_rect)
            cv2.circle(frame, (white_cx, white_cy), 4, (0, 0, 255), -1)

        if zone_left is not None and zone_right is not None:
            cv2.line(frame, (zone_left, 0), (zone_left, frame.shape[0]), (0, 255, 255), 1)
            cv2.line(frame, (zone_right, 0), (zone_right, frame.shape[0]), (0, 255, 255), 1)

        lock_status = "LOCKED" if self.locked_green_rect is not None else "SEARCH"
        status = f"{lock_status} key={self.key_to_press}"

        if white_cx is not None and zone_left is not None and zone_right is not None:
            status += f" white_x={white_cx} zone=({zone_left}..{zone_right})"

        if pressed:
            status += " PRESS"

        cv2.putText(
            frame,
            status,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255) if pressed else (255, 255, 255),
            2,
        )

    def process_frame(self, frame) -> tuple[np.ndarray, bool]:
        green_rect, green_mask = self.find_green_zone(frame)
        white_rect, white_mask = self.find_white_bar(frame)

        self.last_green_mask = green_mask
        self.last_white_mask = white_mask

        now = time.time()
        pressed = False
        white_cx: Optional[int] = None
        zone_left: Optional[int] = None
        zone_right: Optional[int] = None

        self.update_green_lock(green_rect, now)

        if white_rect is not None:
            self.prev_white_rect = white_rect

        if self.locked_green_rect is not None and white_rect is not None:
            hit, white_cx, zone_left, zone_right = self.should_press(white_rect)
            # print(zone_left, white_cx, zone_right, hit)

            if hit:
                pressed = self.press(now)
                if pressed:
                    print("PRESS")

        if self.show_debug:
            self.draw_debug(
                frame=frame,
                green_rect=self.locked_green_rect,
                white_rect=white_rect,
                white_cx=white_cx,
                zone_left=zone_left,
                zone_right=zone_right,
                pressed=pressed,
            )

        return frame, pressed

    def run(self, stop_event: Optional[threading.Event] = None) -> bool:
        try:
            print(f"Бот запущен. Клавиша: {self.key_to_press}")
            print("�������� ������� � ������.")

            self.camera_manager.start()

            while True:
                if stop_event is not None and stop_event.is_set():
                    return False

                frame = self.camera_manager.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                if not self.camera_manager.is_valid_frame_shape(frame):
                    time.sleep(0.001)
                    continue

                debug_frame, pressed = self.process_frame(frame)
                if pressed:
                    return True

                if self.show_debug:
                    cv2.imshow("QTE Debug", debug_frame)

                    if self.show_masks:
                        if self.last_white_mask is not None:
                            cv2.imshow("White Mask", self.last_white_mask)
                        if self.last_green_mask is not None:
                            cv2.imshow("Green Mask", self.last_green_mask)

                    cv2.waitKey(1)

                time.sleep(0.001)
        except KeyboardInterrupt:
            return False

        finally:
            self.camera_manager.stop()
            cv2.destroyAllWindows()

        return False
