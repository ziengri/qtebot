import ctypes
import cv2
import time

from camera import CameraManager


user32 = ctypes.windll.user32

VK_F = 0x46      # виртуальный код клавиши F
VK_ESC = 0x1B    # виртуальный код клавиши Esc


def is_key_pressed(vk_code: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk_code) & 0x8000)


def wait_for_key_press(vk_code: int, poll_interval: float = 0.01) -> None:
    while True:
        if is_key_pressed(vk_code):
            # ждём отпускания, чтобы не было повторного срабатывания
            while is_key_pressed(vk_code):
                time.sleep(poll_interval)
            return
        time.sleep(poll_interval)


def pick_region():
    cam = CameraManager(
        region=(0, 0, 1920, 1080),
        target_fps=30,
        video_mode=True,
        max_buffer_len=8,
        output_color="BGR",
    )

    try:
        cam.start()

        frame = None
        for _ in range(100):
            frame = cam.get_frame()
            if frame is not None and cam.is_valid_frame_shape(frame):
                break
            time.sleep(0.01)

        if frame is None:
            raise RuntimeError("Не удалось получить кадр с экрана")

        x, y, w, h = cv2.selectROI("Select QTE Region", frame, showCrosshair=True)
        cv2.destroyAllWindows()

        region = (int(x), int(y), int(w), int(h))
        print("Selected region:", region)
        return region

    finally:
        cam.stop()
        cv2.destroyAllWindows()


def wait_for_f_and_pick_region():
    print("Нажми F, чтобы выбрать область...")
    wait_for_key_press(VK_F)
    return pick_region()


if __name__ == "__main__":
    region = wait_for_f_and_pick_region()
    print("Итоговый region:", region)