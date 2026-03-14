from camera_mss import CameraManager
from qtebot_motion import InputController, MotionDetector, QTEBotMotion


def main() -> None:
    camera = CameraManager(
        region= (777, 2, 416, 342),
        target_fps=120,
        output_color="BGR",
    )

    detector = MotionDetector(
        max_shift=15,
        move_threshold=2.0,
        blur_kernel=5,
        resize_factor=1.0,
    )

    controller = InputController(
        key_left="d",
        key_right="a",
    )

    bot = QTEBotMotion(
        camera=camera,
        detector=detector,
        controller=controller,
        consecutive_frames_required=2,
        switch_cooldown=0.05,
        loop_sleep=0.001,
        debug=True,
        log_interval=0.2,
        neutral_requires_confirmation=True,
    )
    bot.run()


if __name__ == "__main__":
    main()
