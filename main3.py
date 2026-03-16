from camera_mss import CameraManager
from qtebot_motion import InputController, MotionDetector, QTEBotMotion


def main() -> None:
    camera = CameraManager(
        region= (1400, 1, 200, 250),
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

    bot = QTEBotMotion(
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
    bot.run()


if __name__ == "__main__":
    main()
