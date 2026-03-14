from qtebot import QTEBot


def main():
    bot = QTEBot(
        region=(650, 880, 600, 50) ,
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
    bot.run()


if __name__ == "__main__":
    main()