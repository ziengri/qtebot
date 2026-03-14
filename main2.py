from camera_mss import CameraManager
from qtebot_template import TemplateDetector, TemplateQTEBot

camera = CameraManager(
    region=(1347, 907, 179, 173),  # область захвата
    target_fps=120,
    output_color="BGR",
)
detector = TemplateDetector([
    "templates/template1.png",
    "templates/template2.png",
])

bot = TemplateQTEBot(
    camera=camera,
    detector=detector,
    roi=None,   # ROI внутри кадра камеры; можно None
    threshold=0.82,
    cooldown=0.35,
    min_confirmations=2,
    key="space",
    show_debug=True,
)
bot.run()
