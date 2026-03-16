from camera_mss import CameraManager
from qtebot_template import TemplateDetector, TemplateQTEBot

camera = CameraManager(
    region=(1354, 843, 160, 160),  # область захвата
    target_fps=120,
    output_color="BGR",
)
detector = TemplateDetector([
    "templates/templateP1.png",
    "templates/templateP2.png",
    "templates/templateS1.png",
    "templates/templateS2.png",
    "templates/templateS3.png",
])

bot = TemplateQTEBot(
    camera=camera,
    detector=detector,
    roi=None,   # ROI внутри кадра камеры; можно None
    threshold=0.98,
    cooldown=0.35,
    min_confirmations=2,
    key="space",
    show_debug=True,
)
bot.run()
