import time

from camera_mss import CameraManager
from qtebot_template import TemplateDetector, TemplateQTEBot
from qte_click_bot import QTEClickBot, Point
camera = CameraManager(
    region=(744, 678, 430, 100),  # область захвата
    target_fps=120,
    output_color="BGR",
)
detector = TemplateDetector([
    "templates/finish_template.png"])

bot = TemplateQTEBot(
    camera=camera,
    detector=detector,
    roi=None,   # ROI внутри кадра камеры; можно None
    threshold=0.98,
    cooldown=0.35,
    min_confirmations=2,
    key=None,
    show_debug=True,
)
result = bot.run()
if result:
    print("Задача выполнена.")
    time.sleep(1)
else:
    print("Остановлено без выполнения.")

bot = QTEClickBot(
    pick=[850, 727],
    drop=[1050, 725],
    type="drop"
)
result = bot.run()