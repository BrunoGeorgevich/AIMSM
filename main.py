from app.Backend.MainController import MainController

from PySide2.QtCore import QResource
from PySide2.QtQml import QQmlApplicationEngine
from PySide2.QtWidgets import QApplication

import signal
import sys
import os

signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Material"

    app = QApplication(sys.argv)

    language = "EN-US"
    engine = QQmlApplicationEngine()
    ctx = engine.rootContext()

    main_controller = MainController()
    engine.addImageProvider("provider", main_controller.image_provider)
    ctx.setContextProperty("main_controller", main_controller)

    QResource.registerResource("main.rcc")
    engine.load("qrc:/main.qml")

    sys.exit(app.exec_())
