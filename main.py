import GUI
import sys
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
win = GUI.FaceGUI()
win.show()
sys.exit(app.exec_())