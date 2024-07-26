

import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QToolTip, QMessageBox,
                             QMainWindow, QHBoxLayout, QVBoxLayout, QFileDialog, QSizePolicy,
                             QSlider, QLabel, QLineEdit, QGridLayout, QGroupBox, QListWidget,
                             QTabWidget, QDialog, QCheckBox, QComboBox)
from PyQt6.QtGui import QIcon, QFont, QAction, QGuiApplication
from PyQt6.QtCore import Qt, QTimer, QFile, QTextStream, QSize
from PyQt6.QtWidgets import QApplication, QMainWindow, QDockWidget, QTextEdit



import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import dasQt.das as das
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 16


from dasQt.utools.logPy3 import HandleLog
from .caculateCog import CaculateCog


class CogMainWindow(QMainWindow):
    def __init__(self, MyProgram, title="COG"):
        super().__init__()

        self.is_closed           : bool = False
        self.bool_saveCog         : bool = False
        self.bool_showDispersion : bool = False
        self.editFreqNorm        :str   = 'no'
        self.editTimeNorm        :str   = 'no'
        self.editCogMethod        :str   = 'xcorr'
        self.logger = HandleLog(os.path.split(__file__)[-1].split(".")[0], path=os.getcwd(), level="DEBUG")
        self.MyProgram = CaculateCog(MyProgram)

        self.setWindowTitle(title)
        self.initUI()


    def initUI(self):
        # 创建一个主窗口的中心部件
        central_widget = QWidget()
        central_widget.setLayout(QHBoxLayout())
        self.setCentralWidget(central_widget)
        self.layout = central_widget.layout()
        # 设置主窗口的大小策略
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setSizePolicy(size_policy)

        # self.initDispersion()

        mainFigure = QMainWindow()
        self.layout.addWidget(mainFigure, 1)

        widCogFigure = QWidget()
        dockCogFigure = QDockWidget("Cog", self)
        dockCogFigure.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dockCogFigure.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dockCogFigure.setWidget(widCogFigure)
        mainFigure.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dockCogFigure)
        self.initCogFigure(widCogFigure)

        self.show()



    def initCogFigure(self, widFigure: QWidget) -> None:
        self.figCog = Figure()
        self.canvasCog = FigureCanvas(self.figCog)
        toolbar = NavigationToolbar(self.canvasCog, self)

        newLayout = QVBoxLayout()
        widFigure.setLayout(newLayout)
        newLayout.addWidget(self.canvasCog, 0)
        newLayout.addWidget(toolbar, 0)
        # self.layout.addWidget(widFigure, 1)

