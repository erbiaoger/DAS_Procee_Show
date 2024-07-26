"""
    * @file: dasQt.py
    * @version: v1.0.0
    * @author: Zhiyu Zhang
    * @desc: GUI for DAS data
    * @date: 2023-07-25 10:08:16
    * @Email: erbiaoger@gmail.com
    * @url: erbiaoger.site

"""


import os
import sys
import numpy as np

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 16


from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QToolTip, QMessageBox,
                             QMainWindow, QHBoxLayout, QVBoxLayout, QFileDialog, QSizePolicy,
                             QSlider, QLabel, QLineEdit, QGridLayout, QGroupBox, QListWidget,
                             QTabWidget, QDialog, QCheckBox, QComboBox)
from PyQt6.QtGui import QIcon, QFont, QAction, QGuiApplication
from PyQt6.QtCore import Qt, QTimer, QFile, QTextStream, QSize
from PyQt6.QtWidgets import QApplication, QMainWindow, QDockWidget, QTextEdit

from dasQt import about
import dasQt.das as das

from dasQt.Process.showProcess import ProcessMainWindow
from dasQt.Dispersion.showDispersion import DispersionMainWindow
from dasQt.Inversion1D.showInversion1D import Inversion1DMainWindow
from dasQt.Cog.showCog import CogMainWindow
from dasQt.YOLO.showYolo import YoloMainWindow
from dasQt.utools import dasStartGUI
from dasQt.utools.logPy3 import HandleLog


global MyProgram1
MyProgram1 = das.DAS() 

def apply_stylesheet(app, path):
    file = QFile(path)
    if file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
        stream = QTextStream(file)
        app.setStyleSheet(stream.readAll())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        
        self.MyProgram        = MyProgram1
        self.indexTime       : int  = 0
        self.fIndex          : int  = 0
        self.tabNum          : int  = 0
        self.ms              : int  = 100
        self.colormap        : str  = "rainbow"
        self.readNextDataBool: bool = False
        self.rawDataBool     : bool = True
        self.filterBool       : bool = False
        self.cutDataBool     : bool = False
        self.fig_dispersion   : bool = None
        self.bool_saveCC     : bool = False
        self.logger = HandleLog(os.path.split(__file__)[-1].split(".")[0], path=os.getcwd(), level="DEBUG")

        self.initUI()

        
    def initUI(self) -> None:
        screen = QGuiApplication.primaryScreen()
        width  = screen.geometry().width()
        height = screen.geometry().height()
        self.resize(width, height)
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setSizePolicy(size_policy)
        # 设置主窗口的标题和图标
        self.setWindowTitle('DAS Main Window')
        self.setWindowIcon(QIcon('web.png'))
        QToolTip.setFont(QFont('Times New Roman', 10))
        self.setToolTip('This is <b>DAS Show</b> GUI')
        self.initMenu()

        # 设置控制窗口
        widControl = QWidget()
        dockControl = QDockWidget("Control", self)
        dockControl.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dockControl.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dockControl.setWidget(widControl)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dockControl)
        self.initControl(widControl)



        # 设置显示图像的窗口
        self.layout    = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)
        tabShow = QTabWidget()
        tabShow.tabBarClicked.connect(self.tabBarShowClicked) 
        self.layout.addWidget(tabShow, 4)
        self.initFigure(tabShow)
        self.initWigb(tabShow)
        self.figTabWidget = tabShow



        self.show()


    def tabBarShowClicked(self, index):
        if index == 0:
            self.timerWigb.stop()
            self.timer.start(self.ms)
            self.tabNum = 0
        elif index == 1:
            self.timer.stop()
            self.timerWigb.start(self.ms)
            self.tabNum = 1
        elif index == 3:
            self.timer.stop()
            self.timerWigb.stop()
            self.tabNum = 1

    def initMenuBtn(self, widMenu: QWidget) -> None:
        layout = QGridLayout()
        widMenu.setLayout(layout)

        btnDispersion = QPushButton('Dispersion', self)
        btnDispersion.setIcon(QIcon('dasQt/dispersion.png'))
        btnDispersion.setIconSize(QSize(32, 32))
        btnDispersion.setToolTip('This is a <b>QPushButton</b> widget')
        btnDispersion.clicked.connect(lambda: [self.dispersion()])
        layout.addWidget(btnDispersion, 1, 0)

        btnCog = QPushButton('Cog', self)
        btnCog.setToolTip('This is a <b>QPushButton</b> widget')
        btnCog.clicked.connect(lambda: [self.cog()])
        layout.addWidget(btnCog, 1, 1)

        btnCarClass = QPushButton('CarClass', self)
        btnCarClass.setToolTip('This is a <b>QPushButton</b> widget')
        btnCarClass.clicked.connect(lambda: [self.carClass()])
        layout.addWidget(btnCarClass, 1, 3)

    ###############
    # Init Figure #
    ###############
    def initFigure(self, tabFigure) -> None:
        self.fig    = Figure()
        self.canvas = FigureCanvas(self.fig)
        toolbar     = NavigationToolbar(self.canvas, self)

        widFig      = QWidget()
        newLayout   = QVBoxLayout()
        widFig.setLayout(newLayout)
        newLayout.addWidget(self.canvas, 0)
        newLayout.addWidget(toolbar, 0)
        tabFigure.addTab(widFig, "Figure")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.widfac = 1
        self.highfac = 1
        fontfac = 1
        ax = self.fig.add_subplot(111)
        dasStartGUI.showcsimGPR(ax,dir_path,self.widfac,self.highfac,fontfac)
        
        self.timer      = QTimer()
        self.timer.timeout.connect(self.updateFigure)

    #############
    # Init Wigb #
    #############
    def initWigb(self, tabFigure) -> None:
        self.figWigb = Figure()
        self.canvasWigb = FigureCanvas(self.figWigb)
        toolbar = NavigationToolbar(self.canvasWigb, self)

        widFig = QWidget()
        newLayout = QVBoxLayout()
        widFig.setLayout(newLayout)
        newLayout.addWidget(self.canvasWigb, 0)
        newLayout.addWidget(toolbar, 0)
        tabFigure.addTab(widFig, "Wigb")

        self.timerWigb  = QTimer()
        self.timerWigb.timeout.connect(self.updateWigb)

    #############
    # Init Menu #
    #############
    def initMenu(self) -> None:
        def act(name, shortcut, tip, func) -> None:
            # define a action
            name.setShortcut(shortcut)
            name.setStatusTip(tip)
            name.setCheckable(True)
            name.triggered.connect(func)

        openAct = QAction(QIcon('open.png'), 'Open', self)
        act(openAct, 'Ctrl+O', 'Open new File', \
            lambda: [self.importData(), self.imshowData()])
        
        openFolderAct = QAction(QIcon('open.webp'), 'Open Folder', self)
        act(openFolderAct, 'Ctrl+Shift+O', 'Open new Folder', \
            lambda: [self.openFolder()])
        
        saveAct = QAction(QIcon('dasQt/pic/save.png'), 'Save', self)
        act(saveAct, 'Ctrl+S', 'Save File', \
            lambda: [self.save(self.MyProgram)])
        
        undoAct = QAction(QIcon('undo.png'), 'Undo', self)
        act(undoAct, 'Ctrl+Z', 'Undo', \
            lambda: [self.MyProgram.undo(), self.imshowData()])
        
        redoAct = QAction(QIcon('redo.png'), 'Redo', self)
        act(redoAct, 'Ctrl+R', 'Redo', \
            lambda: [self.MyProgram.redo(), self.imshowData()])

        dispersionAct = QAction(QIcon('dasQt/pic/dispersion.jpeg'), 'Dispersion', self)
        act(dispersionAct, 'Ctrl+D', 'Dispersion', \
            lambda: [self.dispersion()])
        
        saveCCAct = QAction(QIcon('dasQt/pic/saveCC.png'), 'Save CC', self)
        act(saveCCAct, 'Ctrl+S', 'Save File', \
            lambda: [self.saveCC()])
        
        saveDispersionAct = QAction(QIcon('dasQt/pic/saveDispersion.png'), 'Save Dispersion', self)
        act(saveDispersionAct, 'Ctrl+S', 'Save File', \
            lambda: [self.saveDispersion()])
        
        
        Inversion1DAct = QAction(QIcon('dasQt/pic/inversion1D.png'), 'Inversion1D', self)
        act(Inversion1DAct, 'Ctrl+I', 'Inversion1D', \
            lambda: [self.inversion1D()])

        CogAct = QAction(QIcon('dasQt/pic/cog.png'), 'Cog', self)
        act(CogAct, 'Ctrl+C', 'Cog', \
            lambda: [self.cog()])

        aboutAct = QAction(QIcon('about.png'), 'About ', self)
        act(aboutAct, 'Ctrl+U', 'About', \
            lambda: [QMessageBox.about(self, "About", about())])

        processAct = QAction(QIcon('process.png'), 'Process', self)
        act(processAct, 'Ctrl+P', 'Process', \
            lambda: [self.process()])

        YoloAct = QAction(QIcon('dasQt/pic/yolo.png'), 'Yolo', self)
        act(YoloAct, 'Ctrl+Y', 'Yolo', \
            lambda: [self.yolo()])


        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        fileMenu.addAction(openAct)
        fileMenu.addAction(openFolderAct)
        fileMenu.addAction(saveAct)

        ProcessMenu = menubar.addMenu('Process')
        ProcessMenu.addAction(processAct)

        DispersionMenu = menubar.addMenu('Dispersion')
        DispersionMenu.addAction(dispersionAct)
        DispersionMenu.addAction(saveCCAct)
        DispersionMenu.addAction(saveDispersionAct)

        Inversion1DMenu = menubar.addMenu('Inversion1D')
        Inversion1DMenu.addAction(Inversion1DAct)

        CogMenu = menubar.addMenu('Cog')
        CogMenu.addAction(CogAct)

        YoloMenu = menubar.addMenu('Yolo')
        YoloMenu.addAction(YoloAct)

        HelpMenu = menubar.addMenu('Help')
        HelpMenu.addAction(aboutAct)

    ##################
    ## Menu Fuction ##
    ##################
    def openFolder(self):
        folderName = QFileDialog.getExistingDirectory(self, "Select Directory", "./")
        self.folderName = folderName

        emptyFolderName = ''
        i = 0
        if folderName == emptyFolderName:
            if i < 3:
                # 创建并显示消息框
                QMessageBox.warning(self, "Message", "Folder is empty! Please select again.")
                self.logger.error('No folder selected!')
                self.openFolder()
                i += 1
            else:
                self.logger.error('No folder selected!')
                return
        else:
            files = self.MyProgram.openFolder(folderName)
            self.list_widget.clear()
            files = sorted(files)
            for file in files:
                self.list_widget.addItem(file)

    def importData(self):
        filetypes = 'All (*.mat *.h5 *.dat);;HDF5 (*.h5)'
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', filetypes)
        self.logger.info('Importing data from ' + fname)

        self.MyProgram.readData(fname)

    def readNextData(self):
        self.readNextDataBool = True
        print('readNextData')
        #_thread.start_new_thread(self.readNextData, ())
        try:
            self.MyProgram.readNextData()
        except Exception as e:
            self.logger.error(e)
            QMessageBox.warning(self, "Message", "No more data!")
            return

        if self.filterBool:
            self.MyProgram.bandpassData(self.fmin, self.fmax)
        if self.cutDataBool:
            self.MyProgram.cutData(self.Xmin, self.Xmax)

        self.indexTime = 0
        self.fIndex += 1


    ################
    # Init Control #
    ################
    def initControl(self, widControl: QWidget) -> None:
        # Control
        controlLayout = QVBoxLayout()
        widControl.setLayout(controlLayout)

        # Folder
        list_widget = QListWidget(self)
        list_widget.itemClicked.connect(self.clickList)
        widFolder = QGroupBox("", self)
        LayoutFolder = QVBoxLayout()
        LayoutFolder.addWidget(list_widget, 1)
        widFolder.setLayout(LayoutFolder)
        controlLayout.addWidget(widFolder, 2)
        self.list_widget = list_widget

        # Animation 
        widAnimation = QGroupBox("", self)
        LayoutAnimation = QVBoxLayout()
        widAnimation.setLayout(LayoutAnimation)
        
        widGrid = QWidget()
        grid = QGridLayout()
        grid.setSpacing(10)
        widGrid.setLayout(grid)
        
        widSlider = QWidget()
        sliderLayout = QGridLayout()
        sliderLayout.setSpacing(10)
        widSlider.setLayout(sliderLayout)
        
        LayoutAnimation.addWidget(widGrid, 0)
        LayoutAnimation.addWidget(widSlider, 0)
        controlLayout.addWidget(widAnimation, 0)

        btnStarAnimation = QPushButton('start Animation', self)
        btnStarAnimation.setToolTip('This is a <b>QPushButton</b> widget')
        btnStarAnimation.clicked.connect(lambda: [self.startAnimation(tabNum=self.tabNum)])
        grid.addWidget(btnStarAnimation, 1, 0)
        btnStarAnimation.setObjectName("button1")  # 设置对象名称

        btnStopAnimation = QPushButton('Stop Animation', self)
        btnStopAnimation.setToolTip('This is a <b>QPushButton</b> widget')
        btnStopAnimation.clicked.connect(lambda: [self.stopAnimation()])
        grid.addWidget(btnStopAnimation, 1, 1)
        btnStopAnimation.setObjectName("button1")  # 设置对象名称

        btnNextTime = QPushButton('Next Time', self)
        btnNextTime.setToolTip('This is a <b>QPushButton</b> widget')
        btnNextTime.clicked.connect(lambda: [self.nextTime()])
        grid.addWidget(btnNextTime, 2, 0)
        btnNextTime.setObjectName("button1")  # 设置对象名称

        btnPrevTime = QPushButton('Prev Time', self)
        btnPrevTime.setToolTip('This is a <b>QPushButton</b> widget')
        btnPrevTime.clicked.connect(lambda: [self.prevTime()])
        grid.addWidget(btnPrevTime, 2, 1)
        btnPrevTime.setObjectName("button1")  # 设置对象名称

        btnNextData = QPushButton('Next Data', self)
        btnNextData.setToolTip('This is a <b>QPushButton</b> widget')
        btnNextData.clicked.connect(
            lambda: [self.readNextData(), self.imshowData()])
        grid.addWidget(btnNextData, 3, 0)
        btnNextData.setObjectName("button2")  # 设置对象名称

        # 创建下拉菜单
        labColorMap = QLabel('ColorMap', self)
        self.combo_box = QComboBox()
        self.combo_box.addItem("rainbow")
        self.combo_box.addItem("RdBu")
        self.combo_box.addItem("seismic")
        self.combo_box.addItem("jet")
        sliderLayout.addWidget(labColorMap, 1, 0)
        sliderLayout.addWidget(self.combo_box, 1, 1)
        self.combo_box.activated.connect(self.on_combobox_activated) # 绑定事件

        # 创建滑块
        self.sliderLabel = QLabel("Speed: 1")
        self.slider      = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(30)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        sliderLayout.addWidget(self.sliderLabel, 2, 0)
        sliderLayout.addWidget(self.slider, 2, 1)

        self.sliderLabelFig = QLabel("Fig color scale: 10")
        self.sliderFig      = QSlider(Qt.Orientation.Horizontal)
        self.sliderFig.setMinimum(1)
        self.sliderFig.setMaximum(200)
        self.sliderFig.valueChanged.connect(self.sliderFigChanged)
        sliderLayout.addWidget(self.sliderLabelFig, 3, 0)
        sliderLayout.addWidget(self.sliderFig, 3, 1)



        # self.checkBox_fig = QCheckBox('Save Figure', self)
        # self.checkBox_fig.setChecked(False)
        # self.checkBox_fig.stateChanged.connect(self.checkBox_fig_Changed)
        # controlLayout.addWidget(self.checkBox_fig, 2)
        
        # btnYolo = QPushButton('YOLO', self)
        # btnYolo.setToolTip('This is a <b>QPushButton</b> widget')
        # btnYolo.clicked.connect(
        #     lambda: [self.imshowYolo()])
        # controlLayout.addWidget(btnYolo, 2)

    #####################
    ## Folder Function ##
    #####################
    def clickList(self, item):
        self.MyProgram.getFileID(item.text())
        self.indexTime = 0
        try:
            self.MyProgram.readData(os.path.join(self.folderName, item.text()))
        except Exception as e:
            self.logger.error(e)
            return

        self.fig.clear(); ax1 = self.fig.add_subplot(111)
        ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
        self.canvas.draw()

    ########################
    ## Animation Function ##
    ########################
    def startAnimation(self, tabNum=0):
        """Start animation, help function of imshowAllData"""
        if tabNum == 0:
            self.timer.start(self.ms)
        elif tabNum == 1:
            self.timerWigb.start(self.ms)

    def stopAnimation(self):
        """Stop animation, help function of imshowAllData"""
        self.timer.stop()
        self.timerWigb.stop()

    def nextTime(self):
        self.indexTime += int(self.slider.value())
        if self.tabNum == 0:
            self.fig.clear(); ax1 = self.fig.add_subplot(111)
            ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
            self.canvas.draw()
        elif self.tabNum == 1:
            self.figWigb.clear(); ax1 = self.figWigb.add_subplot(111)
            ax1 = self.MyProgram.wigb(ax1, indexTime=self.indexTime, scale=self.sliderFig.value())
            self.canvasWigb.draw()

    def prevTime(self):
        self.indexTime -= int(self.slider.value())
        if self.tabNum == 0:
            self.fig.clear(); ax1 = self.fig.add_subplot(111)
            ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
            self.canvas.draw()
        elif self.tabNum == 1:
            self.figWigb.clear(); ax1 = self.figWigb.add_subplot(111)
            ax1 = self.MyProgram.wigb(ax1, indexTime=self.indexTime, scale=self.sliderFig.value())
            self.canvasWigb.draw()

    def sliderValueChanged(self, value):
        """Set speed of animation, help function of sliderSpeed"""
        self.sliderLabel.setText(f"Speed: {value}")
        self.startAnimation(tabNum=self.tabNum)

    def sliderFigChanged(self, value):
        """Set speed of animation, help function of sliderSpeed"""
        self.sliderLabelFig.setText(f"Fig color scale: {value}")
        self.MyProgram.scale = value
        self.fig.clear(); ax1 = self.fig.add_subplot(111)
        ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
        self.canvas.draw()

    ## TODO: comboBox
    def on_combobox_activated(self, index):
        self.colormap = self.combo_box.currentText()
        self.fig.clear(); ax1 = self.fig.add_subplot(111)
        ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
        self.canvas.draw()



    def process(self):
        self.processMainWindow = ProcessMainWindow(self.MyProgram)
        self.processMainWindow.show()

    def dispersion(self):
        self.dispersionMainWindow = DispersionMainWindow(self.MyProgram, self.sliderFig)
        self.dispersionMainWindow.show()
    
    def saveCC(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CC", "", 
                            ".npz (*.npz);;all (*)")
        if file_path:
            # self.MyProgram.saveCC(file_path)
            self.dispersionMainWindow.saveCC(file_path)
    
    def saveDispersion(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Dispersion", "", 
                            ".npz (*.npz);;all (*)")
        if file_path:
            # self.MyProgram.saveDispersion(file_path)
            self.dispersionMainWindow.saveDispersion(file_path)

    def inversion1D(self):
        self.inversion1DMainWindow = Inversion1DMainWindow(self.MyProgram, self.sliderFig, self.readNextData)
        self.inversion1DMainWindow.show()

    def cog(self):
        self.cogMainWindow = CogMainWindow(self.MyProgram)
        self.cogMainWindow.show()

    def yolo(self):
        self.yoloMainWindow = YoloMainWindow(self.MyProgram)
        self.yoloMainWindow.show()


    ####################
    ## Figure Fuction ##
    ####################
    def imshowData(self):
        """Display data in a figure"""
        self.indexTime = 0
        self.fig.clear(); ax1 = self.fig.add_subplot(111)
        ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
        self.indexTime += int(self.slider.value())
        self.canvas.draw()

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateFigure)
        self.timer.start(self.ms)

        return self.fig, ax1

    def updateFigure(self):
        """Update plot, help function of imshowAllData"""
        self.fig.clear(); ax1 = self.fig.add_subplot(111) 
        ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
        self.canvas.draw()

        display_nt = self.MyProgram.display_T / self.MyProgram.dt
        nt = self.MyProgram.nt
        self.indexTime += int(self.slider.value())

        if int(self.indexTime / self.MyProgram.dt / 100) >= nt - display_nt:
            self.stopAnimation()
            if self.readNextDataBool:
                self.readNextData()
                self.imshowData()

    def wigbShow(self):
        """Display data in a figure"""
        self.figWigb.clear(); ax1 = self.figWigb.add_subplot(111)
        ax1 = self.MyProgram.wigb(ax1, indexTime=self.indexTime)
        self.canvasWigb.draw()

        self.timerWigb = QTimer()
        self.timerWigb.timeout.connect(self.updateWigb)

        return self.figWigb, ax1

    def updateWigb(self):
        """Update plot, help function of imshowAllData"""
        self.figWigb.clear(); ax1 = self.figWigb.add_subplot(111)
        ax1 = self.MyProgram.wigb(ax1, indexTime=self.indexTime)
        self.canvasWigb.draw()

        display_nt = self.MyProgram.display_T / self.MyProgram.dt
        nt = self.MyProgram.nt
        self.indexTime += int(self.slider.value())
        if self.indexTime >= nt - display_nt:
            self.stopAnimation()

