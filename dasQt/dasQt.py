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
from PyQt6.QtCore import Qt, QTimer


from dasQt import about
import dasQt.das as das
from dasQt.CarClass.getPoints import GetPoints
from dasQt.utools.logPy3 import HandleLog
from dasQt.CrossCorrelation.showDispersion import DispersionMainWindow
from dasQt.Cog.showCog import CogMainWindow
from dasQt.CarClass.showCarClass import CarClassMainWindow
from dasQt.YOLO.showYolo import YoloMainWindow
from dasQt.filter.showTrace import TraceMainWindow
from dasQt.utools import dasStartGUI

# define a global variable
global MyProgram1
MyProgram1 = das.DAS()






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

        self.layout    = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)


        # 设置主窗口的标题和图标
        self.setWindowTitle('DAS Show')
        self.setWindowIcon(QIcon('web.png'))
        QToolTip.setFont(QFont('Times New Roman', 10))
        self.setToolTip('This is <b>DAS Show</b> GUI')
        self.initMenu()

        tabFigure = QTabWidget()
        tabFigure.tabBarClicked.connect(self.tabBarClicked) 
        self.initFigure(tabFigure)
        # self.initFigureAll(tabFigure)
        self.initWigb(tabFigure)
        self.layout.addWidget(tabFigure, 3)
        self.figTabWidget = tabFigure

        tabControl = QTabWidget()
        self.layout.addWidget(tabControl, 1)
        self.initControl(tabControl)
        self.initProcess(tabControl)
        self.initDispersion(tabControl)
        self.initcarClass(tabControl)
        # self.initRadon(tabControl)
        self.initCog(tabControl)

        self.timer      = QTimer()
        self.timer.timeout.connect(self.updatePlot)
        self.timerWigb  = QTimer()
        self.timerWigb.timeout.connect(self.updateWigb)

        self.show()


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


    # def initFigureAll(self, tabFigure):
    #     self.figAll    = Figure()
    #     self.canvasAll = FigureCanvas(self.figAll)
    #     toolbar        = NavigationToolbar(self.canvasAll, self)

    #     widFig         = QWidget()
    #     newLayout      = QVBoxLayout()
    #     widFig.setLayout(newLayout)
    #     newLayout.addWidget(self.canvasAll, 0)
    #     newLayout.addWidget(toolbar, 0)
    #     tabFigure.addTab(widFig, "Fig")


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


    def initMenu(self) -> None:
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        EditMenu = menubar.addMenu('Edit')
        # ViewMenu = menubar.addMenu('View')
        # ToolsMenu = menubar.addMenu('Tools')
        HelpMenu = menubar.addMenu('Help')

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
        
        saveAct = QAction(QIcon('save.png'), 'Save', self)
        act(saveAct, 'Ctrl+S', 'Save File', \
            lambda: [self.saveCC(self.MyProgram)])
        
        undoAct = QAction(QIcon('undo.png'), 'Undo', self)
        act(undoAct, 'Ctrl+Z', 'Undo', \
            lambda: [self.MyProgram.undo(), self.imshowData()])
        
        redoAct = QAction(QIcon('redo.png'), 'Redo', self)
        act(redoAct, 'Ctrl+R', 'Redo', \
            lambda: [self.MyProgram.redo(), self.imshowData()])
        
        aboutAct = QAction(QIcon('about.png'), 'About ', self)
        act(aboutAct, 'Ctrl+U', 'About', \
            lambda: [QMessageBox.about(self, "About", about())])
        
        fileMenu.addAction(openAct)
        fileMenu.addAction(openFolderAct)
        fileMenu.addAction(saveAct)
        EditMenu.addAction(undoAct)
        EditMenu.addAction(redoAct)
        # ViewMenu.addAction(aboutAct)
        # ViewMenu.addAction(saveAct)
        HelpMenu.addAction(aboutAct)



    def initControl(self, tabControl: QTabWidget) -> None:
        # Control
        widControl = QGroupBox("Control", self)
        controlLayout = QVBoxLayout()
        widControl.setLayout(controlLayout)
        tabControl.addTab(widControl, "Control")


        # Folder
        list_widget = QListWidget(self)
        list_widget.itemClicked.connect(self.clickList)
        widFolder = QGroupBox("Folder", self)
        LayoutFolder = QVBoxLayout()
        LayoutFolder.addWidget(list_widget, 1)
        widFolder.setLayout(LayoutFolder)
        controlLayout.addWidget(widFolder, 2)
        self.list_widget = list_widget

        # Animation 
        # 创建一个QGroupBox，并设置标题
        widAnimation = QGroupBox("Animation", self)
        grid = QGridLayout()
        grid.setSpacing(10)
        widAnimation.setLayout(grid)
        controlLayout.addWidget(widAnimation, 0)


        btnStarAnimation = QPushButton('start Animation', self)
        btnStarAnimation.setToolTip('This is a <b>QPushButton</b> widget')
        btnStarAnimation.clicked.connect(lambda: [self.startAnimation(tabNum=self.tabNum)])
        grid.addWidget(btnStarAnimation, 1, 0)

        btnStopAnimation = QPushButton('Stop Animation', self)
        btnStopAnimation.setToolTip('This is a <b>QPushButton</b> widget')
        btnStopAnimation.clicked.connect(lambda: [self.stopAnimation()])
        grid.addWidget(btnStopAnimation, 1, 1)
        


        btnNextTime = QPushButton('Next Time', self)
        btnNextTime.setToolTip('This is a <b>QPushButton</b> widget')
        btnNextTime.clicked.connect(lambda: [self.nextTime()])
        grid.addWidget(btnNextTime, 2, 0)

        btnPrevTime = QPushButton('Prev Time', self)
        btnPrevTime.setToolTip('This is a <b>QPushButton</b> widget')
        btnPrevTime.clicked.connect(lambda: [self.prevTime()])
        grid.addWidget(btnPrevTime, 2, 1)

        self.sliderLabel = QLabel("Speed: 1")
        self.slider      = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(30)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        grid.addWidget(self.sliderLabel, 3, 0)
        grid.addWidget(self.slider, 3, 1)

        self.sliderLabelFig = QLabel("Fig color scale: 10")
        self.sliderFig      = QSlider(Qt.Orientation.Horizontal)
        self.sliderFig.setMinimum(1)
        self.sliderFig.setMaximum(200)
        self.sliderFig.valueChanged.connect(self.sliderFigChanged)
        grid.addWidget(self.sliderLabelFig, 4, 0)
        grid.addWidget(self.sliderFig, 4, 1)

        labColorMap = QLabel('ColorMap', self)
        # 创建下拉菜单
        self.combo_box = QComboBox()
        self.combo_box.addItem("rainbow")
        self.combo_box.addItem("RdBu")
        self.combo_box.addItem("seismic")
        self.combo_box.addItem("jet")
        grid.addWidget(labColorMap, 5, 0)
        grid.addWidget(self.combo_box, 5, 1)
        # 连接信号
        self.combo_box.activated.connect(self.on_combobox_activated)

        btnNextData = QPushButton('Next Data', self)
        btnNextData.setToolTip('This is a <b>QPushButton</b> widget')
        btnNextData.clicked.connect(
            lambda: [self.readNextData(), self.imshowData()])
        controlLayout.addWidget(btnNextData, 2)
        
        btnPickPoint = QPushButton('Pick Point', self)
        btnPickPoint.setToolTip('This is a <b>QPushButton</b> widget')
        btnPickPoint.clicked.connect(
            lambda: [self.getPoints()])
        controlLayout.addWidget(btnPickPoint, 2)
        
        self.checkBox_fig = QCheckBox('Save Figure', self)
        self.checkBox_fig.setChecked(False)
        self.checkBox_fig.stateChanged.connect(self.checkBox_fig_Changed)
        controlLayout.addWidget(self.checkBox_fig, 2)
        
        btnYolo = QPushButton('YOLO', self)
        btnYolo.setToolTip('This is a <b>QPushButton</b> widget')
        btnYolo.clicked.connect(
            lambda: [self.imshowYolo()])
        controlLayout.addWidget(btnYolo, 2)

    def initProcess(self, TabWidget: QTabWidget) -> None:
        widProcess = QGroupBox("Process", self)
        processLayout = QVBoxLayout()
        widProcess.setLayout(processLayout)
        TabWidget.addTab(widProcess, "Process")

        self.availableModulesList = QListWidget()
        self.selectedModulesList = QListWidget()
        self.availableModulesList.addItems(["Cut",
                                            "DownSampling",
                                            "Filter",
                                            "SignalTrace",
                                            ])

        self.addButton = QPushButton("Add to selected")
        self.addButton.clicked.connect(self.addModule)

        layout = QVBoxLayout()
        layout.addWidget(self.availableModulesList)
        layout.addWidget(self.addButton)
        layout.addWidget(self.selectedModulesList)
        widList = QGroupBox("List", self)
        widList.setLayout(layout)
        processLayout.addWidget(widList, 0)

        self.selectedModulesList.itemClicked.connect(self.showOptionWindow)


    def initcarClass(self, TabWidget: QTabWidget) -> None:
        widBtn = QGroupBox("Car Class", self)
        carClassLayout = QVBoxLayout()
        widBtn.setLayout(carClassLayout)
        TabWidget.addTab(widBtn, "Car Class")

        grid1 = QGridLayout()
        widCut = QGroupBox(self)
        widCut.setLayout(grid1)
        carClassLayout.addWidget(widCut, 0)
        grid1.setSpacing(10)
        labThreshold       = QLabel('threshold', self)
        labSkipNch         = QLabel('skip_Nch', self)
        labSkipNt          = QLabel('skip_Nt', self)
        labMode            = QLabel('Mode', self)
        self.editThreshold = QLineEdit('0.08', self)
        self.editSkipNch   = QLineEdit('1', self)
        self.editSkipNt    = QLineEdit('200', self)
        self.editMode      = QLineEdit('max', self)
        grid1.addWidget(labThreshold, 1, 0)
        grid1.addWidget(self.editThreshold, 1, 1)
        grid1.addWidget(labSkipNch, 2, 0)
        grid1.addWidget(self.editSkipNch, 2, 1)
        grid1.addWidget(labSkipNt, 3, 0)
        grid1.addWidget(self.editSkipNt, 3, 1)
        grid1.addWidget(labMode, 4, 0)
        grid1.addWidget(self.editMode, 4, 1)

        grid2 = QGridLayout()
        widSeparation = QGroupBox("Auto Separation", self)
        LayoutSeparation = QVBoxLayout()
        widSeparation.setLayout(grid2)
        carClassLayout.addWidget(widSeparation, 0)
        grid2.setSpacing(10)
        labMaxMode         = QLabel('MaxMode', self)
        labMinCarNum       = QLabel('MinCarNum', self)
        labTo              = QLabel('to', self)
        labLine            = QLabel('line', self)
        self.editMaxMode   = QLineEdit('10', self)
        self.editMinCarNum = QLineEdit('15', self)
        self.editTo        = QLineEdit('0.05', self)
        self.editLine      = QLineEdit('0.65', self)
        grid2.addWidget(labMaxMode, 1, 0)
        grid2.addWidget(self.editMaxMode, 1, 1)
        grid2.addWidget(labMinCarNum, 2, 0)
        grid2.addWidget(self.editMinCarNum, 2, 1)
        grid2.addWidget(labTo, 3, 0)
        grid2.addWidget(self.editTo, 3, 1)
        grid2.addWidget(labLine, 4, 0)
        grid2.addWidget(self.editLine, 4, 1)

        btnNextData = QPushButton('Next Data', self)
        btnNextData.setToolTip('This is a <b>QPushButton</b> widget')
        btnNextData.clicked.connect(
            lambda: [self.readNextData(), self.imshowCarClass()])
        carClassLayout.addWidget(btnNextData, 2)

        btnCarClass = QPushButton('Car Class', self)
        btnCarClass.setToolTip('This is a <b>QPushButton</b> widget')
        btnCarClass.clicked.connect(
            lambda: [self.imshowCarClass()])
        carClassLayout.addWidget(btnCarClass, 2)


    def initDispersion(self, TabWidget: QTabWidget) -> None:
        widAll = QWidget()
        layoutAll = QVBoxLayout()
        widAll.setLayout(layoutAll)
        
        widBtn = QGroupBox("CC", self)
        dispersionLayout = QVBoxLayout()
        widBtn.setLayout(dispersionLayout)
        layoutAll.addWidget(widBtn, 0)
        # TabWidget.tabBarClicked.connect(lambda: self.figTabWidget.setCurrentIndex(2))
        TabWidget.addTab(widAll, "Dispersion")

        labFmin                = QLabel('fmin', self)
        labFmax                = QLabel('fmax', self)
        labSmooth_N            = QLabel('smooth_N', self)
        labSmoothspect_N       = QLabel('smoothspect_N', self)
        labMaxlag              = QLabel('maxlag', self)
        labCC_len              = QLabel('CC_len', self)
        labCh1                 = QLabel('Xmin', self)
        labCh2                 = QLabel('Xmax', self)
        self.editFmin          = QLineEdit('1', self)
        self.editFmax          = QLineEdit('30', self)
        self.editSmooth_N      = QLineEdit('5', self)
        self.editSmoothspect_N = QLineEdit('5', self)
        self.editMaxlag        = QLineEdit('0.2', self)
        self.editCC_len        = QLineEdit('5', self)
        self.editCh1           = QLineEdit('41', self)
        self.editCh2           = QLineEdit('90', self)

        grid1 = QGridLayout()
        grid1.setSpacing(10)
        grid1.addWidget(labFmin, 1, 0)
        grid1.addWidget(self.editFmin, 1, 1)
        grid1.addWidget(labFmax, 2, 0)
        grid1.addWidget(self.editFmax, 2, 1)

        grid2 = QGridLayout()
        grid2.setSpacing(10)
        grid2.addWidget(labSmooth_N, 1, 0)
        grid2.addWidget(self.editSmooth_N, 1, 1)
        grid2.addWidget(labSmoothspect_N, 2, 0)
        grid2.addWidget(self.editSmoothspect_N, 2, 1)
        grid2.addWidget(labMaxlag, 3, 0)
        grid2.addWidget(self.editMaxlag, 3, 1)
        
        grid3 = QGridLayout()
        grid3.setSpacing(10)
        grid3.addWidget(labCC_len, 1, 0)
        grid3.addWidget(self.editCC_len, 1, 1)
        grid3.addWidget(labCh1, 2, 0)
        grid3.addWidget(self.editCh1, 2, 1)
        grid3.addWidget(labCh2, 3, 0)
        grid3.addWidget(self.editCh2, 3, 1)

        widFilter = QGroupBox("Filter", self)
        widFilter.setLayout(grid1)
        dispersionLayout.addWidget(widFilter, 1)

        widSmooth = QGroupBox("Smooth", self)
        widSmooth.setLayout(grid2)
        dispersionLayout.addWidget(widSmooth, 1)

        widCC = QGroupBox("CC", self)
        widCC.setLayout(grid3)
        dispersionLayout.addWidget(widCC, 1)

        btnCC = QPushButton('noise cc', self)
        btnCC.setToolTip('This is a <b>QPushButton</b> widget')
        btnCC.clicked.connect(
            lambda: [self.imshowCC()])  # , self.imshowDataAll()

        btnNextData = QPushButton('Next Data', self)
        btnNextData.setToolTip('This is a <b>QPushButton</b> widget')
        btnNextData.clicked.connect(
            lambda: [self.readNextData(), self.imshowCC(),self.saveCC()])   # , self.imshowDataAll()

        btnNext20Data = QPushButton('Next 20 Data', self)
        btnNext20Data.setToolTip('This is a <b>QPushButton</b> widget')
        btnNext20Data.clicked.connect(
            lambda: [self.readNext20Data()])
        self.editNext20Data = QLineEdit('20', self)
        
        self.checkBox = QCheckBox('Save CC', self)
        self.checkBox.setChecked(False)
        self.checkBox.stateChanged.connect(self.checkBoxChanged)

        btnCCAll = QPushButton('Dispersion All', self)
        btnCCAll.setToolTip('This is a <b>QPushButton</b> widget')
        btnCCAll.clicked.connect(
            lambda: [self.CCAll()])
        dispersionLayout.addWidget(btnCCAll, 2)

        layGrid = QGridLayout()
        layGrid.setSpacing(5)
        layGrid.addWidget(btnCC, 1, 0)
        layGrid.addWidget(btnNextData, 1, 1)
        layGrid.addWidget(btnNext20Data, 2, 0)
        layGrid.addWidget(self.editNext20Data, 2, 1)
        layGrid.addWidget(self.checkBox, 3, 0)
        layGrid.addWidget(btnCCAll, 3, 1)

        widGrid = QWidget()
        widGrid.setLayout(layGrid)
        dispersionLayout.addWidget(widGrid, 1)




        widDispersion = QGroupBox("Dispersion", self)
        dispersionLayout1 = QVBoxLayout()
        widDispersion.setLayout(dispersionLayout1)
        layoutAll.addWidget(widDispersion, 0)
        
        widGroup = QGroupBox("Dispersion Parameters", self)
        dispersionLayout1.addWidget(widGroup, 0)
        labCmin = QLabel('Cmin', self)
        labCmax = QLabel('Cmax', self)
        labdc  = QLabel('dc', self)
        labfmin = QLabel('fmin', self)
        labfmax = QLabel('fmax', self)
        self.editCmin          = QLineEdit('10.0', self)
        self.editCmax          = QLineEdit('1000.0', self)
        self.editdc            = QLineEdit('5.0', self)
        self.editfmin_dispersion          = QLineEdit('0.5', self)
        self.editfmax_dispersion          = QLineEdit('30.0', self)
        
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(labCmin, 1, 0)
        grid.addWidget(self.editCmin, 1, 1)
        grid.addWidget(labCmax, 2, 0)
        grid.addWidget(self.editCmax, 2, 1)
        grid.addWidget(labdc, 3, 0)
        grid.addWidget(self.editdc, 3, 1)
        grid.addWidget(labfmin, 4, 0)
        grid.addWidget(self.editfmin_dispersion, 4, 1)
        grid.addWidget(labfmax, 5, 0)
        grid.addWidget(self.editfmax_dispersion, 5, 1)
        widGroup.setLayout(grid)
        
        
        
        
        self.btnUp = QPushButton('Up CC', self)
        self.btnUp.setCheckable(True)
        self.btnUp.setToolTip('This is a <b>QPushButton</b> widget')
        self.btnUp.clicked.connect(lambda: [self.getUpCC()])
        
        self.btnDown = QPushButton('Down CC', self)
        self.btnDown.setCheckable(True)
        self.btnDown.setToolTip('This is a <b>QPushButton</b> widget')
        self.btnDown.clicked.connect(lambda: [self.getDownCC()])
        
        btnGetDispersion = QPushButton('Get Dispersion', self)
        btnGetDispersion.setToolTip('This is a <b>QPushButton</b> widget')
        btnGetDispersion.clicked.connect(lambda: [self.imshowDispersion(
            float(self.editCmin.text()), float(self.editCmax.text()), float(self.editdc.text()),
            float(self.editfmin_dispersion.text()), float(self.editfmax_dispersion.text()))])

        btnNextData = QPushButton('Next Data', self)
        btnNextData.setToolTip('This is a <b>QPushButton</b> widget')
        btnNextData.clicked.connect(
            lambda: [self.readNextData(), self.imshowCC(), 
                     self.imshowDispersion(float(self.editCmin.text()), float(self.editCmax.text()), float(self.editdc.text()),
                                        float(self.editfmin_dispersion.text()), float(self.editfmax_dispersion.text()))])

        btnNext20Data = QPushButton('Next 20 Data', self)
        btnNext20Data.setToolTip('This is a <b>QPushButton</b> widget')
        btnNext20Data.clicked.connect(
            lambda: [self.readNext20DataDispersion()])
        self.editNext20DataDispersion = QLineEdit('20', self)
        
        self.checkBoxDispersion = QCheckBox('Save Dispersion', self)
        self.checkBoxDispersion.setChecked(False)
        self.checkBoxDispersion.stateChanged.connect(self.checkBoxChangedSaveDispersion)

        btnCCAll = QPushButton('Dispersion All', self)
        btnCCAll.setToolTip('This is a <b>QPushButton</b> widget')
        btnCCAll.clicked.connect(
            lambda: [self.imshowDispersionAll(float(self.editCmin.text()), float(self.editCmax.text()), float(self.editdc.text()),
                                        float(self.editfmin_dispersion.text()), float(self.editfmax_dispersion.text()))])
        
        
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(self.btnUp, 1, 0)
        grid.addWidget(self.btnDown, 1, 1)
        grid.addWidget(btnGetDispersion, 2, 0)
        grid.addWidget(btnNextData, 2, 1)
        grid.addWidget(btnNext20Data, 3, 0)
        grid.addWidget(self.editNext20DataDispersion, 3, 1)
        grid.addWidget(self.checkBoxDispersion, 4, 0)
        grid.addWidget(btnCCAll, 4, 1)
        
        widGrid = QWidget()
        widGrid.setLayout(grid)
        
        dispersionLayout1.addWidget(widGrid, 0)


        # dispersionLayout1.addWidget(btnUp, 1)
        # dispersionLayout1.addWidget(btnDown, 1)
        # dispersionLayout1.addWidget(btnGetDispersion, 1)
        # dispersionLayout1.addWidget(btnNextData, 1)


    def initCog(self, TabWidget: QTabWidget) -> None:
        widBtn = QGroupBox("Cog", self)
        dispersionLayout = QVBoxLayout()
        widBtn.setLayout(dispersionLayout)
        TabWidget.addTab(widBtn, "Cog")


        labWin = QLabel('win', self)
        labNwin = QLabel('nwin', self)
        labOverlap = QLabel('overlap', self)
        labOffset = QLabel('offset', self)
        labFminCog        = QLabel('fmin', self)
        labFminCog1       = QLabel('fmin1', self)
        labFmaxCog        = QLabel('fmax', self)
        labFmaxCog1       = QLabel('fmax1', self)
        labNdata         = QLabel('Ndata', self)
        editWin = QLineEdit('300', self)
        editNwin = QLineEdit('40', self)
        editOverlap = QLineEdit('300', self)
        editOffset = QLineEdit('1', self)
        editFminCog  = QLineEdit('2.0', self)
        editFminCog1 = QLineEdit('3.0', self)
        editFmaxCog  = QLineEdit('20.0', self)
        editFmaxCog1 = QLineEdit('21.0', self)
        editNdata    = QLineEdit('100', self)

        grid1 = QGridLayout()
        grid1.setSpacing(10)
        grid1.addWidget(labWin, 1, 0)
        grid1.addWidget(editWin, 1, 1)
        grid1.addWidget(labNwin, 2, 0)
        grid1.addWidget(editNwin, 2, 1)
        grid1.addWidget(labOverlap, 3, 0)
        grid1.addWidget(editOverlap, 3, 1)
        grid1.addWidget(labOffset, 4, 0)
        grid1.addWidget(editOffset, 4, 1)
        grid1.addWidget(labFminCog, 5, 0)
        grid1.addWidget(editFminCog, 5, 1)
        grid1.addWidget(labFminCog1, 6, 0)
        grid1.addWidget(editFminCog1, 6, 1)
        grid1.addWidget(labFmaxCog, 7, 0)
        grid1.addWidget(editFmaxCog, 7, 1)
        grid1.addWidget(labFmaxCog1, 8, 0)
        grid1.addWidget(editFmaxCog1, 8, 1)
        grid1.addWidget(labNdata, 9, 0)
        grid1.addWidget(editNdata, 9, 1)
        widGrid = QWidget()
        widGrid.setLayout(grid1)
        dispersionLayout.addWidget(widGrid, 0)
        
        btnCog = QPushButton('Cog', self)
        btnCog.setToolTip('This is a<b>QPushButton</b> widget')
        btnCog.clicked.connect(lambda: [self.imshowCog(int(editNdata.text()), int(editWin.text()),int(editNwin.text()),
                                int(editOverlap.text()),int(editOffset.text()),
                                fmin1=float(editFminCog.text()),fmin2=float(editFminCog1.text()),
                                fmax1=float(editFmaxCog.text()), fmax2=float(editFmaxCog1.text()))])
        
        dispersionLayout.addWidget(btnCog, 1)
        
        
        
        labYlim = QLabel('Ylim', self)
        labVmin = QLabel('Vmin', self)
        labVmax = QLabel('Vmax', self)
        editYlim = QLineEdit('0.5', self)
        editVmin = QLineEdit('-0.25', self)
        editVmax = QLineEdit('0.25', self)
        grid2 = QGridLayout()
        grid2.setSpacing(10)
        grid2.addWidget(labYlim, 1, 0)
        grid2.addWidget(editYlim, 1, 1)
        grid2.addWidget(labVmin, 2, 0)
        grid2.addWidget(editVmin, 2, 1)
        grid2.addWidget(labVmax, 3, 0)
        grid2.addWidget(editVmax, 3, 1)
        widGrid = QWidget()
        widGrid.setLayout(grid2)
        dispersionLayout.addWidget(widGrid, 0)
        
        btnYlim = QPushButton('Ylim', self)
        btnYlim.setToolTip('This is a<b>QPushButton</b> widget')
        btnYlim.clicked.connect(lambda: [self.imshowCog(int(editNdata.text()),int(editWin.text()),int(editNwin.text()), 
                                        int(editOverlap.text()),int(editOffset.text()), 
                                        fmin1=float(editFminCog.text()), fmin2=float(editFminCog1.text()),
                                        fmax1=float(editFmaxCog.text()), fmax2=float(editFmaxCog1.text()), 
                                        ylim=float(editYlim.text()), vmin=float(editVmin.text()), vmax=float(editVmax.text()))])

        dispersionLayout.addWidget(btnYlim, 1)

    # def initRadon(self, TabWidget):
    #     widBtn = QGroupBox("Radon", self)
    #     dispersionLayout = QVBoxLayout()
    #     widBtn.setLayout(dispersionLayout)
    #     TabWidget.addTab(widBtn, "Radon")

    #     labFminRadon        = QLabel('fmin', self)
    #     labFminRadon1       = QLabel('fmin1', self)
    #     labFmaxRadon        = QLabel('fmax', self)
    #     labFmaxRadon1       = QLabel('fmax1', self)
    #     labVmaxRadon        = QLabel('Vmax', self)
    #     labVminRadon        = QLabel('Vmin', self)
    #     labDvRadon          = QLabel('dv', self)
    #     labDfRadon          = QLabel('df', self)
    #     labCh1Radon         = QLabel('Ch1', self)
    #     labCh2Radon         = QLabel('Ch2', self)
    #     self.editFminRadon  = QLineEdit('0.01', self)
    #     self.editFminRadon1 = QLineEdit('0.1', self)
    #     self.editFmaxRadon  = QLineEdit('1.0', self)
    #     self.editFmaxRadon1 = QLineEdit('2.0', self)
    #     self.editVmaxRadon  = QLineEdit('15', self)
    #     self.editVminRadon  = QLineEdit('-15', self)
    #     self.editDvRadon    = QLineEdit('0.1', self)
    #     self.editDfRadon    = QLineEdit('0.01', self)
    #     self.editCh1Radon   = QLineEdit('30', self)
    #     self.editCh2Radon   = QLineEdit('90', self)

    #     grid1 = QGridLayout()
    #     grid1.setSpacing(10)
    #     grid1.addWidget(labFminRadon, 1, 0)
    #     grid1.addWidget(self.editFminRadon, 1, 1)
    #     grid1.addWidget(labFminRadon1, 2, 0)
    #     grid1.addWidget(self.editFminRadon1, 2, 1)
    #     grid1.addWidget(labFmaxRadon, 3, 0)
    #     grid1.addWidget(self.editFmaxRadon, 3, 1)
    #     grid1.addWidget(labFmaxRadon1, 4, 0)
    #     grid1.addWidget(self.editFmaxRadon1, 4, 1)

    #     btnBPFilter = QPushButton('BPFilter', self)
    #     btnBPFilter.setToolTip('This is a <b>QPushButton</b> widget')
    #     btnBPFilter.clicked.connect(
    #         lambda: [self.bpFilter()])

    #     widFilter = QGroupBox("Filter", self)
    #     widFilter.setLayout(grid1)
    #     dispersionLayout.addWidget(widFilter, 0)
    #     dispersionLayout.addWidget(btnBPFilter, 1)

    #     grid2 = QGridLayout()
    #     grid2.setSpacing(10)
    #     grid2.addWidget(labVminRadon, 1, 0)
    #     grid2.addWidget(self.editVminRadon, 1, 1)
    #     grid2.addWidget(labVmaxRadon, 2, 0)
    #     grid2.addWidget(self.editVmaxRadon, 2, 1)
    #     grid2.addWidget(labDvRadon, 3, 0)
    #     grid2.addWidget(self.editDvRadon, 3, 1)
    #     grid2.addWidget(labDfRadon, 4, 0)
    #     grid2.addWidget(self.editDfRadon, 4, 1)

    #     btnNormalTrace = QPushButton('NormalTrace', self)
    #     btnNormalTrace.setToolTip('This is a <b>QPushButton</b> widget')
    #     btnNormalTrace.clicked.connect(
    #         lambda: [self.norm_trace()])

    #     widSmooth = QGroupBox("Radon", self)
    #     widSmooth.setLayout(grid2)
    #     dispersionLayout.addWidget(widSmooth, 0)
    #     dispersionLayout.addWidget(btnNormalTrace, 1)

    #     grid3 = QGridLayout()
    #     grid3.setSpacing(10)
    #     grid3.addWidget(labCh1Radon, 1, 0)
    #     grid3.addWidget(self.editCh1Radon, 1, 1)
    #     grid3.addWidget(labCh2Radon, 2, 0)
    #     grid3.addWidget(self.editCh2Radon, 2, 1)

    #     widCC = QGroupBox("Channel", self)
    #     widCC.setLayout(grid3)
    #     dispersionLayout.addWidget(widCC, 0)

    #     btnNextData = QPushButton('Next Data', self)
    #     btnNextData.setToolTip('This is a <b>QPushButton</b> widget')
    #     btnNextData.clicked.connect(
    #         lambda: [self.readNextData(), 
    #                  self.bpFilter(), 
    #                  self.norm_trace(), 
    #                  self.imshowRadon(self.MyProgram)])
    #     dispersionLayout.addWidget(btnNextData, 2)

    #     btnRadon = QPushButton('Radon', self)
    #     btnRadon.setToolTip('This is a <b>QPushButton</b> widget')
    #     btnRadon.clicked.connect(
    #         lambda: [self.imshowRadon(self.MyProgram)])
    #     dispersionLayout.addWidget(btnRadon, 2)

    #     btnAllRadon = QPushButton('All Radon', self)
    #     btnAllRadon.setToolTip('This is a <b>QPushButton</b> widget')
    #     btnAllRadon.clicked.connect(
    #         lambda: [self.bpFilter(), self.norm_trace(), self.imshowRadon(self.MyProgram)])
    #     dispersionLayout.addWidget(btnAllRadon, 2)




    #-----------------------------------------------------------
    #
    #  Functions GUI
    #
    #-----------------------------------------------------------

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


    def addModule(self):
        # get the selected item
        selectedItem = self.availableModulesList.currentItem()
        # add the selected item to the selectedModulesList
        if selectedItem:
            self.selectedModulesList.addItem(selectedItem.text())


    def showOptionWindow(self, item):
        moduleName = item.text()
        # selecet the option window according to the module name
        if moduleName == "Cut":
            optionWindow = OptionWindowCut(moduleName)
        elif moduleName == "DownSampling":
            optionWindow = OptionWindowDownSampling(moduleName)
        elif moduleName == "Filter":
            optionWindow = OptionWindowFilter(moduleName)
        elif moduleName == "SignalTrace":
            optionWindow = OptionWindowSignalTrace(moduleName)
        else:
            # option window for default, or prompt that no option is defined for this module
            optionWindow = QDialog()
            optionWindow.setWindowTitle("Option Window")
            layout = QVBoxLayout()
            label = QLabel("No option is defined for this module")
            layout.addWidget(label)
            optionWindow.setLayout(layout)

        # self.imshowDataAll()
        optionWindow.exec()


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

        # if self.figAll:
        #     self.imshowDataAll()


    # def wheelEvent(self, event):
    #     delta = event.angleDelta().y()  # 获取滚轮滚动的距离
    #     print(delta)
    #     if delta > 0:
    #         self.prevTime()
    #     elif delta < 0:
    #         self.nextTime()


    def tabBarClicked(self, index):
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


    def checkBoxChanged(self, state):
        if state == 0:
            self.bool_saveCC = False
            self.MyProgram.bool_saveCC = False
        else:
            self.bool_saveCC = True
            self.MyProgram.bool_saveCC = True
    
    def checkBox_fig_Changed(self, state):
        if state == 0:
            self.bool_saveFig = False
            self.MyProgram.bool_saveFig = False
        else:
            self.bool_saveFig = True
            self.MyProgram.bool_saveFig = True

    def checkBoxChangedSaveDispersion(self, state):
        if state == 0:
            self.bool_saveDispersion = False
            self.MyProgram.bool_saveDispersion = False
        else:
            self.bool_saveDispersion = True
            self.MyProgram.bool_saveDispersion = True

## TODO: comboBox
    def on_combobox_activated(self, index):
        self.colormap = self.combo_box.currentText()
        self.fig.clear(); ax1 = self.fig.add_subplot(111)
        ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
        self.canvas.draw()
        

    #-----------------------------------------------------------
    #
    #  Functions Data
    #
    #-----------------------------------------------------------

    def importData(self):
        filetypes = 'All (*.mat *.h5 *.dat);;HDF5 (*.h5)'
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', filetypes)
        self.logger.info('Importing data from ' + fname)

        self.MyProgram.readData(fname)


    def openFolder(self):
        folderName = QFileDialog.getExistingDirectory(self, "Select Directory", "./")
        self.folderName = folderName

        emptyFolderName = ''
        if folderName == emptyFolderName:
            # 创建并显示消息框
            QMessageBox.warning(self, "Message", "Folder is empty! Please select again.")
            self.logger.error('No folder selected!')
            self.openFolder()
        else:
            files = self.MyProgram.openFolder(folderName)
            self.list_widget.clear()
            files = sorted(files)
            for file in files:
                self.list_widget.addItem(file)


    def readNextData(self):
        self.readNextDataBool = True
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
        


    # TODO : CC data get up and down
    def getUpCC(self):
        if self.btnUp.isChecked():
            up = True
            self.MyProgram.getUpCC(up)
            self.btnUp.setText('Stop')
        else:
            up = False
            self.MyProgram.getUpCC(up)
            self.btnUp.setText('Up CC')
    
    def getDownCC(self):
        if self.btnDown.isChecked():
            down = True
            self.MyProgram.getDownCC(down)
            self.btnDown.setText('Stop')
        else:
            down = False
            self.MyProgram.getDownCC(down)
            self.btnDown.setText('Down CC')






    def imshowData(self):
        """Display data in a figure"""
        self.indexTime = 0
        self.fig.clear(); ax1 = self.fig.add_subplot(111)
        ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
        self.indexTime += int(self.slider.value())
        self.canvas.draw()

        self.timer = QTimer()
        self.timer.timeout.connect(self.updatePlot)
        self.timer.start(self.ms)

        return self.fig, ax1

    # def imshowDataAll(self):
    #     """Display data in a figure"""

    #     self.figAll.clear(); ax1 = self.figAll.add_subplot(111)
    #     ax1 = self.MyProgram.imshowDataAll(ax1)
    #     self.canvasAll.draw()

    #     return self.fig, ax1
    
    def imshowCarClass(self):
        """Display data in a figure"""
        threshold = float(self.editThreshold.text())
        skip_Nch  = int(self.editSkipNch.text())
        skip_Nt   = int(self.editSkipNt.text())
        maxMode   = int(self.editMaxMode.text())
        minCarNum = int(self.editMinCarNum.text())
        to        = float(self.editTo.text())
        mode      = self.editMode.text()
        line      = float(self.editLine.text())
        self.indexTime           = 0
        
        self.fig_car_class = CarClassMainWindow(self.MyProgram, title='Car Class')
        self.fig_car_class.imshowCarClass(scale=self.sliderFig.value(), 
                                           skip_Nch=skip_Nch, skip_Nt=skip_Nt, threshold=threshold,mode=mode,
                                          maxMode=maxMode, minCarNum=minCarNum, to=to, line=line)
        self.fig_car_class.show()


    def imshowCC(self):
        dispersion_parse = {
            'sps'           : 1/self.MyProgram.dt,    # current sampling rate
            'samp_freq'     : 1/self.MyProgram.dt,    # targeted sampling rate
            'freqmin'       : float(self.editFmin.text()),          # pre filtering frequency bandwidth预滤波频率带宽
            'freqmax'       : float(self.editFmax.text()),           # note this cannot exceed Nquist freq
            'freq_norm'     : 'rma',        # 'no' for no whitening, or 'rma' for running-mean average, 'phase_only' for sign-bit normalization in freq domain.
            'time_norm'     : 'one_bit',    # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain
            'cc_method'     : 'xcorr',      # 'xcorr' for pure cross correlation, 'deconv' for deconvolution;
            'smooth_N'      : int(self.editSmooth_N.text()),            # moving window length for time domain normalization if selected (points)
            'smoothspect_N' : int(self.editSmoothspect_N.text()),            # moving window length to smooth spectrum amplitude (points)
            'maxlag'        : float(self.editMaxlag.text()),          # lags of cross-correlation to save (sec)
            'max_over_std'  : 10**9,        # threahold to remove window of bad signals: set it to 10*9 if prefer not to remove them
            'cc_len'        : int(self.editCC_len.text()),            # correlate length in second(sec)
            'cha1'          : int(self.editCh1.text()),           # start channel index for the sub-array
            'cha2'          : int(self.editCh2.text()),           # end channel index for the sub-array
        }

        self.fig_CC = DispersionMainWindow(self.MyProgram, title='CC One')
        self.fig_CC.imshowCC(dispersion_parse, self.sliderFig.value())
        self.fig_CC.show()

    def imshowDispersion(self, Cmin, Cmax, dc, fmin, fmax):
        self.fig_dispersion = DispersionMainWindow(self.MyProgram, title='Dispersion One')
        self.fig_dispersion.imshowDispersion(Cmin, Cmax, dc, fmin, fmax)
        self.fig_dispersion.show()

    def imshowDispersionAll(self, Cmin, Cmax, dc, fmin, fmax):
        self.fig_dispersion = DispersionMainWindow(self.MyProgram, title='Dispersion All')
        self.fig_dispersion.imshowDispersionAll(Cmin, Cmax, dc, fmin, fmax)
        self.fig_dispersion.show()


    # def imshowRadon(self):
    #     radon_parse = {
    #         'fmin' : float(self.editFminRadon.text()),
    #         'fmin1': float(self.editFminRadon1.text()),
    #         'fmax' : float(self.editFmaxRadon.text()),
    #         'fmax1': float(self.editFmaxRadon1.text()),
    #         'Vmax' : float(self.editVmaxRadon.text()),
    #         'Vmin' : float(self.editVminRadon.text()),
    #         'dv'   : float(self.editDvRadon.text()),
    #         'df'   : float(self.editDfRadon.text()),
    #         'cha1' : int(self.editCh1Radon.text()),
    #         'cha2' : int(self.editCh2Radon.text()),
    #     }
    #     self.MyProgram.radon_parse = radon_parse

    #     self.fig.clear(); ax1 = self.fig.add_subplot(121); ax2 = self.fig.add_subplot(122)
    #     ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
    #     self.indexTime += int(self.slider.value())
    #     self.canvas.draw()
    #     self.ax1 = ax1

    #     ax2 = self.MyProgram.imshowRadon(ax2, indexTime=self.indexTime, scale=self.sliderFig.value())

    #     self.timer = QTimer()
    #     self.timer.timeout.connect(self.updateRadon)
    #     self.timer.start(self.ms)

    def imshowCog(self, Ndata, win, nwin, overlap, offset, fmin1=2., fmin2=3., fmax1=20., fmax2=21., ylim=None, vmin=-0.25, vmax=0.25):
        self.figCog = CogMainWindow(self.MyProgram, title='Cog')
        self.figCog.imshowCog(Ndata, win, nwin, overlap, offset, fmin1, fmin2, fmax1, fmax2, ylim, vmin, vmax)
        self.figCog.show()

    def imshowYolo(self):
        self.figYolo = YoloMainWindow(self.MyProgram, title='Yolo')
        self.figYolo.imshowYolo()
        self.figYolo.show()

    def updatePlot(self):
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


    # def updateCarClass(self):
    #     """Update plot, help function of imshowAllData"""
    #     ax1 = self.ax1; ax1.cla()
    #     ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
    #     self.canvas.draw()

    #     display_nt = self.MyProgram.display_T / self.MyProgram.dt
    #     nt = self.MyProgram.nt
    #     self.indexTime += int(self.slider.value())
    #     if self.indexTime >= nt - display_nt:
    #         self.stopAnimation()
    #         if self.readNextDataBool:
    #             self.readNextData()
    #             self.imshowData()


    # def updateRadon(self):
    #     """Update plot, help function of imshowAllData"""
    #     ax1 = self.ax1; ax1.cla()
    #     ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
    #     self.canvas.draw()

    #     display_nt = self.MyProgram.display_T / self.MyProgram.dt
    #     nt = self.MyProgram.nt
    #     self.indexTime += int(self.slider.value())
    #     if self.indexTime >= nt - display_nt:
    #         self.stopAnimation()
    #         if self.readNextDataBool:
    #             self.readNextData()
    #             self.imshowData()


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


    def rawData(self):
        self.filterBool  = False
        self.cutDataBool = False
        self.data = self.MyProgram.rawData()
        self.fig.clear(); ax1 = self.fig.add_subplot(111)
        ax1 = self.MyProgram.imshowData(ax1, indexTime=self.indexTime, colormap=self.colormap)
        self.canvas.draw()


    def getPoints(self):
        self.getPoints = GetPoints()
        self.getPoints.show()

    # def bpFilter(self):
    #     radon_parse = {
    #         'fmin' : float(self.editFminRadon.text()),
    #         'fmin1': float(self.editFminRadon1.text()),
    #         'fmax' : float(self.editFmaxRadon.text()),
    #         'fmax1': float(self.editFmaxRadon1.text()),
    #         'Vmax' : float(self.editVmaxRadon.text()),
    #         'Vmin' : float(self.editVminRadon.text()),
    #         'dv'   : float(self.editDvRadon.text()),
    #         'df'   : float(self.editDfRadon.text()),
    #         'cha1' : int(self.editCh1Radon.text()),
    #         'cha2' : int(self.editCh2Radon.text()),
    #     }
    #     self.MyProgram.radon_parse = radon_parse
    #     self.MyProgram.bpFilter()


    # def norm_trace(self):
    #     radon_parse = {
    #         'fmin' : float(self.editFminRadon.text()),
    #         'fmin1': float(self.editFminRadon1.text()),
    #         'fmax' : float(self.editFmaxRadon.text()),
    #         'fmax1': float(self.editFmaxRadon1.text()),
    #         'Vmax' : float(self.editVmaxRadon.text()),
    #         'Vmin' : float(self.editVminRadon.text()),
    #         'dv'   : float(self.editDvRadon.text()),
    #         'df'   : float(self.editDfRadon.text()),
    #         'cha1' : int(self.editCh1Radon.text()),
    #         'cha2' : int(self.editCh2Radon.text()),
    #     }
    #     self.MyProgram.radon_parse = radon_parse
    #     self.MyProgram.norm_trace()

    def readNext20Data(self):
        for i in range(int(self.editNext20Data.text())):
            self.readNextData()
            # self.imshowDataAll()
            self.imshowCC()
            self.saveCC()

    def readNext20DataDispersion(self):
        for i in range(int(self.editNext20Data.text())):
            self.readNextData()
            # self.imshowDataAll()
            self.imshowCC()
            # self.saveCC()
            self.imshowDispersion(
                float(self.editCmin.text()), float(self.editCmax.text()), float(self.editdc.text()),
                float(self.editfmin_dispersion.text()), float(self.editfmax_dispersion.text()))
            



    def CCAll(self):
        # self.fig_dispersionAll 一定要加 self ,这样主窗口才能接收到 这个新窗口
        self.fig_CCAll = DispersionMainWindow(self.MyProgram, title='CC All')
        self.fig_CCAll.imshowCCAll()
        self.fig_CCAll.show()

    def saveCC(self):
        if self.bool_saveCC:
            self.MyProgram.saveCC()
        else:
            pass


# TODO: add frequency spectrum



class OptionWindowCut(QDialog):
    def __init__(self, moduleName):
        super().__init__()
        self.MyProgram = MyProgram1
        self.setWindowTitle(f"{moduleName} option")
        self.layout = QVBoxLayout()
        self.label  = QLabel(f"{moduleName}: option ")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        # Fig
        LayoutFig = QVBoxLayout()
        widFig    = QGroupBox("Fig", self)
        widFig.setLayout(LayoutFig)
        self.layout.addWidget(widFig, 0)

        labXmin  = QLabel('Xmin', self)
        labXmax  = QLabel('Xmax', self)
        editXmin = QLineEdit('40', self)
        editXmax = QLineEdit('50', self)

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(labXmin, 1, 0)
        grid.addWidget(editXmin, 1, 1)
        grid.addWidget(labXmax, 2, 0)
        grid.addWidget(editXmax, 2, 1)
        widCut = QGroupBox(self)
        widCut.setLayout(grid)
        LayoutFig.addWidget(widCut, 0)

        btnCut = QPushButton('Cut', self)
        btnCut.setToolTip('Cut Data')
        btnCut.clicked.connect(
            lambda: [self.cutData(editXmin.text(), editXmax.text())])
        LayoutFig.addWidget(btnCut, 1)

    def cutData(self, Xmin, Xmax):
        self.MyProgram.bool_cut = True
        self.MyProgram.cutData(Xmin, Xmax)
        self.Xmin = Xmin; self.Xmax = Xmax

class OptionWindowDownSampling(QDialog):
    def __init__(self, moduleName):
        super().__init__()
        self.MyProgram = MyProgram1
        self.setWindowTitle(f"{moduleName} option")
        self.layout = QVBoxLayout()
        self.label  = QLabel(f"{moduleName}: option")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        intNumDownSampling = QLineEdit('10', self)
        btnDownSampling    = QPushButton('Down Sampling', self)
        btnDownSampling.setToolTip('Down Sampling')
        btnDownSampling.clicked.connect(
            lambda: [self.downSampling(intNumDownSampling=int(intNumDownSampling.text()))])
        gridDownSampling = QGridLayout()
        gridDownSampling.setSpacing(10)
        # gridDownSampling.addWidget(labDownSampling, 1, 0)
        gridDownSampling.addWidget(intNumDownSampling, 1, 0)
        gridDownSampling.addWidget(btnDownSampling, 1, 1)
        widDownSampling = QGroupBox(self)
        widDownSampling.setLayout(gridDownSampling)
        self.layout.addWidget(widDownSampling, 0)

    def downSampling(self, intNumDownSampling=2):
        self.MyProgram.bool_downSampling = True
        self.MyProgram.downSampling(intNumDownSampling=intNumDownSampling)


class OptionWindowFilter(QDialog):
    def __init__(self, moduleName):
        super().__init__()
        self.MyProgram = MyProgram1
        self.setWindowTitle(f"{moduleName} option")
        self.layout = QVBoxLayout()
        self.label = QLabel(f"{moduleName}: option")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        labFmin   = QLabel('fmin', self)
        labFmin1  = QLabel('fmin1', self)
        labFmax   = QLabel('fmax', self)
        labFmax1  = QLabel('fmax1', self)
        labWColumn = QLabel('WColumn', self)
        labWRow   = QLabel('WRaw', self)
        editFmin  = QLineEdit('0.01', self)
        editFmin1 = QLineEdit('0.1', self)
        editFmax  = QLineEdit('30', self)
        editFmax1 = QLineEdit('40', self)
        editWColumn = QLineEdit('0.5', self)
        editWRow   = QLineEdit('0.5', self)

        btnBandpass = QPushButton('Bandpass', self)
        btnBandpass.setToolTip('This is a <b>QPushButton</b> widget')
        btnBandpass.clicked.connect(
            lambda: [self.filter(editFmin.text(), editFmin1.text(), editFmax.text(), editFmax1.text())])
        
        btnRawData = QPushButton('Raw Data', self)
        btnRawData.setToolTip('This is a <b>QPushButton</b> widget')
        btnRawData.clicked.connect(
            lambda: [self.rawData()])
        
        btnColumnFK = QPushButton('Column FK', self)
        btnColumnFK.setToolTip('This is a <b>QPushButton</b> widget')
        btnColumnFK.clicked.connect(
            lambda: [self.columnFK(float(editWColumn.text()))])
        
        btnRowFK = QPushButton('Row FK', self)
        btnRowFK.setToolTip('This is a <b>QPushButton</b> widget')
        btnRowFK.clicked.connect(
            lambda: [self.rowFK(float(editWRow.text()))])


        grid = QGridLayout()
        widFs = QGroupBox(self)
        widFs.setLayout(grid)
        grid.setSpacing(10)
        grid.addWidget(labFmin, 1, 0)
        grid.addWidget(editFmin, 1, 1)
        grid.addWidget(labFmin1, 2, 0)
        grid.addWidget(editFmin1, 2, 1)
        grid.addWidget(labFmax, 3, 0)
        grid.addWidget(editFmax, 3, 1)
        grid.addWidget(labFmax1, 4, 0)
        grid.addWidget(editFmax1, 4, 1)
        grid.addWidget(labWColumn, 5, 0)
        grid.addWidget(editWColumn, 5, 1)
        grid.addWidget(labWRow, 6, 0)
        grid.addWidget(editWRow, 6, 1)

        LayoutFilter = QVBoxLayout()
        widFilter    = QGroupBox("Filter", self)
        widFilter.setLayout(LayoutFilter)
        LayoutFilter.addWidget(widFs, 0)
        LayoutFilter.addWidget(btnBandpass, 1)
        LayoutFilter.addWidget(btnRawData, 2)
        LayoutFilter.addWidget(btnColumnFK, 3)
        LayoutFilter.addWidget(btnRowFK, 4)

        self.layout.addWidget(widFilter, 0)

    def filter(self, editFmin, editFmin1, editFmax, editFmax1):
        self.fmin = float(editFmin); self.fmin1 = float(editFmin1)
        self.fmax = float(editFmax); self.fmax1 = float(editFmax1)
        # self.MyProgram.bandpassData(
        #     float(editFmin), float(editFmax))
        self.MyProgram.bool_filter = True
        self.MyProgram.RawDataBpFilter(self.fmin, self.fmin1, self.fmax, self.fmax1)

    def rawData(self):
        self.MyProgram.bool_rawData = True
        self.MyProgram.RawData()
        
    def columnFK(self, w):
        self.MyProgram.bool_columnFK = True
        self.MyProgram.RawDataFKFilterColumn(w)
        
    def rowFK(self, w):
        self.MyProgram.bool_rowFK = True
        self.MyProgram.RawDataFKFilterRow(w)

# TAG: SignalTrace
class OptionWindowSignalTrace(QDialog):
    def __init__(self, moduleName):
        super().__init__()
        self.MyProgram = MyProgram1
        self.setWindowTitle(f"{moduleName} option")
        self.layout = QVBoxLayout()
        self.label = QLabel(f"{moduleName}: option")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        labTraceMilter   = QLabel('trace milter', self)
        editTraceMilter  = QLineEdit('10', self)

        grid = QGridLayout()
        widFs = QGroupBox(self)
        widFs.setLayout(grid)
        self.layout.addWidget(widFs, 0)
        grid.setSpacing(10)
        grid.addWidget(labTraceMilter, 1, 0)
        grid.addWidget(editTraceMilter, 1, 1)
        

        btnSignalTrace = QPushButton('SignalTrace', self)
        btnSignalTrace.setToolTip('This is a <b>QPushButton</b> widget')
        btnSignalTrace.clicked.connect(
            lambda: [self.signalTrace(editTraceMilter.text())])
        self.layout.addWidget(btnSignalTrace, 1)


    def signalTrace(self, editTraceMilter):
        t, data, xf, yf = self.MyProgram.signalTrace(int(editTraceMilter))
        self.trace = TraceMainWindow()
        self.trace.imshowTrace(t, data, xf, yf)
        self.trace.show()

