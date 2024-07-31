

import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QToolTip, QMessageBox,
                             QMainWindow, QHBoxLayout, QVBoxLayout, QFileDialog, QSizePolicy,
                             QSlider, QLabel, QLineEdit, QGridLayout, QGroupBox, QListWidget,
                             QTabWidget, QDialog, QCheckBox, QComboBox, QTableWidget, QTableWidgetItem,
                             QDoubleSpinBox)
from PyQt6.QtGui import QIcon, QFont, QAction, QGuiApplication
from PyQt6.QtCore import Qt, QTimer, QFile, QTextStream, QSize
from PyQt6.QtWidgets import QApplication, QMainWindow, QDockWidget, QTextEdit

import os
import csv
import numpy as np
from disba import PhaseDispersion, depthplot
from utils import gen_model1
from evodcinv import EarthModel, Layer, Curve

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 16

import dasQt.das as das
from dasQt.utools.logPy3 import HandleLog


class Inversion1DMainWindow(QMainWindow):
    def __init__(self, MyProgram, sliderFig, 
                 f_readNextData,
                 title="Inversion1D"):
        super().__init__()

        self.is_closed           : bool = False
        self.bool_log           : bool = False
        
        self.logger = HandleLog(os.path.split(__file__)[-1].split(".")[0], path=os.getcwd(), level="DEBUG")
        self.MyProgram = MyProgram
        self.sliderFig = sliderFig
        self.f_readNextData = f_readNextData
        # self.f_readNext20DataDispersion = f_readNext20DataDispersion
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

        self.initControl()
        
        mainFigure = QMainWindow()
        self.layout.addWidget(mainFigure, 1)


        widDispersionFigure = QWidget()
        dockDispersionFigure = QDockWidget("Dispersion", self)
        dockDispersionFigure.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dockDispersionFigure.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dockDispersionFigure.setWidget(widDispersionFigure)
        mainFigure.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dockDispersionFigure)
        self.initDispersionFigure(widDispersionFigure)

        widModelFigure = QWidget()
        dockModelFigure = QDockWidget("Model", self)
        dockModelFigure.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dockModelFigure.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dockModelFigure.setWidget(widModelFigure)
        mainFigure.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dockModelFigure)
        self.initModelFigure(widModelFigure)

        self.show()



    def initModelFigure(self, widFigure: QWidget) -> None:
        self.figModel = Figure()
        self.canvasModel = FigureCanvas(self.figModel)
        toolbar = NavigationToolbar(self.canvasModel, self)

        newLayout = QVBoxLayout()
        widFigure.setLayout(newLayout)
        newLayout.addWidget(self.canvasModel, 0)
        newLayout.addWidget(toolbar, 0)
        # self.layout.addWidget(widFigure, 1)


    def initDispersionFigure(self, widFigure: QWidget) -> None:
        self.figDispersion = Figure()
        self.canvasDispersion = FigureCanvas(self.figDispersion)
        toolbar = NavigationToolbar(self.canvasDispersion, self)

        newLayout = QVBoxLayout()
        widFigure.setLayout(newLayout)
        newLayout.addWidget(self.canvasDispersion, 0)
        newLayout.addWidget(toolbar, 0)
        # self.layout.addWidget(widFigure, 1)















    def initControl(self) -> None:
        widAll = QTabWidget()
        self.layout.addWidget(widAll, 0)
        layoutAll = QVBoxLayout()
        widAll.setLayout(layoutAll)


        widForward = QGroupBox("Forward", self)
        layForward = QVBoxLayout()
        widForward.setLayout(layForward)
        widAll.addTab(widForward, "Forward")

        # 创建表格
        self.tableForward = QTableWidget(3, 4, self)  # 默认3行3列
        self.tableForward.setHorizontalHeaderLabels(['Thick', 'Vs', 'Vp', 'Rho'])
        layForward.addWidget(self.tableForward)
        
        gridForward = QGridLayout()
        gridForward.setSpacing(10)
        layForward.addLayout(gridForward)
        
        btnAddRow = QPushButton('Add Row', self)
        btnAddRow.clicked.connect(self.addRowForward)
        gridForward.addWidget(btnAddRow, 0, 1)
        
        btnForward = QPushButton('Forward', self)
        btnForward.clicked.connect(self.forward)
        gridForward.addWidget(btnForward, 0, 0)

        btnSave = QPushButton('Save Par', self)
        btnSave.clicked.connect(self.saveForwardPar)
        gridForward.addWidget(btnSave, 1, 1)
        
        btnLoad = QPushButton('Load Par', self)
        btnLoad.clicked.connect(self.loadForwardPar)
        gridForward.addWidget(btnLoad, 1, 0)

        self.checkBoxLog = QCheckBox('Log Plot', self)
        self.checkBoxLog.setChecked(False)
        self.checkBoxLog.stateChanged.connect(self.isLog)
        gridForward.addWidget(self.checkBoxLog, 2, 0)

        btnSaveDispersionData = QPushButton('Save Dispersion', self)
        btnSaveDispersionData.clicked.connect(self.saveDispersionData)
        gridForward.addWidget(btnSaveDispersionData, 2, 1)


        widInversion = QGroupBox("Inversion", self)
        layInversion = QVBoxLayout()
        widInversion.setLayout(layInversion)
        widAll.addTab(widInversion, "Inversion")

        # 创建表格
        self.tableInversion = QTableWidget(3, 4, self)  # 默认3行3列
        self.tableInversion.setHorizontalHeaderLabels(['Thick Min', 'Thick Max', 'Vs Min', 'Vs Max'])
        layInversion.addWidget(self.tableInversion)

        gridInversion = QGridLayout()
        gridInversion.setSpacing(10)
        layInversion.addLayout(gridInversion)

        btnAddRow = QPushButton('Add Row', self)
        btnAddRow.clicked.connect(self.addRowInversion)
        gridInversion.addWidget(btnAddRow, 0, 1)
        
        btnInversion = QPushButton('Inversion', self)
        btnInversion.clicked.connect(self.inversion)
        gridInversion.addWidget(btnInversion, 0, 0)

        btnSave = QPushButton('Save Par', self)
        btnSave.clicked.connect(self.saveInversionPar)
        gridInversion.addWidget(btnSave, 1, 1)
        
        btnLoad = QPushButton('Load Par', self)
        btnLoad.clicked.connect(self.loadInversionPar)
        gridInversion.addWidget(btnLoad, 1, 0)

        btnLoadDispersionData = QPushButton('Load Dispersion', self)
        btnLoadDispersionData.clicked.connect(self.loadDispersionData)
        gridInversion.addWidget(btnLoadDispersionData, 2, 0)

        labMaxIter = QLabel('Max Iteration', self)
        self.spinboxMaxIter = QDoubleSpinBox()
        self.spinboxMaxIter.setRange(100, 3000)  # 设置值的范围
        self.spinboxMaxIter.setSingleStep(50)  # 设置每次增减的步长
        self.spinboxMaxIter.setValue(500)  # 设置默认值
        gridInversion.addWidget(labMaxIter, 3, 0)
        gridInversion.addWidget(self.spinboxMaxIter, 3, 1)





    def saveForwardPar(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存参数', '', 'CSV文件 (*.csv)')
        if path:
            try:
                with open(path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    rows = self.tableForward.rowCount()
                    cols = self.tableForward.columnCount()
                    header = [self.tableForward.horizontalHeaderItem(col).text() for col in range(cols)]
                    writer.writerow(header)
                    for row in range(rows):
                        row_data = []
                        for col in range(cols):
                            item = self.tableForward.item(row, col)
                            row_data.append(item.text() if item is not None else '')
                        writer.writerow(row_data)
                QMessageBox.information(self, '成功', '参数已保存')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'保存参数时出错：{e}')

    def loadForwardPar(self):
        path, _ = QFileDialog.getOpenFileName(self, '加载参数', '', 'CSV文件 (*.csv)')
        if path:
            try:
                with open(path, 'r') as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    self.tableForward.setRowCount(0)
                    self.tableForward.setColumnCount(len(header))
                    self.tableForward.setHorizontalHeaderLabels(header)
                    for row_data in reader:
                        row = self.tableForward.rowCount()
                        self.tableForward.insertRow(row)
                        for col, data in enumerate(row_data):
                            self.tableForward.setItem(row, col, QTableWidgetItem(data))
                QMessageBox.information(self, '成功', '参数已加载')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'加载参数时出错：{e}')


    def saveInversionPar(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存参数', '', 'CSV文件 (*.csv)')
        if path:
            try:
                with open(path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    rows = self.tableInversion.rowCount()
                    cols = self.tableInversion.columnCount()
                    header = [self.tableInversion.horizontalHeaderItem(col).text() for col in range(cols)]
                    writer.writerow(header)
                    for row in range(rows):
                        row_data = []
                        for col in range(cols):
                            item = self.tableInversion.item(row, col)
                            row_data.append(item.text() if item is not None else '')
                        writer.writerow(row_data)
                QMessageBox.information(self, '成功', '参数已保存')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'保存参数时出错：{e}')

    def loadInversionPar(self):
        path, _ = QFileDialog.getOpenFileName(self, '加载参数', '', 'CSV文件 (*.csv)')
        if path:
            try:
                with open(path, 'r') as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    self.tableInversion.setRowCount(0)
                    self.tableInversion.setColumnCount(len(header))
                    self.tableInversion.setHorizontalHeaderLabels(header)
                    for row_data in reader:
                        row = self.tableInversion.rowCount()
                        self.tableInversion.insertRow(row)
                        for col, data in enumerate(row_data):
                            self.tableInversion.setItem(row, col, QTableWidgetItem(data))
                QMessageBox.information(self, '成功', '参数已加载')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'加载参数时出错：{e}')




    def addRowForward(self):
        current_row_count = self.tableForward.rowCount()
        self.tableForward.insertRow(current_row_count)

    def addRowInversion(self):
        current_row_count = self.tableInversion.rowCount()
        self.tableInversion.insertRow(current_row_count)

    def saveDispersionData(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Dispersion Curve', '', '.npz (*.npz)')
        if path:
            try:
                np.savez(path, f=self.f, vel=self.vel)
                QMessageBox.information(self, '成功', '参数已保存')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'保存参数时出错：{e}')

    def loadDispersionData(self, data):
        path, _ = QFileDialog.getOpenFileName(self, 'Load Dispersion Curve', '', '.npz (*.npz)')
        if path:
            try:
                data = np.load(path)
                f = data['f']
                vel = data['vel']
                self.f = f
                self.velocity = vel
                self.imshowDispersion(f, vel)
                QMessageBox.information(self, '成功', '参数已加载')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'加载参数时出错：{e}')



    def isLog(self, state):
        if state == 0:
            self.bool_log = False
        else:
            self.bool_log = True



    def forward(self):
        try:
            rows = self.tableForward.rowCount()
            cols = self.tableForward.columnCount()
            data_dict = {}
            for col in range(cols):
                data_dict[col] = []
                for row in range(rows):
                    item = self.tableForward.item(row, col)
                    if item is not None:
                        data_dict[col].append(float(item.text()))
                    else:
                        data_dict[col].append(0)  # 如果单元格为空，填0
                        

            # print(data_dict)
            # if sum(data_dict[2]) == 0:
            #     thick_true,vp_true,vs_true,rho_true = gen_model1(data_dict[0], data_dict[1],area=True)
            #     data_dict[2] = vp_true.tolist()
            #     data_dict[3] = rho_true.tolist()
            # elif sum(data_dict[3]) == 0:
            #     thick_true,vp_true,vs_true,rho_true = gen_model1(data_dict[0], data_dict[1],data_dict[2],data_dict[3],area=True)
            #     data_dict[3] = rho_true.tolist()
            thick_true,vp_true,vs_true,rho_true = gen_model1(data_dict[0], data_dict[1],area=True)
            data_dict[2] = vp_true.tolist()
            data_dict[3] = rho_true.tolist()

            for col in range(cols):
                for row in range(rows):
                    item = QTableWidgetItem(str(data_dict[col][row]))
                    self.tableForward.setItem(row, col, item)

            velocity_model = np.array([
                data_dict[0],   # thickness
                data_dict[2],   # vp
                data_dict[1],   # vs
                data_dict[3]    # rho
            ]).T
            self.velocity_model = velocity_model
            print(velocity_model)
            
            
            start, end = 0.2, 15
            t = np.logspace(np.log10(1/end), np.log10(1/start), 100)

            pd = PhaseDispersion(*velocity_model.T)
            cpr = pd(t, mode=0, wave="rayleigh")
            self.cpr = cpr
            self.f = 1./cpr.period
            self.vel = cpr.velocity
            
            self.imshowDispersion(1./cpr.period, cpr.velocity)
            
        except ValueError:
            print('ValueError')
            QMessageBox.critical(self, '错误', '输入格式有误，请输入有效的数字。')

    def inversion(self):
        try:
            rows = self.tableInversion.rowCount()
            cols = self.tableInversion.columnCount()
            data_dict = {}
            for row in range(rows):
                data_dict[row] = []
                for col in range(cols):
                    item = self.tableInversion.item(row, col)
                    if item is not None:
                        data_dict[row].append(float(item.text()))
                    else:
                        data_dict[row].append(0)  # 如果单元格为空，填0

        except ValueError:
            print('ValueError')
            QMessageBox.critical(self, '错误', '输入格式有误，请输入有效的数字。')
            return

        print(data_dict)

        # Initialize model
        model = EarthModel()

        for i in range(rows):
            #                   d [km]     vs [km/s]
            #                min    max    min  max
            model.add(Layer([data_dict[i][0], data_dict[i][1]],
                            [data_dict[i][2], data_dict[i][3]]))

        # #                   d [km]     vs [km/s]
        # #                min    max    min  max
        # model.add(Layer([0.02, 0.1], [0.4, 0.7]))  # Layer 1
        # model.add(Layer([0.02, 0.1], [0.4, 1.3]))  # Layer 1
        # model.add(Layer([0.02, 0.1], [0.4, 1.3]))  # Layer 1
        # model.add(Layer([0.10, 0.3], [0.4, 1.3]))  # Layer 1

        # Configure model
        maxiter = int(self.spinboxMaxIter.value())
        model.configure(
            optimizer      = "cmaes",       # Evolutionary algorithm,'cmaes','cpso','de','na','pso','vdcma'
            misfit          = "rmse",        # Misfit function type, 'rmse','norm1','norm2'
            optimizer_args = {
                        "popsize": 10,      # Population size
                        "maxiter": maxiter,    # Number of iterations
                        "workers": -1,      # Number of cores
                        "seed"   : 10,
                    },
        )

        period = 1./self.f
        data = self.velocity
        curves = [Curve(period, data, 0, "rayleigh", "phase", weight=1.0, uncertainties=None)]

        # Run inversion
        res = model.invert(curves)
        print(res)
        
        pd = PhaseDispersion(*res.model.T)
        cpr = pd(period, mode=0, wave="rayleigh")
        
        f = [self.f, 1./cpr.period]
        data = [self.velocity, cpr.velocity]
        
        self.imshowDispersion(f, data, label=['True', 'Inverted'])
        self.imshowModel(res)

    def getXinZin(self, thickness, parameter, ax=None):
        z = np.cumsum(thickness)    # Cumulative sum of thickness
        n = z.size 

        tmp = np.array(thickness)
        tmp[-1] = tmp[:-1].min()
        zmax = tmp.sum()

        # Build layered model
        xin = np.empty(2 * n)
        xin[1::2] = parameter
        xin[2::2] = parameter[1:]
        xin[0] = xin[1]

        zin = np.zeros_like(xin)
        zin[1:-1:2] = z[:-1]
        zin[2::2] = z[:-1]
        zin[-1] = max(z[-1], zmax)
        
        return xin, zin, zmax


    def imshowDispersion(self, f, data, label='True'):
        self.figDispersion.clear(); ax1 = self.figDispersion.add_subplot(111)
        ax1.cla()
        
        num = len(data)

        if np.array(f).ndim == 1:
            ax1.plot(f, data, 'k', label=label)
        else:
            for i in range(num):
                if i == 0:
                    ax1.plot(f[i], data[i], 'k', label=label[i])
                elif i == num-1:
                    ax1.plot(f[i], data[i], 'ro', label=label[i])
                    
                ax1.plot(f[i], data[i], 'gray', label=label[i])
                
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.legend()
        if self.bool_log:
            ax1.set_xscale('log')

        self.canvasDispersion.draw()

    def imshowModel(self, res, label='True'):
        self.figModel.clear(); ax = self.figModel.add_subplot(111)
        ax.cla()

        misfit_min = res.misfit
        misfit_max = res.misfit*500.1


        for i, (model, misfit) in enumerate(zip(res.models, res.misfits)):
            if i % 20 == 0 and i > 400:
                xin, zin, zmax = self.getXinZin(model[:, 0], model[:, 2])
                # ax.plot(xin, zin, color=plt.cm.viridis_r((misfit-misfit_min)/(misfit_max-misfit_min)), alpha=0.9)
                ax.plot(xin, zin, color=plt.cm.viridis_r((misfit - misfit_min) / (misfit_max - misfit_min)), alpha=0.9)

        mappable = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(vmin=misfit_min, vmax=misfit_max))
        cbar = self.figModel.colorbar(mappable, ax=ax, label='Misfit Value')

        # 设置色阶条的标签为科学计数法
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))
        cbar.formatter = formatter
        cbar.update_ticks()


        # thick_true, vs_true = self.velocity_model[:, 0], self.velocity_model[:, 2]
        # xin, zin, _ = self.getXinZin(thick_true, vs_true)
        # ax.plot(xin, zin, 'k', label='True', linewidth=3)


        i = 2           #  1: Vp, 2: Vs, 3: density
        thickness = res.model[:, 0]
        parameter = res.model[:, i]
        xin, zin, zmax = self.getXinZin(thickness, parameter)
        ax.plot(xin, zin, 'r', linestyle='--', linewidth=1.5, label='Inverted')


        # zmax = zmax if zmax < thick_true.sum() else thick_true.sum()
        ax.set_ylim(zin.min(),zmax)
        ax.xaxis.set_ticks_position('top')  #将X坐标轴移到上面
        ax.invert_yaxis()                   #反转Y坐标轴
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')  # 将x轴的标签也移动到顶部
        ax.set_xlabel('Vs (km/s)')
        ax.set_ylabel('Depth (km)')
        # ax.set_title('Inverted Model')
        ax.legend()




        self.canvasModel.draw()




    def closeEvent(self, event):
        """重写关闭事件"""
        self.is_closed = True
        print("窗口已关闭")
        event.accept()  # 接受关闭事件，完成窗口关闭
        
    def isVisible(self):
        return self.is_closed







if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Inversion1DMainWindow(das.DAS())
    window.show()
    sys.exit(app.exec())