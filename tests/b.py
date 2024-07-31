from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QDoubleSpinBox

class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Settings')
        self.setGeometry(300, 300, 300, 150)
        
        layout = QVBoxLayout()

        # Waterfall Amp Scope
        self.label_amp_scope = QLabel('Waterfall Amp Scope')
        self.spinbox_amp_scope = QDoubleSpinBox()
        self.spinbox_amp_scope.setRange(0.0, 1.0)  # 设置值的范围
        self.spinbox_amp_scope.setSingleStep(0.001)  # 设置每次增减的步长
        self.spinbox_amp_scope.setValue(0.500)  # 设置默认值
        layout.addWidget(self.label_amp_scope)
        layout.addWidget(self.spinbox_amp_scope)
        
        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication([])
    win = SettingsWindow()
    win.show()
    app.exec()
