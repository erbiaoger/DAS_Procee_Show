from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QLineEdit, QLabel

class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口的基本属性
        self.setWindowTitle('Settings')
        self.setGeometry(300, 300, 300, 150)
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 添加Waterfall Display Mode设置
        self.label_display_mode = QLabel('Waterfall Display Mode')
        self.combo_display_mode = QComboBox()
        self.combo_display_mode.addItems(['Rainbow', 'Other modes...'])
        layout.addWidget(self.label_display_mode)
        layout.addWidget(self.combo_display_mode)
        
        # 添加Waterfall Amp Scope设置
        self.label_amp_scope = QLabel('Waterfall Amp Scope')
        self.line_edit_amp_scope = QLineEdit()
        self.line_edit_amp_scope.setText('0.500')
        layout.addWidget(self.label_amp_scope)
        layout.addWidget(self.line_edit_amp_scope)
        
        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication([])
    win = SettingsWindow()
    win.show()
    app.exec()
