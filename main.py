# __main__.py 程序入口

import sys
from PyQt6.QtWidgets import QApplication
from dasQt.dasQt_plus import MainWindow, apply_stylesheet

def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    
    # 应用样式表
    apply_stylesheet(app, 'dasQt/style.qss')
    sys.exit(app.exec())

if __name__ == '__main__':
    main()