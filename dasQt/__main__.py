# __main__.py 程序入口

import sys
from PyQt6.QtWidgets import QApplication
from dasQt.dasQt import MainWindow

def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()