import cv2
import sys
import pandas as pd
from PyQt5 import uic
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import *

SD_UI_PATH = r'LHG_FN\ui\monitor_N_btn.ui'
main_window_form_class = uic.loadUiType(SD_UI_PATH)[0]

class QHG_Thread(QThread):
    '''
    사용법
        self.a = MyThread(self.connect_tcp)
        self.a.start()

        self.a = MyThread(self.connect_tcp, args=(abc,edf))
        self.a.start()

    '''
    def __init__(self, func, args=()):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)

class QHG_Default(QtWidgets.QMainWindow, main_window_form_class):
    
    def __init__(self):
        # set UI
        super(QHG_Default, self).__init__()
        self.setupUi(self)

        # config
        self.img_mul = 0.7
        self.video_width = 640
        self.video_height = 480

        # btn
        self.LEFT.clicked.connect(self.left_fn)
        self.MID.clicked.connect(self.mid_fn)
        self.RIGHT.clicked.connect(self.right_fn)

        # indicator
        self.Monitors = [self.Monitor_1,self.Monitor_2,self.Monitor_3]
        self.Labels = [self.label_1,self.label_2,self.label_3]

    def mid_fn(self):
        print('mid_fn 안짯는데?')
        pass

    def left_fn(self):
        print('left_fn 안짯는데?')
        pass

    def right_fn(self):
        print('right_fn 안짯는데?')
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.left_fn()
        elif event.key() == Qt.Key_S:
            self.mid_fn()
        elif event.key() == Qt.Key_D:
            self.right_fn()
    
    def set_label(self, text_bar_num : int, txt: str):
        ''' 텍스트를 모니터 밑의 Q_label에 넣어줌 
        좌측 라벨 : 0
        중앙 라벨 : 1
        우측 라벨 : 2'''
        text_bar = self.Labels[text_bar_num]
        text_bar.setText(str(txt))

    def set_img(self, monitor_num : int, cv_img):
        """이미지 처리는 CV2 상태에서 처리된다. QT에 송출을 위해서는 Qpixmap으로 변경시켜야함
        해당 함수에서 cv를 qtimg로 바꿔준채로 송출함
        추가로 해당 모니터의 크기에 맞춰서 이미지 리사이즈 진행함
        좌측 모니터 : 0
        중앙 모니터 : 1
        우측 모니터 : 2
        """
        monitor = self.Monitors[monitor_num]
        # self.img_mul_set()     #한번만 합시다.
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = QImage(img.data, img.shape[1],
                     img.shape[0], QImage.Format_RGB888)
        img = img.scaled(
            int(round(self.img_mul * self.video_width, 0)),
            int(round(self.img_mul * self.video_height, 0)),
            Qt.KeepAspectRatio,)  # 이미지 배율만큼 변환
        img = QPixmap.fromImage(img)
        monitor.setPixmap(img)

def QHG_Default_active(QHG_Default :QHG_Default):
    App = QApplication(sys.argv)
    Root = QHG_Default
    Root.showNormal()
    sys.exit(App.exec())

class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()

        if orientation == Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError,):
                return QVariant()
        elif orientation == Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError,):
                return QVariant()

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()

        if not index.isValid():
            return QVariant()

        return QVariant(str(self._df.iloc[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.iloc[index.row()]
        row[index.column()] = value
        return True

    def rowCount(self, parent=QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(
            colname, ascending=order == Qt.AscendingOrder, inplace=True
        )
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()
