import sys
# from PyQt5.QtWidgets import *
from PyQt5 import uic

import cv2
import threading
import sys
import numpy as np
import time
import datetime
from playsound import playsound

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from DW_detect import DW_detect, area_work

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
main_window_form_class = uic.loadUiType("gui/GUI_ver5.ui")[0]
setting_dialog_form_class = uic.loadUiType("gui/GUI_setting.ui")[0]

#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QtWidgets.QMainWindow, main_window_form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        ### statusbar 설정 ###
        self.label_time = QtWidgets.QLabel()
        self.statusbar.addPermanentWidget(self.label_time)

        ### 초기 변수 선언 ###
        self.running = False
        self.NG_count = 0
        self.NG_count_cut = 0
        self.NG_count_null = 0
        self.AI_Detector = DW_detect('models/CJ_220303.pt')
        self.area_func = area_work(85, 5, 465, 475, 3, 3) # (x0, y0, x1, y1, x_num, y_num)
        self.cap = cv2.VideoCapture(0)
        # self.cap = cv2.VideoCapture('video/220121_5.avi')
        self.direction = 'LR' # 라인 이동 방향, 시작 > 끝 순서 코드, LR, RL, UD, DU
        self.save_path = 'images_220218/worked/' # 이미지 저장경로, Full path로 입력 요

        ### 기능 할당 / 가동 ###
        self.btn_start.clicked.connect(self.onoffCam)
        # self.actionSet_Detect_area.triggered.connect(self.setting_run)
        self.actionexit.triggered.connect(self.onExit)

        self.set_direction_status(self.direction)
        self.run_clock()

    ### 메인 검사 가동 ###
    def run(self):
        cap = self.cap
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.label_rt.resize(width, height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 모델 레이어 구동
        AI_Detector = self.AI_Detector
        area_func = self.area_func
        pred_array_before = np.zeros((3, 3), dtype=np.uint8) # 0 - 정상, 1 - 절단, 2 - 누락
        t0 = time.time()

        while self.running:
            ret, img = cap.read()
            if ret:
                # AI 모델 적용
                self.img_origin = img.copy()
                img, pred_result = AI_Detector.AI_detect(img) # AI 객체탐지
                pred_list = area_func.change_predlist(pred_result) # 결과 리스트 / xyxy를 중심좌표로 변경
                img, pred_array_now= area_func.pred_area_check(img, pred_list)

                self.area_func.rotate_arr(pred_array_now, self.direction) # 작업 방향 조정

                bl_NG_cut, bl_NG_null = area_func.compare_pred(pred_array_now, pred_array_before)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.img_w = img.copy()

                t1 = time.time()
                time_check = t1 - t0

                if (bl_NG_cut or bl_NG_null) and time_check >= 1: # NG 발생시 컨트롤 부, 1초마다 작동
                    self.show_image2qlabel(img, self.label_ng)
                    # self.captrue_image()
                    ## NG 수량 표시 ##
                    self.plus_NG()
                    if bl_NG_null:
                        self.plus_NG_null()
                    if bl_NG_cut:
                        self.plus_NG_cut()
                    self.NG_alert()
                    t0 = time.time() # 시간 초기화

                pred_array_before = pred_array_now

                self.show_image2qlabel(img, self.label_rt)

            else:
                self.label_tx.setText("cannot read frame.")
                break
        cap.release()
        self.label_tx.setText("Detecting end.")
        self.btn_start.setText('Detect start')


    def captrue_image(self):
        time = datetime.datetime.today().strftime('%m%d%H%M%S')
        img_wk = cv2.cvtColor(self.img_w, cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.save_path + time + '_wk.bmp', img_wk) # 판정 결과 포함
        cv2.imwrite(self.save_path + time + '_org.bmp', self.img_origin) # 원본 이미지
        self.label_tx.setText(time + '_saved')

    ### 시계 ###
    def run_clock(self):
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.show_datetime)
        self.timer.start()

    def show_datetime(self):
        qDateTimeVar = QtCore.QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')
        self.label_time.setText(qDateTimeVar)
        # self.statusBar().showMessage(qDateTimeVar)

    ### NG count 제어 ###
    def plus_NG(self):
        self.NG_count += 1
        self.label_ng_total.setText(str(self.NG_count))

    def plus_NG_null(self):
        self.NG_count_null += 1
        self.label_ng_null.setText(str(self.NG_count_null))

    def plus_NG_cut(self):
        self.NG_count_cut += 1
        self.label_ng_cut.setText(str(self.NG_count_cut))

    def reset_NGcount(self):
        self.NG_count = 0
        self.NG_count_cut = 0
        self.NG_count_null = 0
        self.label_ng_total.setText(str(self.NG_count))
        self.label_ng_null.setText(str(self.NG_count_null))
        self.label_ng_cut.setText(str(self.NG_count_cut))

    ### 작업 방향 표시 ###
    def direction_change(self, direction):
        if direction == 'LR':
            direct_status = '좌 > 우 (▶)'
        elif direction == 'RL':
            direct_status = '우 > 좌 (◀)'
        elif direction == 'UD':
            direct_status = '상 > 하 (▼)'
        elif direction == 'DU':
            direct_status = '하 > 상 (▲)'
        else:
            direct_status = '확인불가'
        return direct_status
    
    def set_direction_status(self, direction):
        direct_status = self.direction_change(direction)
        self.label_direction.setText('라인 진행 방향 : ' + direct_status)   

    ### 이미지 표시 ###
    def show_image2qlabel(self, iamge, qlabel_name):
        img = cv2.resize(iamge, dsize=(qlabel_name.width(), qlabel_name.height()), interpolation=cv2.INTER_AREA)
        h,w,c = img.shape
        qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        qlabel_name.setPixmap(pixmap)

    ### 검사 모드 가동/정지 ###
    def onoffCam(self):
        if self.btn_start.isChecked():
            self.btn_start.setText('Detect stop')
            self.start()
        else:
            self.btn_start.setText('Detect start')
            self.stop()

    ### 경고음 재생 ###
    def beep_sound(self):
        playsound('gui/beep.wav')
    
    def NG_alert(self):
        alert_th = threading.Thread(target=self.beep_sound)
        alert_th.start()
    

    def stop(self):
        # global running
        self.running = False
        self.label_tx.setText("stoped..")

    def start(self):
        # global running
        self.running = True
        th = threading.Thread(target=self.run)
        th.start()
        self.label_tx.setText('Detecting..')


    def setting_run(self):
        dlg = setting_dialog(self)
        dlg.exec_()

    
    def onExit(self):
        self.label_tx.setText("exit")
        self.stop()
        sys.exit()


class setting_dialog(QtWidgets.QDialog, setting_dialog_form_class) :
    def __init__(self, parents_self) :
        super().__init__()
        self.setupUi(self)

        self.btn_imgset_1.clicked.connect(self.set_xy1)
        self.btn_imgset_2.clicked.connect(self.set_xy2)

        # ret, img = parents_self.cap.read()

        img = parents_self.setting_img

        # print(ret)
        # if ret == False:
        #     cap = cv2.VideoCapture(0)
        #    ret, img = cap.read()
        
        parents_self.show_image2qlabel(img, self.label_img)

        self.setMouseTracking(True)

    def mousePressEvent(self, event): # e ; QMouseEvent
        self.X = event.x()-20
        self.Y = event.y()-30
        self.label_status_1.setText('X : ' + str(self.X))
        self.label_status_2.setText('Y : ' + str(self.Y))

    def set_xy1(self):
        self.x1 = self.X
        self.y1 = self.Y
        self.label_x1.setText(str(self.x1))
        self.label_y1.setText(str(self.y1))

    def set_xy2(self):
        self.x2 = self.X
        self.y2 = self.Y
        self.label_x2.setText(str(self.x2))
        self.label_y2.setText(str(self.y2))






    

if __name__ == "__main__" :
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QtWidgets.QApplication(sys.argv) 

    #WindowClass의 인스턴스 생성
    myWindow = WindowClass() 

    #프로그램 화면을 보여주는 코드
    # myWindow.show()
    myWindow.showMaximized()

    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()