# 최종 수정일 : 20211216
# 수정 내용 : cropimg - 대상 없는 이미지에 대한 예외처리 추가

import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import random

# from sqlalchemy import false

# DW--------------------------------------
# YOLOv5 기반
from DW_utils.models.common import DetectMultiBackend
from DW_utils.utils.torch_utils import select_device
from DW_utils.models.experimental import attempt_load
from DW_utils.utils.augmentations import letterbox
from DW_utils.utils.general import check_img_size, non_max_suppression, scale_coords
from DW_utils.utils.plots import Annotator, colors

import torch
import math

#----------------------------------------

class DW_detect:
    def __init__(self, strWeightPath):
        self.device = select_device()

        self.model = attempt_load(strWeightPath, map_location=self.device)

        # self.model = DetectMultiBackend(strWeightPath, device=self.device, dnn=False)

        self.stride = int(self.model.stride.max())
        # self.stride = self.model.stride
        self.augment=False
        self.visualize=False
        # self.imgsz = check_img_size([640, 640], s=self.stride)
        self.imgsz = check_img_size(640, s=self.stride)

        self.names = self.model.names


        ### non_max_suppression opt. ###
        self.conf_thres=0.60
        self.iou_thres=0.45
        self.max_det=1000
        self.classes = None
        self.agnostic_nms = False

    ### 이미지 전처리 단계 ###
    def img_prep(self, img):
        img = letterbox(img, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(self.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0, 정규화
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        return im

    ### 모델 가동 및 NMS 처리 ###
    def detect(self, img):

        pred = self.model(img, augment=self.augment, visualize=self.visualize)[0]

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        return pred # 예측 결과 BBox 위치 및 클래스 정도가 포함된 list
    
    ### Bbox를 원본 이미지에 draw ###
    def draw_boxes(self, im, img0, pred):
        img0 = np.ascontiguousarray(img0) # 에러 방지
        annotator = Annotator(img0, line_width=2, font_size=3)          #라인 두께
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = "" #self.names[c]
                    annotator.box_label(xyxy, label, color=colors(c, True))

        return img0

    ### 메인 프로세스(전체 통합) ###
    def AI_detect(self, img):
        prep_img = self.img_prep(img)
        pred_result = self.detect(prep_img)
        result_img = self.draw_boxes(prep_img, img, pred_result)

        pred_check = pred_result[0]

        return result_img, pred_check

    

    ###################################################################

class area_work:
    def __init__(self, x0, y0, x1, y1, x_num, y_num):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x_num = x_num
        self.y_num = y_num
        self.x_gap = int((self.x1 - self.x0) / self.x_num)
        self.y_gap = int((self.y1 - self.y0) / self.y_num)
        self.all_count = int(self.x_num*self. y_num)
        self.area_list = self.area_df()

    def  make_area(self, num):
        # x_base, y_base = divmod(num, self.x_num)
        y_base, x_base = divmod(num, self.x_num)

        x0_point = self.x0 + (self.x_gap * x_base)
        y0_point = self.y0 + (self.y_gap * y_base)
        x1_point = self.x0 + (self.x_gap * (x_base + 1))
        y1_point = self.y0 + (self.y_gap * (y_base + 1))

        return (x0_point, y0_point, x1_point, y1_point)

    def area_df(self):
        area_list = list(map(self.make_area, range(self.all_count)))

        # area_df = pd.DataFrame(area_list, columns = ['x0' , 'y0', 'x1', 'y1', 'check'])

        return area_list

    def change_predlist(self, pred_list):
        pred_list0 = pred_list.tolist()

        pred_list1 = []

        for i in pred_list0:
            x_center = int((i[0] + i[2]) / 2)
            y_center = int((i[1] + i[3]) / 2)
            class_index = i[5]
            # cv2.circle(img2, (x_center, y_center), 2, (0, 255, 255), thickness = -1)
            pred_list1.append((x_center, y_center, class_index))

            # self.pred_list = pred_list1

        return pred_list1

    def pred_area_check(self, img, pred_list):
        img_empty = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
        pred_array = np.zeros((self.y_num, self.x_num))  #, dtype=np.uint8

        for index, i in enumerate(self.area_list):
            Ok = False
            NG_cut = False
            NG_null = False
            area_padding = 5

            for j in pred_list:
                center_rac = ((j[0] - area_padding), (j[1] - area_padding), (j[0] + area_padding), (j[1] + area_padding)) # (x1, y1, x2, y2)
                if self.overlap_rac(i, center_rac) and (j[2] == 0 or j[2] == 1 or j[2] == 2):
                    Ok = True
                    break
                elif self.overlap_rac(i, center_rac) and (j[2] == 3 or j[2] == 4 or j[2] == 5):
                    NG_cut = True
                    break
                elif not self.overlap_rac(i, center_rac):
                    NG_null = True

            if Ok:
                img_empty = cv2.rectangle(img_empty, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), -1) # 초록색 BGR
            elif NG_cut:
                img_empty = cv2.rectangle(img_empty, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), -1) # 적색
                pred_array.itemset(index, 1) # NG위치를 1으로 변경        
            elif NG_null:
                img_empty = cv2.rectangle(img_empty, (i[0], i[1]), (i[2], i[3]), (0, 105, 255), -1) # 노란색
                pred_array.itemset(index, 2) # NG위치를 0으로 변경                

        img0 = img.copy()

        img0_temp = img0[self.y0:self.y1, self.x0:self.x1]
        img_empty_temp = img_empty[self.y0:self.y1, self.x0:self.x1]

        dst = cv2.addWeighted(img0_temp, 0.80, img_empty_temp, 0.20, 0)

        img0[self.y0:self.y1, self.x0:self.x1] = dst

        pred_array.transpose()

        return img0, pred_array

    def compare_pred(self, arr_now, arr_before):
        bl_NG_cut = False
        bl_NG_null = False
        for y in range(self.y_num):
            for x in range(self.x_num - 1):
                # if int(arr_before[y][x]) == 0 and int(arr_now[y][x + 1]) == 0:
                if arr_before[y][x] == 1 and arr_now[y][x + 1] == 1:
                    bl_NG_cut = True
                elif arr_before[y][x] == 2 and arr_now[y][x + 1] == 2:
                    bl_NG_null = True
        return bl_NG_cut, bl_NG_null

    def rotate_arr(self, arr, direction):
        if direction == 'LR':
            arr_worked = arr.copy()
        elif direction == 'RL':
            arr_worked = np.fliplr(arr).copy()
        elif direction == 'UD':
            n = len(arr)
            arr_worked = [[0] * n for _ in range(n)]

            for r in range(n):
                for c in range(n):
                    arr_worked[n-1-c][r] = arr[r][c]
            arr_worked = np.array(arr_worked)
        elif direction == 'DU':
            n = len(arr)
            arr_worked = [[0] * n for _ in range(n)]

            for r in range(n):
                for c in range(n):
                    arr_worked[n-1-c][r] = arr[r][c]
            arr_worked = np.array(arr_worked)
            arr_worked = np.fliplr(arr_worked).copy()
        else:
            arr_worked = arr
        return arr_worked

    def overlap_rac(self, rect1, rect2):
        return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[1] > rect2[3] or rect1[3] < rect2[1])



def result_check(pred):
    pred_result =''
    result_box = 0
    for i_, det_ in enumerate(pred):
        result_box += len(det_)

    if result_box == 0:
        pred_result = 'OK'
    else:
        pred_result = 'NG'
    return pred_result

def set_detect_area(x, x0, y, y0, x_num, y_num):
    area_list=[]
    x_gap = int((x0 - x) / x_num)
    y_gap = int((y0 - y) / y_num)

    for j in range(y_num):
        y_point = y + y_gap * (j)
        y0_point = y + y_gap * (j + 1)

        for i in range(x_num):
            x_point = x + x_gap * (i)  
            x0_point = x + x_gap * (i + 1)        

            point = (x_point, x0_point, y_point, y0_point)

            area_list.append(point)

    return area_list