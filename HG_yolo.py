#작성일 221223
import torch
import sys
import copy
import numpy as np
import os
from torch.cuda.amp import autocast
from PIL import ImageFont
#test'''''
# import cv2
# test_img = r'C:\Users\DW\Desktop\porg\dataset\YOLO_DATASET\SD_Classification\images/dish_22.11.22_15.11.25.jpg'
# test_img = cv2.imread(test_img)
#'''''''''
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))
try:
    from utils.torch_utils import select_device
    from models.experimental import attempt_load
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
    from utils.plots import Annotator, colors
except:
    print('폴더 내 yolo 가 없는듯 합니다')

class HG_yolo:
    def __init__(self, ptPath, device = '', class_path = False ,pil=False) -> None:
        
        self.device = select_device(device)
        self.model = attempt_load(ptPath, device=self.device)

        self.stride = int(self.model.stride.max())
        self.augment=False
        self.visualize=False
        
        ##Annotator
        self.pil = pil
        self.set_class_name(class_path)
    
        self.name = self.model.names

        self.imgsz = check_img_size(640, s=self.stride)
        self.fp16 = False

        ### non_max_suppression opt. ###
        self.conf_thres=0.60
        self.iou_thres=0.45
        self.max_det=1000
        self.classes = None
        self.agnostic_nms = False

    def set_class_name(self,path):
        if path:
            txt = open(path, 'r', encoding= 'utf-8').readlines()
            txt_list = ''.join(txt).replace('\n',',').split(',')
            self.class_name = txt_list
        else:
            self.class_name = False
    

    def _img_prep_detect(self, img):
        '''
        이미지 tesnor 0.0 - 1.0 변환
        '''
        im = letterbox(img, self.imgsz, stride=self.stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32

        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        #with autocast(device_type=self.device):
        pred = self.model(im, augment=self.augment, visualize=self.visualize)[0]

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        return im, pred

    def __call__(self, img, mode = 0):
        img = copy.deepcopy(img)
        yolo_img, pred = self._img_prep_detect(img)
        if mode:
            yolo_img = self.draw_boxes(yolo_img,img,pred)
        return yolo_img,pred[0]
        

    def draw_boxes(self, im, img0, pred):
        img0 = np.ascontiguousarray(img0) # 에러 방지
        annotator = Annotator(img0, line_width=2, font_size=3)          #라인 두께
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = "" #self.names[c]
                    annotator.box_label(xyxy, label, color=colors(c, True))

        return img0

