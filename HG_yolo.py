# -*- coding: utf-8 -*-

#작성일 221223
import torch
import numpy as np

try:
    from models.common import DetectMultiBackend
    from utils.plots import Annotator, colors
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
    from utils.augmentations import letterbox
except Exception as e:
    print('Directory empty or')
    print(e)
    import sys
    for i in sys.path:
        print(i)
        
class HgYolo:
    def __init__(self, 
                ptPath,         #pt 파일 경로
                device = '',    #가용 디바이스
                
                img_size = (640, 640),  #돌리고자 하는 높이, 넓이
                
                conf_thres=0.5,  # confidence threshold
                iou_thres=0.45,  # NMS IOU threshold
                max_det=1000,  # maximum detections per image


                draw_box_opt = True, # 박스 칠껀지 안칠껀지.
                draw_label_opt = True
                ) :
        
        # condition
        self.warmup = True
        self.draw_box_opt = draw_box_opt
        self.draw_label_opt = draw_label_opt

        # Model Load
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights = ptPath, device = self.device)

        #img check
        self.img_size = img_size
        self.line_thickness = 2
        
        #NMS config
        self.conf_thres=conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        


        
    def _warmup(self):
        stride = self.model.stride
        img_size = check_img_size(self.img_size, s=stride)  # check image size
        self.model.warmup(imgsz=(1, 3, *img_size))
        self.warmup = False

    def detect(self,img):

        if self.warmup:
            self._warmup()

        if self.draw_box:
            origin_img = img.copy()

        #img_preprocess
        img = letterbox(img, self.img_size, self.model.stride, auto=True)[0] # padded resize
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous 

        img = torch.from_numpy(img).to(self.model.device)
        img = img.float()  # uint8 to fp32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        #model
        pred = self.model(img)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)

        # draw box
        if self.draw_box_opt:
            for det in pred:  # per image
                annotator = Annotator(origin_img, line_width=self.line_thickness, example=str(self.model.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], origin_img.shape).round()
                    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        if self.draw_label_opt:
                            label = f'{self.model.names[c]} {conf:.2f}'
                        else: 
                            label = ''
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        
                # Stream results
                result_img = annotator.result()
        else:
            result_img = False

        return [result_img, pred]

    def __call__(self, img):
        return self.detect(img)
        

    def draw_boxes(self, img, pred, label_opt = False):
        origin_img = img.copy()

        img = letterbox(img, self.img_size, self.model.stride, auto=True)[0] # padded resize
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous 

        img = torch.from_numpy(img).to(self.model.device)
        img = img.float()  # uint8 to fp32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        for det in pred:  # per image
                annotator = Annotator(origin_img, line_width=self.line_thickness, example=str(self.model.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], origin_img.shape).round()
                    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        if label_opt:
                            label = f'{self.model.names[c]} {conf:.2f}'
                        else: 
                            label = ''
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        
                # Stream results
                result_img = annotator.result()

        return result_img

