
import os
import sys
import torch
import numpy as np
from pathlib import Path

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Detector:
    def __init__(
            self,
            weight=ROOT / 'yolov5s.pt',  # model path or triton URL
            data=None,  #ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.5,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,
            line_thickness=2,  # bounding box thickness (pixels)
    ):
        self.weight = weight
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.line_thickness = line_thickness
        
        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weight=self.weight, device=self.device)
    
    def detect(self, img):
        stride, names = self.model.stride, self.model.names
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        
        # Run inference
        self.model.warmup(imgsz=(1, 3, *imgsz))  # warmup
        origin_img = img.copy()
        
        # preprocess img
        img = letterbox(img, imgsz, stride=stride, auto=True)[0]  # padded resize
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous  
        
        img = torch.from_numpy(img).to(self.model.device)
        img = img.float()  # uint8 to fp32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = self.model(img)
        
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, max_det=self.max_det)

        # Process predictions 
        for det in pred:  # per image
            annotator = Annotator(origin_img, line_width=self.line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], origin_img.shape).round()
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
            # Stream results
            result_img = annotator.result()

        return [result_img, pred]