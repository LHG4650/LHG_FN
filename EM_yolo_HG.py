from LHG_FN.EM_yolo.utils.detect import Detector

class EM_YOLO(Detector):
    def __init__(self, weight=..., data=None, imgsz=..., conf_thres=0.5, iou_thres=0.45, max_det=1000, device='', classes=None, line_thickness=2):
        super().__init__(weight, data, imgsz, conf_thres, iou_thres, max_det, device, classes, line_thickness)