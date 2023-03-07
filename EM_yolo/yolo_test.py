
import cv2

from utils.detect import Detector


def img_load():
    pass

# ------------
# 데이터 로드
# ------------
# source_path = 'image_12.jpg'
source_path = 'test_data.jpg'
# img = img_load(source_path)
img = cv2.imread(source_path)
img = cv2.resize(img, (640, 640))

# ------------------------
# yolo 모델 로드 및 detect
# ------------------------
# pt_path = 'C:/Users/user/Desktop/vision_model_learning_v1/yolov5/runs/train/seaweed_ep5002/weights/best.pt'
pt_path = 'best.pt'
detector = Detector(weight=pt_path)
result_img, pred = detector.detect(img)

# ---------------------
# show result
# ---------------------
cv2.imshow('img', img)
cv2.imshow('result', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()