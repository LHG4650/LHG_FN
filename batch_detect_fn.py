import cv2

def fh2_img_const_zero_bler(img,cosnt_alpha=10):
    _, zero = cv2.threshold(img,70,255, cv2.THRESH_TOZERO)

    lab = cv2.cvtColor(zero, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cosnt_alpha,tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    cont_dst = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    cont2blur10 = cv2.fastNlMeansDenoisingColored(cont_dst,None,5,5,7,21)

    return cont2blur10