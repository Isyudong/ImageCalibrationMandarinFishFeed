import cv2
import numpy as np

def ImageProcessing(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化处理
    retval, dst = cv2.threshold(gray,thresh=50,maxval=255,type=cv2.THRESH_BINARY_INV) # 二值化阈值(87, 255)
    # 定义结构元素
    kernel = np.ones((5, 5), np.uint8)
    # 膨胀与腐化
    dilated_image = cv2.dilate(dst, kernel, iterations=3)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=3)
    # 查找轮廓
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours