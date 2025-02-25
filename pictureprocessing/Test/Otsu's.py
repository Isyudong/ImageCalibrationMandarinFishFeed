import cv2
import numpy as np

# 读取图像
image = cv2.imread('C:\\Users\\chyun\\Pictures\\cltp\\03.jpg', 0)  # 以灰度模式读取图像

# 使用 Otsu's 方法自动计算阈值
_, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('Otsu Threshold Result', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()