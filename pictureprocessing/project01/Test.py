import cv2
import numpy as np

# 自定义轮廓颜色和线宽
contour_color = (0, 0, 255)  # 轮廓颜色
contour_width = 2  # 轮廓线宽


# 读取图像
image = cv2.imread('C:\\Users\\chyun\\Pictures\\cltp\\01.jpg')
if image is None:
    raise FileNotFoundError("Image not found. Please check the path.")

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化
retval, dst = cv2.threshold(gray,thresh=50,maxval=255,type=cv2.THRESH_BINARY_INV) # 二值化阈值(87, 255)

# 定义核
kernel = np.ones((5, 5), np.uint8)
# #膨胀与腐化
# dilated_image = cv2.dilate(dst, kernel, iterations=3)
# eroded_image = cv2.erode(dilated_image, kernel, iterations=3)
closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

# 查找轮廓
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 区分上轮廓和下轮廓
height = image.shape[0]
upper_contours = []
lower_contours = []
# 遍历轮廓
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # 根据轮廓的位置将其添加到上部或下部轮廓列表中
    if y < height / 2:
        upper_contours.append(contour)
    else:
        lower_contours.append(contour)

# 绘制上部和下部轮廓
sb = cv2.drawContours(image, upper_contours, -1, (0, 0, 255), 2)
xb = cv2.drawContours(image, lower_contours, -1, (255, 0, 0), 2)


# 显示结果
# cv2.imshow('Original Image', image)
# cv2.imshow('gray',gray)
# cv2.imshow('erzhihua',dst)
# cv2.imshow('fs',eroded_image)
cv2.namedWindow('finalImage',cv2.WINDOW_NORMAL)
cv2.resizeWindow('finalImage',800, 600)
cv2.imshow('finalImage',closed)

cv2.waitKey(0)
cv2.destroyAllWindows()