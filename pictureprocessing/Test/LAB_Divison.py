import cv2
import numpy as np


def lab_segmentation(image, lower_lab, upper_lab):
    # 将图像从 BGR 转换为 LAB 颜色空间
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # 将 LAB 范围值转换为 uint8 并映射 A 和 B 通道
    lower_lab_mapped = np.array([lower_lab[0], lower_lab[1] + 128, lower_lab[2] + 128], dtype=np.uint8)
    upper_lab_mapped = np.array([upper_lab[0], upper_lab[1] + 128, upper_lab[2] + 128], dtype=np.uint8)
    # 创建遮罩
    mask = cv2.inRange(image_lab, lower_lab_mapped, upper_lab_mapped)
    # 对原图像和遮罩进行位运算，提取感兴趣区域
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def binarize_and_find_contours(image, contour_color=(0, 255, 0), contour_width=2):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 自适应阈值处理
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    output = image.copy()
    cv2.drawContours(output, contours, -1, contour_color, contour_width)
    return output


# 读取图像
image = cv2.imread('C:\\Users\\chyun\\Pictures\\cltp\\03.jpg')
if image is None:
    raise FileNotFoundError("Image not found. Please check the path.")
# 定义 LAB 范围，这里使用 (44, 100, -127, 127, -66, 127)
lower_lab = (44, -127, -66)  # 下限 (L, A, B)
upper_lab = (100, 127, 127)  # 上限 (L, A, B)

# 进行 LAB 分割
segmented_image = lab_segmentation(image, lower_lab, upper_lab)

# 自定义轮廓颜色和线宽
contour_color = (255, 0, 0)  # 轮廓颜色，这里设置为蓝色 (B, G, R)
contour_width = 3  # 轮廓线宽，这里设置为 3 像素

# 二值化并绘制轮廓
contour_image = binarize_and_find_contours(segmented_image, contour_color=contour_color, contour_width=contour_width)

# 显示结果
cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imshow('Segmented Image', cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
cv2.imshow('Contours Image', cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()