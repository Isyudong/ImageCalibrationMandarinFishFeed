import cv2
import numpy as np


def resize_image_to_screen(img, screen_width=800, screen_height=600):
    h, w = img.shape[:2]
    scale = min(screen_width / w, screen_height / h)
    new_size = (int(w * scale), int(h * scale))
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return resized_img


# 读取图像
image = cv2.imread('C:\\Users\\chyun\\Pictures\\cltp\\03.jpg')
if image is None:
    raise FileNotFoundError("Image not found. Please check the path.")

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 调整灰度图像的大小
gray_resized = resize_image_to_screen(gray)
cv2.imshow('Gray Image', gray_resized)
cv2.waitKey(1)  # 修改为 cv2.waitKey(1)

# 使用Canny边缘检测
edges = cv2.Canny(gray, 50, 150)
# 调整边缘图像的大小
edges_resized = resize_image_to_screen(edges)
cv2.imshow('Edges', edges_resized)
cv2.waitKey(1)  # 修改为 cv2.waitKey(1)

# 检测轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print("No contours found. Please check the image and preprocessing steps.")
else:
    # 绘制轮廓
    output = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    # 调整轮廓图像的大小
    output_resized = resize_image_to_screen(output)
    cv2.imshow('Contours', output_resized)
    cv2.waitKey(1)  # 修改为 cv2.waitKey(1)


# 调整图片大小以适应屏幕大小
resized_output = resize_image_to_screen(output)

# 显示结果
cv2.imshow('Resized Contours', resized_output)
cv2.waitKey(0)
cv2.destroyAllWindows()