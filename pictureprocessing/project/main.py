import cv2
import numpy as np
from image_processing import ImageProcessing
from image_interactor import handle_mouse_interactions
from centerline_analysis import compute_centerline, save_centerline_to_file, draw_centerline

def main():
    # 读取图像
    image_path = 'C:\\Users\\chyun\\Pictures\\cltp\\10.jpg'  # 替换为你的图像路径
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("图像未找到，请检查路径！")

    # 查找轮廓
    contours = ImageProcessing(image)

    # 获取最大轮廓
    max_contour = max(contours, key=cv2.contourArea) if contours else None

    # 启动鼠标交互
    handle_mouse_interactions(image, max_contour)

if __name__ == "__main__":
    main()