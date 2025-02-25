import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    # 形态学操作：膨胀
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    output = image.copy()
    if len(contours) > 0:
        # 找到最大轮廓（假设最大轮廓是最外层的轮廓）
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(output, [max_contour], -1, contour_color, contour_width)
    return output


def extract_line_from_contour(contour, start_index, end_index):
    # 从轮廓中截取线段
    line = contour[start_index:end_index]
    return line


def fit_curve(points, degree=2):
    x = np.array([p[0][0] for p in points])
    y = np.array([p[0][1] for p in points])
    coefficients = np.polyfit(x, y, degree)
    poly_func = np.poly1d(coefficients)
    return poly_func


def main():
    image_path = 'C:\\Users\\chyun\\Pictures\\cltp\\03.jpg'
    image = cv2.imread(image_path)
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
    # 查找轮廓
    gray = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # 假设我们只对最大轮廓感兴趣
        max_contour = max(contours, key=cv2.contourArea)
        # 假设截取轮廓线的一部分，你可以根据实际情况修改 start_index 和 end_index
        start_index = 0
        end_index = len(max_contour) // 2
        extracted_line = extract_line_from_contour(max_contour, start_index, end_index)
        # 曲线拟合
        curve = fit_curve(extracted_line)
        # 生成用于绘制曲线的 x 坐标
        x = np.linspace(0, len(extracted_line), 100)
        y = curve(x)
        # 绘制结果
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.plot([p[0][0] for p in extracted_line], [p[0][1] for p in extracted_line], 'r-')
        plt.subplot(122)
        plt.plot(x, y, 'b-')
        plt.show()
    else:
        print("No contours found. Try adjusting LAB range or image preprocessing steps.")


if __name__ == "__main__":
    main()