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
    if len(contours) > 0:
        # 找到最大轮廓（假设最大轮廓是最外层的轮廓）
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(output, [max_contour], -1, contour_color, contour_width)
    return output, max_contour


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
contour_image, max_contour = binarize_and_find_contours(segmented_image, contour_color=contour_color, contour_width=contour_width)

# 显示结果
cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imshow('Segmented Image', cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB) )
cv2.imshow('Contours Image', cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))


# 鼠标事件处理函数
def on_mouse(event, x, y, flags, param):
    global start_point, end_point, selecting, segment_count
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        selecting = True
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        selecting = False
        print(f"Selected segment: from {start_point} to {end_point}")


# 截取线段的函数
def extract_line_segment(contour, start_point, end_point):
    def point_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    min_start_dist = float('inf')
    min_end_dist = float('inf')
    start_index = 0
    end_index = 0
    for i, point in enumerate(contour):
        dist_start = point_distance(point[0], start_point)
        dist_end = point_distance(point[0], end_point)
        if dist_start < min_start_dist:
            min_start_dist = dist_start
            start_index = i
        if dist_end < min_end_dist:
            min_end_dist = dist_end
            end_index = i
    if start_index > end_index:
        start_index, end_index = end_index, start_index
    segment = contour[start_index:end_index + 1]
    return segment

# 注册鼠标事件
start_point = None
end_point = None
selecting = False
segment_count = 0
segment1 = []
segment2 = []
cv2.setMouseCallback('Contours Image', on_mouse)


while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按 ESC 键退出
        break
    if start_point is not None and end_point is not None:
        segment = extract_line_segment(max_contour, start_point, end_point)
        if segment_count == 0:
            segment1 = segment
            color = (0, 255, 0)  # 第一段线段用绿色
        elif segment_count == 1:
            segment2 = segment
            color = (0, 0, 255)  # 第二段线段用红色
        temp_image = image.copy()
        for i in range(len(segment) - 1):
            cv2.line(temp_image, tuple(segment[i][0]), tuple(segment[i + 1][0]), color, 2)
        cv2.imshow('Selected Segment', cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
        segment_count += 1
        start_point = None
        end_point = None

# 保存截取线段的坐标到文件
def save_segments_to_file(segments, filename='saved_segments.csv'):
    with open(filename, 'w') as f:
        for i, segment in enumerate(segments):
            f.write(f"Segment {i + 1}\n")
            for point in segment:
                f.write(f"{point[0]},{point[1]}\n")


# 保存截取的线段
save_segments_to_file(segment1, filename='segment1.csv')
save_segments_to_file(segment2, filename='segment2.csv')


cv2.destroyAllWindows()