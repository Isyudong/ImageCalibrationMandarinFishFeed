import cv2
import numpy as np
from contour_extraction import extract_line_segment, save_segments_to_file
from centerline_analysis import compute_centerline, save_centerline_to_file, draw_centerline
from curvature_analysis import compute_curvature, plot_curvature
# 全局变量初始化
start_point = None
end_point = None
selecting = False
segment_count = 0
segments = []

def handle_mouse_event(event, x, y, flags, param):
    global start_point, end_point, selecting
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        selecting = True
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        selecting = False
        print(f"Selected segment: from {start_point} to {end_point}")

def handle_mouse_interactions(image, max_contour):
    global start_point, end_point, selecting, segment_count, segments

    # 绘制初始轮廓
    contour_image = image.copy()
    if max_contour is not None:
        cv2.drawContours(contour_image, [max_contour], -1, (255, 0, 0), 3)
    cv2.namedWindow('Contours Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Contours Image', 800, 600)
    cv2.imshow('Contours Image', cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))

    # 设置鼠标回调
    cv2.setMouseCallback('Contours Image', handle_mouse_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 按 ESC 键退出
            break

        if not selecting and start_point is not None and end_point is not None:
            if max_contour is not None:
                segment = extract_line_segment(max_contour, start_point, end_point)
                segments.append(segment)
                segment_count += 1
                start_point = end_point = None  # 重置起点和终点

                # 绘制选中的线段
                temp_image = image.copy()
                for i, segment in enumerate(segments):
                    color = (0, 255, 0) if i == 0 else (0, 0, 255)  # 第一段绿色，第二段红色
                    for j in range(len(segment) - 1):
                        # 直接使用整数索引
                        pt1 = tuple(segment[j])
                        pt2 = tuple(segment[j + 1])
                        cv2.line(temp_image, pt1, pt2, color, 2)
                cv2.namedWindow('Contours Image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Contours Image', 800, 600)
                cv2.imshow('Contours Image', cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))

    # 保存线段数据到文件
    save_segments_to_file(segments)

    # 计算中心线并绘制
    if len(segments) == 2:
        centerline = compute_centerline(segments[0], segments[1])
        image_with_centerline = draw_centerline(image.copy(), centerline, (255, 255, 0))  # 中心线为黄色

        # 保存中心线数据到文件
        np.savetxt('centerline.csv', centerline, delimiter=',', fmt='%d')
        # # 计算曲率
        # curvatures = compute_curvature(centerline)
        # if curvatures:
        #     plot_curvature(curvatures)
        # else:
        #     print("曲率计算至少需要3个点")

        cv2.namedWindow('Contours Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Contours Image', 800, 600)
        cv2.imshow('Contours Image', cv2.cvtColor(image_with_centerline, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
    cv2.destroyAllWindows()