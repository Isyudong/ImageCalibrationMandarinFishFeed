import cv2
import numpy as np

def extract_line_segment(contour, start_point, end_point):
    """在轮廓上提取两点之间的最短线段"""
    # 计算轮廓上所有点到起始点和结束点的距离
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

    # 确保起始点在结束点之前
    if start_index > end_index:
        start_index, end_index = end_index, start_index

    # 提取两点之间的线段，并调整形状为 (M, 2)
    segment = contour[start_index:end_index + 1].reshape(-1, 2)
    return segment

def save_segments_to_file(segments, filenames=['segment1.csv', 'segment2.csv']):
    for idx, segment in enumerate(segments):
        if idx >= len(filenames):
            break
        np.savetxt(filenames[idx], segment, delimiter=',', fmt='%d')