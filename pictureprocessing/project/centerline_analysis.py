import numpy as np
import cv2
import numpy as np

def compute_centerline(segment1, segment2):
    """计算两组边缘点的中心线，并处理顺序反转的情况"""
    # 确保输入数组是二维数组
    segment1 = np.array(segment1)
    segment2 = np.array(segment2)

    # 调整 segment1 的顺序为正序（如果需要）
    if segment1[0][0] > segment1[-1][0]:  # 判断是否为倒序
        segment1 = segment1[::-1]  # 反转段1，使其正序排列

    # 确保 segment2 的顺序为正序（如果需要）
    if segment2[0][0] > segment2[-1][0]:
        segment2 = segment2[::-1]

    # 计算段1的参数化
    t1 = np.cumsum(np.sqrt(np.sum(np.diff(segment1, axis=0) ** 2, axis=1)))
    t1 = np.insert(t1, 0, 0) / t1[-1]

    # 计算段2的参数化
    t2 = np.cumsum(np.sqrt(np.sum(np.diff(segment2, axis=0) ** 2, axis=1)))
    t2 = np.insert(t2, 0, 0) / t2[-1]

    # 生成新的参数值，并插值
    num_points = max(len(segment1), len(segment2))
    new_t = np.linspace(0, 1, num_points)

    segment1_interp = np.column_stack((
        np.interp(new_t, t1, segment1[:, 0]),
        np.interp(new_t, t1, segment1[:, 1])
    ))

    segment2_interp = np.column_stack((
        np.interp(new_t, t2, segment2[:, 0]),
        np.interp(new_t, t2, segment2[:, 1])
    ))

    # 计算中点
    centerline = np.column_stack((
        (segment1_interp[:, 0] + segment2_interp[:, 0]) / 2,
        (segment1_interp[:, 1] + segment2_interp[:, 1]) / 2
    ))

    return centerline
def save_centerline_to_file(centerline, filename='centerline.csv'):
    """将中心线数据保存到文件"""
    with open(filename, 'w') as f:
        for point in centerline:
            f.write(f"{point[0]},{point[1]}\n")

def draw_centerline(image, centerline, color=(0, 255, 0), thickness=2):
    """在图像上绘制中心线"""
    for i in range(len(centerline) - 1):
        # 直接使用整个行作为点坐标（已确保 centerline 的形状为 (n, 2)）
        pt1 = tuple(centerline[i].astype(int))
        pt2 = tuple(centerline[i + 1].astype(int))
        cv2.line(image, pt1, pt2, color, thickness)
    return image