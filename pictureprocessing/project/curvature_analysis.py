import numpy as np
import matplotlib.pyplot as plt

def compute_curvature(centerline):
    """计算中心线的曲率"""
    # 确保中心线至少有3个点
    if len(centerline) < 3:
        return []

    curvatures = []
    for i in range(1, len(centerline) - 1):
        # 前一点、当前点、后一点
        p_prev = centerline[i - 1]
        p_curr = centerline[i]
        p_next = centerline[i + 1]

        # 计算向量
        vector1 = p_prev - p_curr
        vector2 = p_next - p_curr

        # 计算夹角的余弦值
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        if magnitude1 == 0 or magnitude2 == 0:
            cosine_theta = 1.0
        else:
            cosine_theta = dot_product / (magnitude1 * magnitude2)

        # 计算角度
        theta = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
        # 转换为角度制
        angle_deg = np.degrees(theta)

        # 曲率与角度有关，这里可以简单地取角度作为曲率值
        curvature = angle_deg

        curvatures.append(curvature)

    return curvatures

def plot_curvature(curvatures):
    """绘制曲率图"""
    plt.switch_backend('TkAgg')
    plt.figure(figsize=(10, 5))
    plt.plot(curvatures, marker='o', linestyle='-', color='b')
    plt.title("Curvature along the Centerline")
    plt.xlabel("Point Index")
    plt.ylabel("Curvature (degrees)")
    plt.grid()
    plt.show()