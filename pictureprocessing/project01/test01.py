'''
直接在原始数据点上进行计算的，这种方法应该能最大程度保留原始数据的特征。
它用中心差分法计算导数，然后用弧长积分来计算总曲率。
这种方法应该是比较直接的，应该挺适合处理离散点的情况。
如果数据点不均匀或者有点噪声，这种方法会不会不太稳定呢？毕竟它是直接用点来计算的，没有经过任何平滑处理。
'''
import pandas as pd
import numpy as np

#文件地址
image_path = 'C:\\Users\\chyun\\Documents\\AMyAllCode Files\\pycharmProject\\pictureprocessing\\project\\centerline.csv'
# 读取CSV文件
df = pd.read_csv(image_path, header=None, names=['x', 'y'])
x = df['x'].values.astype(float)
y = df['y'].values.astype(float)

# 去除连续重复的点
mask = np.concatenate(([True], (np.diff(x) != 0) | (np.diff(y) != 0)))
x_filtered = x[mask]
y_filtered = y[mask]

n = len(x_filtered)
if n < 3:
    print("错误：至少需要3个非重复点以计算曲率。")
else:
    # 计算相邻点之间的弧长ds
    dx = np.diff(x_filtered)
    dy = np.diff(y_filtered)
    ds = np.sqrt(dx ** 2 + dy ** 2)

    # 计算中间点的平均弧长微元（ds_mid）
    ds_mid = (ds[:-1] + ds[1:]) / 2.0

    total_curvature = 0.0

    for i in range(1, n - 1):
        # 中心差分计算一阶导数
        dx_dt = (x_filtered[i + 1] - x_filtered[i - 1]) / 2.0
        dy_dt = (y_filtered[i + 1] - y_filtered[i - 1]) / 2.0

        # 中心差分计算二阶导数
        d2x_dt2 = x_filtered[i + 1] - 2 * x_filtered[i] + x_filtered[i - 1]
        d2y_dt2 = y_filtered[i + 1] - 2 * y_filtered[i] + y_filtered[i - 1]

        # 计算曲率的分子和分母
        numerator = abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        denominator = (dx_dt ** 2 + dy_dt ** 2) ** 1.5

        # 处理分母为零的情况
        if denominator == 0:
            curvature = 0.0
        else:
            curvature = numerator / denominator

        # 获取对应的弧长微元（若i在有效范围内）
        if (i - 1) < len(ds_mid):
            ds_i = ds_mid[i - 1]
        else:
            ds_i = 0.0

        total_curvature += abs(curvature) * ds_i

    print(f"总绝对曲率为：{total_curvature:.4f}")