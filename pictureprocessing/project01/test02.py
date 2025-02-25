'''
先用多项式拟合原始数据，然后再在拟合后的曲线上计算曲率。
这种方法看起来更平滑一些，因为多项式拟合会把原始数据的离散点转化成一条平滑的曲线。
不过，多项式拟合可能会引入一些误差，特别是当数据分布复杂或者有噪声的时候，拟合出来的曲线可能不一定完全符合原始数据的实际情况。
'''
import numpy as np
import pandas as pd

# 重新读取数据
file_path = 'C:\\Users\\chyun\\Documents\\AMyAllCode Files\\pycharmProject\\pictureprocessing\\project\\centerline.csv'
data = pd.read_csv(file_path, header=None)

# 重新进行多项式拟合
coefficients = np.polyfit(data[0], data[1], 3)
polynomial = np.poly1d(coefficients)

# 生成用于绘制拟合曲线的X值
x_fit = np.linspace(data[0].min(), data[0].max(), 500)
y_fit = polynomial(x_fit)

# 计算拟合曲线的一阶和二阶导数
polynomial_derivative_1 = np.polyder(polynomial, 1)
polynomial_derivative_2 = np.polyder(polynomial, 2)

# 计算曲率
y_prime = polynomial_derivative_1(x_fit)
y_double_prime = polynomial_derivative_2(x_fit)
curvature = np.abs(y_double_prime) / (1 + y_prime**2)**(3/2)

# 计算总绝对曲率
total_absolute_curvature = np.trapezoid(curvature, x_fit)

print(f'绝对总曲率为：{total_absolute_curvature}')