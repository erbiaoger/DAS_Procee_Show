#%%
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

from dasQt import das
from fSpectrm import computeFFT, plot_fft_result
from bpFilter import bpFilter
from fkDip import fkDip

#%%
def loadFile(file):
    das1 = das.DAS()
    das1.readData(file)
    data = das1.data[:, 73:-1]
    dx = das1.dx
    dt = das1.dt

    return data, dx, dt

dir_path = pathlib.Path('/Volumes/CSIM_LAB/DATA/Car-JLU-2024-01-31/ovlink/2024-02-01')
files = sorted(dir_path.glob('*.dat'))
data, dx, dt = loadFile(files[20])
nt, nx = data.shape

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.imshow(data, aspect='auto', cmap='seismic', extent=[0, nx*dx, nt*dt, 0])
ax.set_title('Original Data')
plt.show()

#%%

freq, fft_result = computeFFT(data[:, 20], dt, N=2500)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plot_fft_result(ax, freq, fft_result)
ax.set_title('Original Data FFT')
plt.show()

#%%
f1, f2, f3, f4 = 0.01, 0.1, 2.3, 2.5

data_bp = bpFilter(data, dt, f1, f2, f3, f4)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.imshow(data_bp, aspect='auto', cmap='seismic', extent=[0, nx*dx, nt*dt, 0])
ax.set_title('Bandpass Filtered Data')
plt.show()

#%%
freq, fft_result = computeFFT(data_bp[:, 20], dt, N=2500)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plot_fft_result(ax, freq, fft_result)
ax.set_title('Bandpass Filtered Data FFT')
plt.show()

#%%
data_fk2 = data_bp - fkDip(data_bp,0.08)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.imshow(data_fk2, aspect='auto', cmap='seismic', extent=[0, nx*dx, nt*dt, 0])
ax.set_title('Bandpass Filtered Data')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.imshow(data_fk2-data_bp, aspect='auto', cmap='seismic', extent=[0, nx*dx, nt*dt, 0])
ax.set_title('Bandpass Filtered Data')
plt.show()

# %%
import numpy as np
from pycurvelab import fdct2, ifdct2  # 确保已安装PyCurvelab或找到替代的曲波变换库

def denoise_with_curvelets(data, alpha=3, niter=10):
    """
    使用曲波变换对图像进行去噪。

    参数:
    data : ndarray
        输入的二维数据数组（图像数据）。
    alpha : float
        阈值调整参数。
    niter : int
        动态阈值迭代次数。

    返回:
    denoised_data : ndarray
        去噪后的输出数据。
    """
    data = data.T  # 转置数据以匹配MATLAB的方向
    n1, n2 = data.shape
    is_real = True
    finest = 1  # 在最精细的层级使用曲波

    # 对全1数组进行曲波变换，用以计算曲波的范数
    F = np.ones((n1, n2))
    X = np.fft.fftshift(np.fft.ifft2(F)) * np.sqrt(np.prod(F.shape))
    C = fdct2(X, is_real=is_real, finest=finest)

    # 计算曲波的范数
    E = [np.sqrt(np.sum(np.abs(c**2)) / c.size) for c in C]

    # 对数据进行曲波变换
    Cdn = fdct2(data, is_real=is_real, finest=finest)

    # 阈值处理
    Smax = len(Cdn)
    Sigma0 = alpha * np.median(np.abs(Cdn[Smax - 1])) / 0.35
    sigma = [Sigma0 * (2.5 - 2 * i / (niter - 1)) for i in range(niter)]
    Sigma = sigma[0]

    for s in range(1, len(Cdn)):
        thresh = Sigma + Sigma * s
        for w in range(len(Cdn[s])):
            Cdn[s][w] = Cdn[s][w] * (np.abs(Cdn[s][w]) > thresh * E[s])

    # 进行逆曲波变换
    denoised_data = np.real(ifdct2(Cdn, is_real=is_real, n1=n1, n2=n2))

    return denoised_data

# 示例用法
data = np.random.randn(256, 256)  # 使用实际的数据替换此处
denoised_data = denoise_with_curvelets(data)








