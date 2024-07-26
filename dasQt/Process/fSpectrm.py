import numpy as np
import matplotlib.pyplot as plt


def computeFFT(sr, dt, N=None):
    if N is None:
        N = len(sr)

    Y = np.fft.fft(sr, n=N, axis=0)

    freq = np.fft.fftfreq(len(Y), dt)[:len(Y)//2]
    Y = np.abs(Y[:len(Y)//2])

    return freq, Y

def fft_shift(signal, dt, N=None):
    if N is None:
        N = len(signal)
    # FFT计算和fftshift处理
    fft_result = np.fft.fft(signal, n=N)
    fft_result_shifted = np.fft.fftshift(fft_result)

    # 生成fftshift后的频率横坐标
    freq_shifted = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    
    return freq_shifted, fft_result_shifted


def plot_fft_result(ax, freq, fft_result):
    # 绘制傅里叶变换结果
    ax.plot(freq, fft_result)
    ax.set_title('Fourier Transform Result')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')

def main():
    # 创建一维信号
    f     = 5.5
    dt    = 0.004
    v0    = 250
    dv    = 180
    sigma = 20
    decay = 50000
    x     = np.arange(0, 1000, 8)


    x, y = ricker(f, dt)
    

    # 计算傅里叶变换
    freq, fft_result = compute_fft(y, dt, N=2500)
    
    # 应用fftshift将零频分量移至中心
    freq_shift, fft_shift_result  = fft_shift(y, dt)

    # 绘制原始信号
    plt.subplot(3, 1, 1)
    plt.plot(x, y)
    plt.title('Original Signal')

    # 绘制傅里叶变换结果
    plt.subplot(3, 1, 2)
    plot_fft_result(freq, fft_result)

    # 绘制移位后的傅里叶变换结果
    plt.subplot(3, 1, 3)
    plt.plot(freq_shift, np.abs(fft_shift_result))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()