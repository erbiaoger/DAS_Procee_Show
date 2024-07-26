import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift 

def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    i = 0
    while n < N: 
        n *= 2
        i += 1
    return i

def fkDip(d, w):
    """
    FK dip滤波器的Python实现。

    参数:
    d : ndarray
        输入的二维数据数组。
    w : float
        锥形滤波器的半宽度（以百分比表示）。

    返回:
    d0 : ndarray
        应用FK dip滤波器后的输出数据。
    """
    # 获取输入数据的维度
    n1, n2 = d.shape
    # 计算进行FFT所需的最优长度
    nf = 2 ** nextpow2(n1)
    nk = 2 ** nextpow2(n2)
    nf2 = nf // 2
    nk2 = nk // 2

    # 沿第一个维度进行FFT
    Dfft1 = fft(d, n=nf, axis=0)
    
    # 截取到频谱的一半并继续处理
    Dtmp = Dfft1[:nf2 + 1, :]

    # 沿第二个维度进行FFT
    Dtmp2 = fft(Dtmp, n=nk, axis=1)
    Dtmp2 = fftshift(Dtmp2, axes=1)

    # 创建锥形遮罩
    nw = int(w * nk)
    nn1, nn2 = Dtmp2.shape
    mask = np.zeros((nn1, nn2))

    # 填充遮罩
    for i1 in range(nn1):
        for i2 in range(nn2):
            if i1 > (nn1 / nw) * (i2 - (nk2)) and i1 > (nn1 / nw) * ((nk2) - i2):
                mask[i1, i2] = 1

    # 应用遮罩
    Dtmp2 *= mask
    Dtmp = ifft(ifftshift(Dtmp2, axes=1), n=nk, axis=1)

    # 为逆FFT尊重对称性
    Dfft2 = np.zeros((nf, nk), dtype=np.complex128)
    Dfft2[:nf2+1, :] = Dtmp
    Dfft2[nf2+1:, :] = np.conj(np.flipud(Dtmp[1:-1, :]))

    # 最终逆FFT并裁剪到原始尺寸
    d0 = np.real(ifft(Dfft2, n=nf, axis=0))
    d0 = d0[:n1, :n2]

    return d0
