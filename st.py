import numpy as np
import scipy.fftpack as fft
import scipy.signal as signal

def st(timeseries, minfreq=0, maxfreq=None, samplingrate=1, freqsamplingrate=1):
    """
    Returns the Stockwell Transform of the timeseries.
    Converted to Python from MATLAB code by Robert Glenn Stockwell.

    Parameters:
    timeseries (numpy.ndarray): vector of data to be transformed
    minfreq (int, optional): the minimum frequency to be sampled (Default=0)
    maxfreq (int, optional): the maximum frequency to be sampled (Default=Nyquist)
    samplingrate (int, optional): the minimum frequency to be sampled (Default=1)
    freqsamplingrate (int, optional): the frequency-sampling interval of the spectrum (Default=1)

    Returns:
    st (numpy.ndarray): a complex matrix containing the Stockwell transform.
    t (numpy.ndarray): a vector containing the sampled times.
    f (numpy.ndarray): a vector containing the sampled frequencies.
    """
    
    # Default parameters
    verbose = True
    power = False
    amplitude = False
    removeedge = True
    analytic_signal = True
    factor = 1

    # Make sure the timeseries is a column vector
    timeseries = np.asarray(timeseries)
    if timeseries.ndim == 1:
        timeseries = timeseries[:, np.newaxis]
    
    if timeseries.shape[1] > 1:
        raise ValueError("Please enter a *vector* of data, not matrix")
    
    if maxfreq is None:
        maxfreq = len(timeseries) // 2

    minfreq, maxfreq, samplingrate, freqsamplingrate = check_input(
        minfreq, maxfreq, samplingrate, freqsamplingrate, verbose, timeseries
    )

    if verbose:
        print(f"Minfreq = {minfreq}")
        print(f"Maxfreq = {maxfreq}")
        print(f"Sampling Rate (time domain) = {samplingrate}")
        print(f"Sampling Rate (freq. domain) = {freqsamplingrate}")
        print(f"The length of the timeseries is {len(timeseries)} points")

    # calculate the sampled time and frequency values from the two sampling rates
    t = np.arange(len(timeseries)) * samplingrate
    spe_nelements = int(np.ceil((maxfreq - minfreq + 1) / freqsamplingrate))
    f = (minfreq + np.arange(spe_nelements) * freqsamplingrate) / (samplingrate * len(timeseries))
    if verbose:
        print(f"The number of frequency voices is {spe_nelements}")

    # The actual S Transform function is here:
    st = strans(timeseries, minfreq, maxfreq, samplingrate, freqsamplingrate, verbose, removeedge, analytic_signal, factor)

    if power:
        st = np.abs(st) ** 2
    elif amplitude:
        st = np.abs(st)

    return st, t, f

def strans(timeseries, minfreq, maxfreq, samplingrate, freqsamplingrate, verbose, removeedge, analytic_signal, factor):
    """
    Returns the Stockwell Transform of the timeseries.
    """

    n = len(timeseries)
    original = timeseries.copy()

    if removeedge:
        if verbose:
            print("Removing trend with polynomial fit")
        ind = np.arange(n)
        r = np.polyfit(ind, timeseries.flatten(), 2)
        fit = np.polyval(r, ind)
        timeseries = timeseries - fit[:, np.newaxis]

        if verbose:
            print("Removing edges with 5% hanning taper")
        sh_len = n // 10
        wn = signal.hann(sh_len)
        if sh_len == 0:
            sh_len = n
            wn = np.ones(sh_len)
        timeseries[:sh_len//2] *= wn[:sh_len//2, np.newaxis]
        timeseries[-sh_len//2:] *= wn[-sh_len//2:, np.newaxis]

    if analytic_signal:
        if verbose:
            print("Calculating analytic signal (using Hilbert transform)")
        ts_spe = fft.fft(timeseries.flatten())
        h = np.zeros(n)
        h[0] = 1
        h[1:n//2] = 2
        h[n//2] = 1
        ts_spe *= h
        timeseries = fft.ifft(ts_spe)[:, np.newaxis]

    vector_fft = fft.fft(timeseries.flatten())
    vector_fft = np.concatenate([vector_fft, vector_fft])
    st = np.zeros((int((maxfreq - minfreq + 1) / freqsamplingrate), n), dtype=complex)

    if verbose:
        print("Calculating S transform...")

    if minfreq == 0:
        st[0, :] = np.mean(timeseries) * np.ones(n)
    else:
        st[0, :] = fft.ifft(vector_fft[minfreq:minfreq+n] * g_window(n, minfreq, factor))

    for banana in range(freqsamplingrate, maxfreq - minfreq + 1, freqsamplingrate):
        st[banana // freqsamplingrate, :] = fft.ifft(vector_fft[minfreq + banana:minfreq + banana + n] * g_window(n, minfreq + banana, factor))

    if verbose:
        print("Finished Calculation")

    return st

def g_window(length, freq, factor):
    """
    Function to compute the Gaussian window for the S transform.
    """
    vector = np.array([np.arange(length), -np.arange(length, 0, -1)]) ** 2
    vector = vector * (-factor * 2 * np.pi**2 / freq**2)
    gauss = np.sum(np.exp(vector), axis=0)
    return gauss

def check_input(minfreq, maxfreq, samplingrate, freqsamplingrate, verbose, timeseries):
    """
    Checks and validates input parameters, replacing with defaults if invalid.
    """
    n = len(timeseries)

    if minfreq < 0 or minfreq > n // 2:
        minfreq = 0
        if verbose:
            print("Minfreq < 0 or > Nyquist. Setting minfreq = 0.")

    if maxfreq > n // 2 or maxfreq < 0:
        maxfreq = n // 2
        if verbose:
            print(f"Maxfreq < 0 or > Nyquist. Setting maxfreq = {maxfreq}")

    if minfreq > maxfreq:
        minfreq, maxfreq = maxfreq, minfreq
        if verbose:
            print("Swapping maxfreq <=> minfreq.")

    if samplingrate < 0:
        samplingrate = abs(samplingrate)
        if verbose:
            print("Samplingrate < 0. Setting samplingrate to its absolute value.")

    if freqsamplingrate < 0:
        freqsamplingrate = abs(freqsamplingrate)
        if verbose:
            print("Frequency Samplingrate negative, taking absolute value")

    return minfreq, maxfreq, samplingrate, freqsamplingrate
