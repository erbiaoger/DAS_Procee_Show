import numpy as np

def ncf_corre_cog(data1, nch, win, nwin, overlap, offset):
    """
    Calculate the normalized cross-correlation function (NCF) for ambient noise data.
    
    Parameters:
    data1 : numpy.ndarray
        Input data array with dimensions [time, channel].
    nch : int
        Number of channels in the data.
    win : int
        Length of the window for processing.
    nwin : int
        Number of windows to process.
    overlap : int
        Number of samples to overlap between windows.
    offset : int
        Offset between channels for cross-correlation.
        
    Returns:
    numpy.ndarray
        Normalized cross-correlation function array with dimensions [win, nch].
    """
    # Initialize the NCF array with zeros
    ncf = np.zeros((win, nch))
    
    # Loop over each window
    for iwin in range(nwin):
        start_idx = iwin * overlap
        end_idx = start_idx + win
        # Extract the current window of data
        data_cal = data1[start_idx:end_idx, :]
        # Perform FFT (Fast Fourier Transform) on the extracted data
        data_cal = np.fft.fft(data_cal, axis=0)
        
        # Loop over each trace/channel
        for itrace in range(nch):
            target_trace = itrace + offset
            # Check if the target trace is within the valid range
            if 0 <= target_trace < nch:
                # Compute cross-correlation
                temp1 = data_cal[:, itrace]
                temp2 = data_cal[:, target_trace]
                temp3 = np.fft.ifft(temp1 * np.conj(temp2)).real
                # Normalize the cross-correlation
                temp3 /= np.linalg.norm(temp3)
                # Accumulate the result in the NCF array
                ncf[:, target_trace] += temp3
                
    return ncf
