import numpy as np
import pandas as pd
from scipy import signal
import pywt  # For wavelet denoising

def remove_noise(data_path, method='wavelet', wavelet='db4', level=3, lowcut=0.5, highcut=30, fs=178):
    """
    EEG noise removal using:
    - Wavelet denoising (default)
    - Bandpass filtering (Butterworth)
    
    Parameters:
        data_path: Path to EEG data (CSV)
        method: 'wavelet' or 'bandpass'
        wavelet: Wavelet type (e.g., 'db4')
        level: Decomposition level for wavelet
        lowcut/highcut: Frequency range (Hz) for bandpass
        fs: Sampling frequency (Hz)
    """
    # Load data
    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:-1].values  # EEG channels
    y = df.iloc[:, -1].values
    
    # Binary labels: Class 1 (seizure) vs. rest
    y = np.where(y == 1, 1, 0)
    
    # Noise removal per channel
    X_clean = np.zeros_like(X)
    
    for i in range(X.shape[1]):  # Iterate over channels
        channel_data = X[:, i]
        
        if method == 'wavelet':
            # Wavelet Denoising
            coeffs = pywt.wavedec(channel_data, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-level])) / 0.6745  # Noise estimation
            uthresh = sigma * np.sqrt(2 * np.log(len(channel_data)))
            coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
            X_clean[:, i] = pywt.waverec(coeffs, wavelet)
            
        elif method == 'bandpass':
            # Bandpass Filter (Butterworth)
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            X_clean[:, i] = signal.filtfilt(b, a, channel_data)
    
    return X_clean, y

if __name__ == "__main__":
    # Example usage
    X_clean, y = remove_noise("data/epileptic_seizure.csv", method='wavelet')
    print("Cleaned data shape:", X_clean.shape)