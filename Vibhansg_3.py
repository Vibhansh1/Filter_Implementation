import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, welch, freqz
from numpy.polynomial.polynomial import polymul

# Load audio
y, sr = librosa.load("F:\\Python\\EE200\\EE200_practical_programming\\song_with_2piccolo.wav", sr=None)

# Bandstop definition
def butter_bandstop(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='bandstop')

# Frequencies to suppress
flute_bands = [
    (800, 2100),
    (2600, 5300),
    (8000, 12000)
]

# Apply each filter sequentially
filtered = y.copy()
for lowcut, highcut in flute_bands:
    b, a = butter_bandstop(lowcut, highcut, sr)
    filtered = filtfilt(b, a, filtered)

# Save output
sf.write("restored_audio.wav", filtered, sr)

# PSD comparison
f_orig, psd_orig = welch(y, sr, nperseg=2048)
f_filt, psd_filt = welch(filtered, sr, nperseg=2048)

plt.figure(figsize=(12, 6))
plt.semilogy(f_orig, psd_orig, label='Original')
plt.semilogy(f_filt, psd_filt, label='Filtered')
plt.title('Power Spectral Density (PSD)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Bode + Pole-Zero plot per filter
for i, (lowcut, highcut) in enumerate(flute_bands):
    b, a = butter_bandstop(lowcut, highcut, sr)
    w, h = freqz(b, a, worN=4096, fs=sr)
    h_db = 20 * np.log10(np.abs(h) + 1e-10)
    zeros = np.roots(b)
    poles = np.roots(a)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Bode Magnitude
    axs[0].plot(w, h_db, 'm')
    axs[0].set_title(f"Bode Magnitude Response: {lowcut}-{highcut} Hz")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Gain (dB)")
    axs[0].grid(True)

    # Pole-Zero
    axs[1].plot(np.real(zeros), np.imag(zeros), 'go', label='Zeros')
    axs[1].plot(np.real(poles), np.imag(poles), 'rx', label='Poles')
    axs[1].add_artist(plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--'))
    axs[1].set_title("Pole-Zero Plot")
    axs[1].set_xlabel("Real")
    axs[1].set_ylabel("Imaginary")
    axs[1].axis('equal')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
