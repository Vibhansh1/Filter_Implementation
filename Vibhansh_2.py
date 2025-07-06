import numpy as np
import matplotlib.pyplot as plt
import cv2

def magnitude_spectra(image):
    fft_result = np.fft.fft2(image)
    magnitude = np.abs(fft_result)
    bode_magnitude = 20 * np.log(magnitude + 1)

    fft_new = np.fft.fftshift(fft_result)
    magnitude_new = np.abs(fft_new)
    bode_new = 20 * np.log(magnitude_new + 1)

    return magnitude, bode_magnitude, magnitude_new, bode_new

image = cv2.imread('F:\\Python\\EE200\\EE200_practical_programming\\dog_gray.jpg', cv2.IMREAD_GRAYSCALE)
image_rotated = np.rot90(image)

mag, bode, mag_new, bode_new = magnitude_spectra(image)
magRot, bodeRot, mag_newRot, bode_newRot = magnitude_spectra(image_rotated)

plt.figure(figsize=(20, 6))

# Original Magnitude
plt.subplot(2, 4, 1)
plt.imshow(mag, cmap='gray', vmax=np.percentile(mag_new, 99))
plt.title('Magnitude')
plt.axis('off')

# Original Bode
plt.subplot(2, 4, 2)
plt.imshow(bode, cmap='gray')
plt.title('Bode (dB)')
plt.axis('off')

# new Magnitude
plt.subplot(2, 4, 3)
plt.imshow(mag_new, cmap='gray', vmax=np.percentile(mag_new, 99))
plt.title('new Magnitude')
plt.axis('off')

# new Bode
plt.subplot(2, 4, 4)
plt.imshow(bode_new, cmap='gray')
plt.title('new Bode (dB)')
plt.axis('off')

# Rotated Magnitude
plt.subplot(2, 4, 5)
plt.imshow(magRot, cmap='gray', vmax=np.percentile(magRot, 99))
plt.title('Magnitude (90deg Rotated)')
plt.axis('off')

# Rotated Bode
plt.subplot(2, 4, 6)
plt.imshow(bodeRot, cmap='gray')
plt.title('Bode (90deg) Rotated')
plt.axis('off')

# new Rotated Magnitude
plt.subplot(2, 4, 7)
plt.imshow(mag_newRot, cmap='gray', vmax=np.percentile(mag_newRot, 99))
plt.title('new Magnitude (90deg) Rotated')
plt.axis('off')

# new Rotated Bode
plt.subplot(2, 4, 8)
plt.imshow(bode_newRot, cmap='gray')
plt.title('new Bode (90deg) Rotated')
plt.axis('off')

plt.tight_layout()
plt.show()
