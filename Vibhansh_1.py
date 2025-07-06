import numpy as np
import matplotlib.pyplot as plt
import cv2

def create_gaussian_kernel_2d(shape, sigma):
    """Generate 2D Gaussian in frequency domain"""
    rows, cols = shape
    x = np.linspace(-cols//2, cols//2, cols)
    y = np.linspace(-rows//2, rows//2, rows)
    X, Y = np.meshgrid(x, y)
    D2 = X**2 + Y**2
    gaussian = np.exp(-D2 / (2 * sigma**2))
    return gaussian

def apply_filter_in_freq_domain(image, kernel):
    """Apply frequency-domain Gaussian filter to an image"""
    image_fft = np.fft.fft2(image, axes=(0, 1))
    kernel_shifted = np.fft.fftshift(kernel)

    # If image is color (3 channels), apply same filter to each
    if image.ndim == 3 and image.shape[2] == 3:
        filtered = np.zeros_like(image_fft)
        for c in range(3):
            filtered[:, :, c] = image_fft[:, :, c] * kernel_shifted
        result = np.fft.ifft2(filtered, axes=(0, 1)).real
    else:
        filtered = image_fft * kernel_shifted
        result = np.fft.ifft2(filtered).real

    return result

def get_low_freq(image, sigma=15):
    shape = image.shape[:2]
    kernel = create_gaussian_kernel_2d(shape, sigma)
    return apply_filter_in_freq_domain(image, kernel)

def get_high_freq(image, sigma=15):
    low = get_low_freq(image, sigma)
    high = image - low
    return high

def frequency_mix(image1, image2, sigma=15):
    low2 = get_low_freq(image2, sigma)
    high1 = get_high_freq(image1, sigma)
    mixed = low2 + high1
    return mixed

# ---- Main ---- #
shape = (512, 512)
sigma = 15

# Generate filter plots
lpf = create_gaussian_kernel_2d(shape, sigma)
hpf = 1 - lpf

# Load and prepare images
img1 = cv2.imread('F:\\Python\\EE200\\EE200_practical_programming\\cat_gray.jpg')
img2 = cv2.imread('F:\\Python\\EE200\\EE200_practical_programming\\dog_gray.jpg')

img1 = cv2.resize(img1, shape).astype(np.float32)
img2 = cv2.resize(img2, shape).astype(np.float32)

# Apply hybrid processing using custom Gaussian
mixed_img = frequency_mix(img1, img2, sigma)
mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)

# Prepare plots
titles = ['High Pass Filter for edges', 'Low Pass Filter for fill',
          'Image 1 (High Frequencies)', 'Image 2 (Low Frequencies)', 'Hybrid Image']
images = [
    hpf, lpf,
    get_high_freq(img1, sigma),
    get_low_freq(img2, sigma),
    mixed_img
]

# Plot results
plt.figure(figsize=(15, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    if images[i].ndim == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i].astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
