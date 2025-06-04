import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and grayscale
img = cv2.imread('assets\c1.jpg')  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 8))

# Step 1: Grayscale
plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title("Step 1: Grayscale")
plt.axis('off')

# Step 2: Thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

plt.subplot(2, 3, 2)
plt.imshow(thresh, cmap='gray')
plt.title("Step 2: Thresholded")
plt.axis('off')

# Step 3: Morphological operations with different kernels
kernel_open = np.ones((2, 2), np.uint8)     # Remove text
kernel_close = np.ones((5, 5), np.uint8)    # Fill coin holes

opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=3)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=10)

plt.subplot(2, 3, 3)
plt.imshow(closed, cmap='gray')
plt.title("Step 3: Morphology (Improved)")
plt.axis('off')

# Step 4: Apply white background mask
result = np.ones_like(img) * 255
result[closed == 255] = img[closed == 255]

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Step 4: Final Result (White BG)")
plt.axis('off')

plt.tight_layout()
plt.show()
