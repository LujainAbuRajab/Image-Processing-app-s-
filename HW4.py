import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/c1.jpg') 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title("Step 1: Grayscale")
plt.axis('off')

_, thresh = cv2.threshold(gray, 100, 150, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

plt.subplot(2, 3, 2)
plt.imshow(thresh, cmap='gray')
plt.title("Step 2: Thresholded")
plt.axis('off')

kernel = np.ones((2, 2), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=5)

plt.subplot(2, 3, 3)
plt.imshow(closing, cmap='gray')
plt.title("Step 3: Morphology (Cleaned)")
plt.axis('off')

mask = cv2.bitwise_not(closing)
result = cv2.bitwise_and(img, img, mask=mask)

plt.subplot(2, 3, 4)
plt.imshow(mask, cmap='gray')
plt.title("Step 4: Mask")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Step 4: Final Result (Coins Only)")
plt.axis('off')

plt.tight_layout()
plt.show()
