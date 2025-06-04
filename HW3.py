import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r'C:\Users\Lujain AbuRajab\OneDrive\Documents\ImgProcessing_HW\assets\football_field.jpg')  # or use r'assets\field.png'
if image is None:
    raise FileNotFoundError("Image not found. Please check the path.")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

red_overlay = np.zeros_like(image)
red_overlay[mask == 255] = [0, 0, 255]
result = cv2.addWeighted(image, 1.0, red_overlay, 0.5, 0)

combined = np.hstack((image, result))

# Use this if you're running in a notebook or headless environment
plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
plt.title('Original vs Detected Field')
plt.axis('off')
plt.show()
