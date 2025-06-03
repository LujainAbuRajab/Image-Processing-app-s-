import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('football_field.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# green regions mask
mask = cv2.inRange(hsv, lower_green, upper_green)

# red overlay 
red_overlay = np.zeros_like(image)
red_overlay[mask == 255] = [0, 0, 255]
result = cv2.addWeighted(image, 1.0, red_overlay, 0.5, 0)

combined = np.hstack((image, result))


cv2.imshow('Original vs Detected Field', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
