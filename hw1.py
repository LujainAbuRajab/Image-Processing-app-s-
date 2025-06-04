import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r'assets\flower_gray.jpg', cv2.IMREAD_GRAYSCALE)

size = image.shape
thresholded_image = image.copy()

first_threshold_val = int(input("Enter threshold value (0-255): "))
sec_threshold_val = int(input("Enter threshold value (0-255): "))


for i in range(size[0]):
    for j in range(size[1]):
        if image[i, j] > first_threshold_val and image[i, j] < sec_threshold_val:
            coordinates_image = image[i, j]   
        else:
            coordinates_image = 0  
            
        thresholded_image[i, j] = coordinates_image

plt.imshow(thresholded_image)
plt.show()