import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_edge_detection(image, threshold=100, apply_smoothing=False):

    if apply_smoothing:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))

    _, thresholded = cv2.threshold(sobel_magnitude, threshold, 255, cv2.THRESH_BINARY)

    return sobel_magnitude, thresholded
 

if __name__ == "__main__":
    
    img = "LujainAbuRajab_HW2\flower_gray.jpg"  
    gray = cv2.imread(img)
    
    # Sobel edge detection
    edges, Threshold = sobel_edge_detection(gray, threshold=100, apply_smoothing=False)
    edge_smooth, Threshold_smooth = sobel_edge_detection(gray, threshold=100, apply_smoothing=True)

    edges_canny_mid = cv2.Canny(gray, 50, 60)
    edges_canny_larger_val = cv2.Canny(gray, 100, 200)

    titles = ['Original Image', 'img Edges', 'Smoothed Edges', 'Canny Edge Detection-50,60', 'Canny Edge Detection-100,200']
    images = [gray, edges, edge_smooth, edges_canny_mid, edges_canny_larger_val]

    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()
