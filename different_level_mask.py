import cv2
import numpy as np

#30 50 70 80 100 
def GaussianBlur(image_path, num_steps):
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_epsilon = 300 

    for step in range(num_steps):
        result = np.zeros_like(mask)
        scale_factor = (num_steps - step) / num_steps
        epsilon = scale_factor * max_epsilon
        approx_contour = cv2.approxPolyDP(contours[0], epsilon=epsilon, closed=True)

        cv2.drawContours(result, [approx_contour], -1, 255, thickness=cv2.FILLED)
    bluerd_mask = np.array(result)
    return bluerd_mask 

