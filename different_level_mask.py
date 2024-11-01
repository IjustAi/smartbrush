import cv2
import numpy as np

# k=(1,1) must be odd number, sigma=0 won't change image
def GaussianBlur(image_path, k, sigma):
    image = cv2.imread(image_path)
    image = np.array(image)
    blurred_image = cv2.GaussianBlur(image, (k,k), sigma)

    return blurred_image
    #cv2.imwrite(save_path, blurred_image)

