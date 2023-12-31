from gaussianblur import gaussian_blur
from expandimage import expand_image
from greyscale import greyscale_filter
from PIL import Image
import numpy as np
import sys

def sobel_filter(image : np.array, colored : bool) -> np.array:
    grey_image = image

    if colored:
        grey_image = greyscale_filter(image)
    kernel_radius = 2 # Modifiable value, the higher it is the more edges are ignored
    kernel_width = kernel_radius * 2 + 1
    grey_image_blurred = gaussian_blur(grey_image, kernel_radius)
    expanded_image = expand_image(grey_image_blurred, kernel_width + 1)

    horizontal_kernel = np.reshape(np.array([[1.0 ,2.0 ,1.0 ], [0.0 ,0.0 ,0.0], [-1.0, -2.0 , -1.0]]), (3,3,1))
    vertical_kernel = np.reshape(np.array([[1.0, 0.0, -1.0], [2.0 ,0.0 ,-2.0], [1.0 ,0.0 ,-1.0]]), (3,3,1))
    sobel_filtered_image = np.zeros_like(image)


    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            neighbourhood_values = expanded_image[row + kernel_width : row + kernel_width + 3, column + kernel_width: column + kernel_width + 3, 0:1]
            gx = np.sum(np.multiply(neighbourhood_values, horizontal_kernel))
            gy = np.sum(np.multiply(neighbourhood_values, vertical_kernel))
            sobel_filtered_image[row : row + 1, column : column + 1] = np.sqrt(gx ** 2 + gy ** 2)

    if sobel_filtered_image.shape[2] == 4:
        sobel_filtered_image[:,:,3:4] = 255

    return sobel_filtered_image.astype("uint8")

if __name__ == "__main__":
    image = np.asarray(Image.open(sys.path[0] + '/../Test Image.png'))
    Image.fromarray(sobel_filter(image, True)).show()

