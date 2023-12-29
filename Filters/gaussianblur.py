
from expandimage import expand_image
from PIL import Image
import numpy as np
import math
import sys

#Calculate the kernel matrix
def gaussian_kernel_matrix(radius : int, kernel_width: int) -> np.array:
    sigma = max(float(radius / 2) , 1)
    kernel = [[0.0 for _ in range(kernel_width)] for _ in range(kernel_width)]

    sum = 0
    for x in range(-radius, radius):
        for y in range(-radius, radius):
            exponentNumerator = float(-(x * x + y * y))
            exponentDenominator = (2 * sigma * sigma)

            eExpression = math.pow(math.e, exponentNumerator / exponentDenominator)
            kernelValue = (eExpression / (2 * math.pi * sigma * sigma))
            kernel[x + radius][y + radius] = kernelValue
            sum += kernelValue

    for x in range (0, kernel_width):
        for y in range(0, kernel_width):
            kernel[x][y] /= sum

    kernel = np.reshape(kernel, (kernel_width, kernel_width, 1))

    return kernel

#Place the center of the kernel at each pixel of the image
#Multiply the kernel matrix values with the corresponding pixel values in the local neighbourhood
#Sum up these products to obtain the new value for the pixel
def gaussian_blur(image: np.array, radius : int) -> np.array:
    gaussian_filtered_image = np.zeros_like(image)
    kernel_width = (2 * radius) + 1

    kernel_matrix = gaussian_kernel_matrix(radius, kernel_width)
    expanded_image = expand_image(image, kernel_width)

    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            neighbourhood_red = expanded_image[row : row + kernel_width, column : column + kernel_width, 0:1]
            neighbourhood_green = expanded_image[row : row + kernel_width, column : column + kernel_width, 1:2]
            neighbourhood_blue = expanded_image[row : row + kernel_width, column : column + kernel_width, 2:3]

            red_sum = np.sum(np.multiply(neighbourhood_red, kernel_matrix))
            green_sum = np.sum(np.multiply(neighbourhood_green, kernel_matrix))
            blue_sum = np.sum(np.multiply(neighbourhood_blue, kernel_matrix))

            gaussian_filtered_image[row : row + 1, column : column + 1, 0:1] = red_sum
            gaussian_filtered_image[row : row + 1, column : column + 1, 1:2] = green_sum
            gaussian_filtered_image[row : row + 1, column : column + 1, 2:3] = blue_sum

    if gaussian_filtered_image.shape[2] == 4:
        gaussian_filtered_image[:,:,3:4:1] = 255

    return gaussian_filtered_image.astype("uint8")

if __name__ == "__main__":
    image1 = np.asarray(Image.open(sys.path[0] + '/../Test Image.png'))
    image2 = np.asarray(Image.open(sys.path[0] + '/../Valve.png'))
    image3 = np.asarray(Image.open(sys.path[0] + '/../Test Image 2.png'))

    Image.fromarray(gaussian_blur(image1, 15)).show()
    Image.fromarray(gaussian_blur(image2, 15)).show()
    Image.fromarray(gaussian_blur(image3, 15)).show()




"""
THINGS TO IMPROVE:

Use vectorizaiton with numpy where loops are explictly used
Gaussian blur separation by using 1D kernels.
Multithreading to parallelize the computation of separate color channels.
Minimize unneccesary memory allocation by reusing arrays or preallocating memory where possible.
Optimize image expansion function for border handling
Use Cython or Numba
Perform benchmark and profiling using Python's timeit module to detect bottlenecks.
Rule of thumb: if K x K is the size of the kernel matrix, set Kernel size to be  K = 2 pi sigma
"""