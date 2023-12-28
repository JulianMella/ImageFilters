
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
def gaussian_blur(image: np.array, expanded_image: np.array, kernel_matrix: np.array, kernel_matrix_size: int) -> np.array:

    #Set transparency level of PNG image to max
    if image.shape[2] == 4:
        image[:,:,3:4:1] = 255

    for row in range(0, image.shape[1]):  # Rows
        for column in range(0, image.shape[0]):  # Columns
            # Define the neighborhood region around the current pixel
            neighborhood_red = expanded_image[column : column + kernel_matrix_size, row : row + kernel_matrix_size, 0:1]
            neighborhood_green = expanded_image[column : column + kernel_matrix_size, row : row + kernel_matrix_size, 1:2]
            neighborhood_blue = expanded_image[column : column + kernel_matrix_size, row : row + kernel_matrix_size, 2:3]

            # Perform element-wise multiplication with the kernel matrix
            red_sum = np.sum(neighborhood_red * kernel_matrix)
            green_sum = np.sum(neighborhood_green * kernel_matrix)
            blue_sum = np.sum(neighborhood_blue * kernel_matrix)

            if row == 100 and column == 100:
                print(red_sum)
                print(green_sum)
                print(blue_sum)

            # Assign the summed values to the corresponding pixel channels after normalization if needed
            image[column][row][0] = red_sum
            image[column][row][1] = green_sum
            image[column][row][2] = blue_sum

    return image.astype("uint8")

def gaussian_filter(image: np.array, radius: int) -> np.array:
    kernel_width = (2 * radius) + 1

    kernel_matrix = gaussian_kernel_matrix(radius, kernel_width)
    expanded_image = expand_image(image, kernel_width)

    blurred_image = gaussian_blur(np.zeros_like(image), expanded_image, kernel_matrix, kernel_width)
    return blurred_image


if __name__ == "__main__":
    image = np.asarray(Image.open(sys.path[0] + '/../Test Image.png'))
    Image.fromarray(image).show()
    Image.fromarray(gaussian_filter(image, 15)).show()



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