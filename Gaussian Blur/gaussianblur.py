import math
import numpy as np
from PIL import Image
import sys


#Calculate the kernel matrix
def calculate_kernel_matrix(kernel_matrix_size: int, stdev: float) -> list:
    matrix = []

    #Gaussian function coefficient
    coefficient = 1 / (2 * math.pi * (stdev ** 2))

    #Top left coordinate of kernel matrix based off of the fact the center of the matrix has coordinates 0, 0 in mathematical environment. Necessary for gaussian function
    x = -(kernel_matrix_size // 2)
    y = -(kernel_matrix_size // 2)

    for _ in range(kernel_matrix_size):
        matrix.append([0] * kernel_matrix_size)

    for row in range(len(matrix)):
        for col in range(kernel_matrix_size):
            matrix[row][col] = coefficient * math.exp(-(((x ** 2) + (y ** 2)) / (2 * (stdev ** 2))))
            x += 1
        y += 1
        x = -(kernel_matrix_size // 2)

    return matrix

# Expand image in accordance to the kernel matrix size.
def expand_image(image: np.array, kernel_matrix_size: int) -> np.array:
    expansion_length = kernel_matrix_size // 2
    padded_image = image
    width = image.shape[0]
    height = image.shape[1]

    left_wall = np.fliplr(image[0 : , 0 : expansion_length])
    padded_image = np.concatenate((left_wall, padded_image), axis=1)

    right_wall = np.fliplr(image[0 : , width - expansion_length : width])
    padded_image = np.concatenate((padded_image, right_wall), axis=1)

    top_left_corner = np.fliplr(image[0 : expansion_length, 0 : expansion_length])
    top_right_corner = np.fliplr(image[0 : expansion_length, width - expansion_length : width])
    top_row = np.concatenate((np.concatenate((top_left_corner, image[0 : expansion_length, 0 : ]), axis=1), top_right_corner), axis=1)
    top_row = np.flipud(top_row)

    padded_image = np.concatenate((top_row, padded_image), axis=0)

    bottom_left_corner = np.fliplr(image[height - expansion_length : height, 0 : expansion_length])
    bottom_right_corner = np.fliplr(image[height - expansion_length : height, width - expansion_length : width])
    bottom_row = np.concatenate((np.concatenate((bottom_left_corner, image[height - expansion_length : height, 0 : ]), axis=1), bottom_right_corner), axis=1)

    padded_image = np.concatenate((padded_image, bottom_row), axis=0)
    print(padded_image)
    return padded_image

#Place the center of the kernel at each pixel of the image
#Multiply the kernel matrix values with the corresponding pixel values in the local neighbourhood
#Sum up these products to obtain the new value for the pixel
def gaussian_blur(image: np.array, expanded_image: np.array, kernel_matrix: list, kernel_matrix_size: int) -> np.array:
    red_matrix = expanded_image[0 : kernel_matrix_size, 0 : kernel_matrix_size]

    Image.fromarray(red_matrix).show()
    Image.fromarray(expanded_image).show()
    Image.fromarray(image[0:1,0:1]).show()
    Image.fromarray(image).show()
    print(red_matrix)
    """
    for row in range(image.shape[1] - 1):
        for column in range(image.shape[0] - 1):
            red = image[row][column][0]
            red_matrix = expanded_image[column : kernel_matrix_size, row : kernel_matrix_size, 0:1]


            green = image[row][column][1]
            green_matrix = expanded_image[column : kernel_matrix_size, row : kernel_matrix_size, 0:1]


            blue = image[row][column][2]
            blue_matrix = expanded_image[column : kernel_matrix_size, row : kernel_matrix_size, 0:1]

    """



    return 0

def gaussian_filter(image: np.array, kernel_matrix_size: int, stdev: float) -> np.array:
    kernel_matrix = calculate_kernel_matrix(kernel_matrix_size, stdev)
    expanded_image = expand_image(image, kernel_matrix_size)
    blurred_image = gaussian_blur(image, expanded_image, kernel_matrix, kernel_matrix_size)

    return blurred_image



# Perform

#Rule of thumb, if K x K is the size of the kernel matrix, set Kernel size to be  K = 2 pi sigma

# A Separable Gaussian function is more efficient.
if __name__ == "__main__":
    image = np.asarray(Image.open(sys.path[0] + '/../Test Image.png'))
    gaussian_filter(image, 3, 1)