import math
import numpy as np
from PIL import Image

#Calculate the kernel matrix


def calculate_kernel_matrix(kernel_matrix_size: int, stdev: int) -> list:
    matrix = []

    #Gaussian function coefficient
    coefficient = 1 / (2 * math.pi * (stdev ** 2))

    #Top left coordinate of kernel matrix based off of the fact the center of the matrix has coordinates 0, 0 in mathematical environment. Necessary for gaussian function
    x = -(kernel_matrix_size // 2)
    y = -(kernel_matrix_size // 2)

    for _ in range(kernel_matrix_size):
        matrix.append([])

    for row in matrix:
        for col in range(kernel_matrix_size):
            matrix[row][col] = coefficient * math.exp(-(((x ** 2) + (y ** 2)) / (2 * (stdev ** 2))))
            x += 1
        y += 1

    return matrix

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

    return padded_image

def gaussian_filter(image: np.array, kernel_matrix_size: int, stdev: int) -> np.array:
    kernel_matrix = calculate_kernel_matrix(kernel_matrix_size, stdev)

    gaussian_blurred_image = np.zeros_like(image)

    return gaussian_blurred_image


# Expand image in accordance to the kernel matrix size.

# Perform


if __name__ == "__main__":
    gaussian_filter("../Test Image.png", 3, 0)