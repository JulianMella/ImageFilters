from gaussianblur import gaussian_blur, gaussian_kernel_matrix
from expandimage import expand_image
from greyscale import greyscale_filter
from PIL import Image
import numpy as np
import math

def sobel_convolution(image : np.array, expanded_image : np.array, kernel_matrix : np.array, kernel_matrix_size : int ) -> np.array:
    for row in range(0, image.shape[1]):  # Rows
        for column in range(0, image.shape[0]):  # Columns
            # Define the neighborhood region around the current pixel
            neighborhood_red = expanded_image[column : column + kernel_matrix_size, row : row + kernel_matrix_size, 0:1]
            neighborhood_green = expanded_image[column : column + kernel_matrix_size, row : row + kernel_matrix_size, 1:2]
            neighborhood_blue = expanded_image[column : column + kernel_matrix_size, row : row + kernel_matrix_size, 2:3]

            # Perform element-wise multiplication with the kernel matrix
            red_sum = abs(np.sum(neighborhood_red * kernel_matrix))
            green_sum = abs(np.sum(neighborhood_green * kernel_matrix))
            blue_sum = abs(np.sum(neighborhood_blue * kernel_matrix))

            if row == 100 and column == 100:
                print(red_sum)
                print(green_sum)
                print(blue_sum)

            # Assign the summed values to the corresponding pixel channels after normalization if needed
            image[column][row][0] = red_sum
            image[column][row][1] = green_sum
            image[column][row][2] = blue_sum


    #Set transparency level of PNG image to max
    if image.shape[2] == 4:
        image[:,:,3:4:1] = 255



    return image.astype("uint8")

def magnitude_of_edge(vertical : np.array, horizontal : np.array) -> np.array:
    magnitude = np.zeros_like(vertical)

    for row in range(0, magnitude.shape[1]-1):  # Rows
        for column in range(0, magnitude.shape[0]-1):  # Columns
            magnitude_value = int(math.sqrt((vertical[column][row][0] ** 2) + (horizontal[column][row][0])))
            magnitude[column][row][0] = magnitude_value
            magnitude[column][row][1] = magnitude_value
            magnitude[column][row][2] = magnitude_value

    if magnitude.shape[2] == 4:
        magnitude[:,:,3:4:1] = 255

    Image.fromarray(magnitude).show()

    min_val = np.min(magnitude)
    max_val = np.max(magnitude)

    print("Minimum value:", min_val)
    print("Maximum value:", max_val)


def sobel_edge_detector(image : np.array):
    grey_image = greyscale_filter(image)

    expanded_image = expand_image(grey_image, 3)
    blurred = gaussian_blur(np.zeros_like(image), expanded_image, gaussian_kernel_matrix(1, 3), 3)
    vertical_kernel = np.array([[1,0,-1],
                                [2,0,-2],
                                [1,0,-1]])
    horizontal_kernel = np.array([[1,2,1],
                                  [0,0,0],
                                  [-1,-2,-1]])
    vertical_kernel = np.reshape(vertical_kernel,(3,3,1))
    horizontal_kernel = np.reshape(horizontal_kernel,(3,3,1))

    testh = sobel_convolution(np.zeros_like(image), expand_image(blurred, 3), horizontal_kernel, 3)

    Image.fromarray(testh).show()

    testv = sobel_convolution(np.zeros_like(image), expand_image(blurred, 3), vertical_kernel, 3)

    Image.fromarray(testv).show()

    magnitude_of_edge(testv, testh)


if __name__ == "__main__":
    image = np.asarray(Image.open("../Test Image.png"))
    sobel_edge_detector(image)
    #test = np.array([[[ 157, 0, -157],[348,0,-348],[191,0,-191]],[[ 155,    0, -155],[ 346,0, -346],[ 191,0, -191]],[[ 154,    0, -154],[ 344,0, -344],[ 190,0,-190]]])
    #print(np.sum(test))
    #Image.fromarray(test).show()