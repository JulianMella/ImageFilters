from gaussianblur import gaussian_blur, gaussian_kernel_matrix
from expandimage import expand_image
from greyscale import greyscale_filter
from PIL import Image
import numpy as np
import math

def sobel_convolution(image : np.array, colored : bool) -> np.array:
    grey_image = image

    if colored:
        grey_image = greyscale_filter(image)


    #grey_image_blurred = gaussian_blur(np.zeros_like(image), grey_image, gaussian_kernel_matrix(1, 3), 3)
    expanded_image = expand_image(grey_image, 3)

    horizontal_kernel = np.reshape(np.array([[1.0 ,2.0 ,1.0 ], [0.0 ,0.0 ,0.0], [-1.0, -2.0 , -1.0]]), (3,3,1))
    vertical_kernel = np.reshape(np.array([[1.0, 0.0, -1.0], [2.0 ,0.0 ,-2.0], [1.0 ,0.0 ,-1.0]]), (3,3,1))
    sobel_filtered_image = np.zeros_like(image)


    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            neighbourhood_values = expanded_image[row : row + 3, column : column + 3, 0:1]
            gx = np.sum(np.multiply(neighbourhood_values, horizontal_kernel))
            gy = np.sum(np.multiply(neighbourhood_values, vertical_kernel))
            sobel_filtered_image[row : row + 1, column : column + 1] = np.sqrt(gx ** 2 + gy ** 2)

    if sobel_filtered_image.shape[2] == 4:
        sobel_filtered_image[:,:,3:4] = 255
    Image.fromarray(sobel_filtered_image).show()
    return image
"""
def magnitude_of_edge(vertical : np.array, horizontal : np.array) -> np.array:
    magnitude = np.zeros_like(vertical)

    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            magnitude_value =
    for row in range(0, magnitude.shape[1]):  # Rows
        for column in range(0, magnitude.shape[0]):  # Columns
            magnitude_value = math.sqrt((vertical[column][row][0] ** 2) + (horizontal[column][row][0] ** 2))
            magnitude[column][row][0] = magnitude_value
            magnitude[column][row][1] = magnitude_value
            magnitude[column][row][2] = magnitude_value

    if magnitude.shape[2] == 4:
        magnitude[:,:,3:4] = 255


    #for row in range(0, magnitude.shape[1]):  # Rows
    #    for column in range(0, magnitude.shape[0]):  # Columns

    #        magnitude[column][row][0] = magnitude[column][row][0] * 0.5
     #       magnitude[column][row][1] = magnitude[column][row][0] * 0.5
     #       magnitude[column][row][2] = magnitude[column][row][0] * 0.5

    magnitude = magnitude.astype("uint8")
    Image.fromarray(magnitude).show()
"""
def sobel_edge_detector(image : np.array):
    sobel_convolution(image, True)
    #testh = sobel_convolution(np.zeros_like(image), expand_image(blurred, 3), horizontal_kernel, 3)



    #testv = sobel_convolution(np.zeros_like(image), expand_image(blurred, 3), vertical_kernel, 3)



    #magnitude_of_edge(testv, testh)


if __name__ == "__main__":
    image = np.asarray(Image.open("../Valve.png"))
    sobel_edge_detector(image)
    #test = np.array([[[ 157, 0, -157],[348,0,-348],[191,0,-191]],[[ 155,    0, -155],[ 346,0, -346],[ 191,0, -191]],[[ 154,    0, -154],[ 344,0, -344],[ 190,0,-190]]])
    #print(np.sum(test))
    #Image.fromarray(test).show()