import numpy as np

# Expand image in accordance to the kernel matrix size.
def expand_image(image: np.array, kernel_width: int) -> np.array:
    expansion_length = kernel_width
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