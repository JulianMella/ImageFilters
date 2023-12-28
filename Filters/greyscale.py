import numpy as np
from PIL import Image

def greyscale_filter(image: np.array) -> np.array:
    """Convert rgb pixel array to grayscale

    Args:
        image (np.array)
    Returns:
        np.array: gray_image
    """

    gray_image = np.zeros_like(image)

    # Multiply all pixels by their corresponding values
    red_pixels = image[::1,::1,0:1:1] * 0.21
    green_pixels = image[::1,::1,1:2:1] * 0.72
    blue_pixels = image[::1,::1,2:3:1] * 0.07

    # Add all values together
    weighted_sum = red_pixels + green_pixels + blue_pixels
    weighted_sum = weighted_sum.astype("uint8")

    # Add weighted_sum to all values in gray_image
    gray_image += weighted_sum

    return gray_image


if __name__ == "__main__":
    image = np.asarray(Image.open("../Test Image.png"))
    grey_image = greyscale_filter(image)
    Image.fromarray(grey_image).show()

