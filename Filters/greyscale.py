import numpy as np
from PIL import Image

image = np.asarray(Image.open("../Test Image.png"))
grey_image = np.zeros_like(image)

red_pixels = image[::1,::1,0:1:1] * 0.21
print(red_pixels.shape)
green_pixels = image[::1,::1,1:2:1] * 0.72
blue_pixels = image[::1,::1,2:3:1] * 0.07

# Add all values together
weighted_sum = red_pixels + green_pixels + blue_pixels
weighted_sum = weighted_sum.astype("uint8")

# Add weighted_sum to all values in gray_image
grey_image += weighted_sum
Image.Image.show(Image.fromarray(grey_image))



