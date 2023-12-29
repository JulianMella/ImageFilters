from PIL import Image
import numpy as np
import math
import sys
from rgb2hsv import rgb2hsv



def hsv2rgb(image : np.array) -> np.array:
    RGB_image = np.copy(image)

    for row in range(RGB_image.shape[0]):
        for column in range(RGB_image.shape[1]):
            arr = RGB_image[row : row + 1, column : column + 1, 0:3]
            print(arr)
            H = arr[:,:,0:1]
            S = arr[:,:,1:2]
            V = arr[:,:,2:3]

            h = H / 60
            i = math.floor(h)
            f = h - i
            p = V * (1 - S)
            q = V * (1 - S * f)
            t = V * (1 - S * (1 - f))

            match i:
                case 0:
                    R = V
                    G = t
                    B = p
                case 1:
                    R = q
                    G = V
                    B = p
                case 2:
                    R = p
                    G = V
                    B = t
                case 3:
                    R = p
                    G = q
                    B = V
                case 4:
                    R = t
                    G = p
                    B = V
                case 5:
                    R = V
                    G = p
                    B = q

            R *= 255
            R = R[0][0][0]
            G *= 255
            G = G[0][0][0]
            B *= 255
            B = B[0][0][0]

            RGB = np.array([R, G, B])

            RGB_image[row : row + 1, column : column + 1] = RGB

    return RGB_image.astype("uint8")


if __name__ == "__main__":
    image = np.asarray(Image.open(sys.path[0] + '/../Test Image 2.png'))
    HSV_image = rgb2hsv(image)
    RGB_image = hsv2rgb(HSV_image)
    Image.fromarray(RGB_image).show()