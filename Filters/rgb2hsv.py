from PIL import Image
import numpy as np
import math
import sys



def rgb2hsv(image : np.array) -> np.array:
    HSV_image = np.copy(image[:,:,0:3]).astype("float32")

    for row in range(HSV_image.shape[0]):
        for column in range(HSV_image.shape[1]):
            arr = HSV_image[row : row + 1, column : column + 1, 0:3]
            Cmax = np.max(arr)
            Cmin = np.min(arr)
            delta = Cmax - Cmin

            R = arr[:,:,0:1]
            G = arr[:,:,1:2]
            B = arr[:,:,2:3]

            if delta == 0.0:
                H = 0.0
            else:
                if Cmax == R:
                    H = (((G - B) / delta) % 6.0)

                elif Cmax == G:
                    H = (((B - R) / delta) + 2.0)

                elif Cmax == B:
                    H = (((R - G) / delta) + 4.0)

                H *= 60
                H = H[0][0][0]
            if Cmax > 0.0:
                S = delta / Cmax
            else:
                S = 0.0
            V = Cmax

            HSV = np.array([H, S, V])

            HSV_image[row : row + 1, column : column + 1] = HSV

    return HSV_image


if __name__ == "__main__":
    image = np.asarray(Image.open(sys.path[0] + '/../Test Image.png'))
    HSV_image = rgb2hsv(image)