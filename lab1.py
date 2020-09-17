import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import jit

from PyQt5.QtWidgets import QListWidget



class ImageViewer(object):
    def __init__(self, image, ax):
        self.rgb_image = image.get_array().astype(np.float32)
        self.ax = ax
        self.norm_rgb = self.rgb_image / 255
        self.__prepare_hsv_image()
        self.__prepare_xyz_image()
        self.__prepare_cieLab()
        self.__show_hist()


    def __prepare_hsv_image(self):
        self.hsv_image = np.apply_along_axis(
                self.__pixel_to_hsv, -1, self.norm_rgb)

    def __prepare_xyz_image(self):
        '''
            X, Y and Z output refer to a D65/2° standard illuminant.
        '''
        self.xyz_image = np.apply_along_axis(
                self.__pixel_to_xyz, -1, self.norm_rgb)

    def __prepare_cieLab(self):
        self.cie_Lab_image = np.apply_along_axis(
                self.__pixel_to_cieLab, -1, self.xyz_image)


    def __show_hist(self):
        ax[1].hist(self.cie_Lab_image[:,:,0].flatten(), bins=100)

    
    def __pixel_to_cieLab(self, pixel):

        '''
            D65	95.047	100.000	108.883	94.811	100.000	107.304	Daylight, sRGB, Adobe-RGB
        '''

        var_X, var_Y, var_Z = pixel
        Reference_X, Reference_Y, Reference_Z = (100.0, 108.883, 94.811)

        var_X = var_X / Reference_X
        var_Y = var_Y / Reference_Y
        var_Z = var_Z / Reference_Z

        if var_X > 0.008856:
            var_X = var_X ** (1/3)
        else:
            var_X = (7.787 * var_X) + (16 / 116)

        if var_Y > 0.008856:
            var_Y = var_Y ** (1/3)
        else:        
            var_Y = (7.787 * var_Y) + (16 / 116)

        if var_Z > 0.008856:
            var_Z = var_Z ** (1/3)
        else:
            var_Z = (7.787 * var_Z) + (16 / 116)

        CIE_L = (116 * var_Y) - 16
        CIE_a = 500 * (var_X - var_Y)
        CIE_b = 200 * (var_Y - var_Z)

        return CIE_L, CIE_a, CIE_b

    def __pixel_to_xyz(self, pixel):
        var_R, var_G, var_B = pixel

        if var_R > 0.04045:
            var_R = ((var_R + 0.055) / 1.055) ** 2.4
        else:
            var_R = var_R / 12.92

        if var_G > 0.04045:
            var_G = ((var_G + 0.055) / 1.055) ** 2.4
        else:       
            var_G = var_G / 12.92

        
        if var_B > 0.04045:
            var_B = ((var_B + 0.055) / 1.055) ** 2.4
        else:
            var_B = var_B / 12.92

        var_R = var_R * 100
        var_G = var_G * 100
        var_B = var_B * 100

        X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
        Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
        Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

        return X, Y, Z


    def __pixel_to_hsv(self, pixel):

        r, g, b = pixel
        
        cmin = np.min(pixel)
        cmax = np.max(pixel)
        diff = cmax - cmin

        
        hue, saturation = -1, -1
        if diff == 0:
            hue = 0
        elif cmax == r:
            hue = (60 * ((g - b) / diff))
        elif cmax == g:
            hue = (60 * ((b - r) / diff) + 2)
        elif cmax == b:
            hue = (60 * ((r - g) / diff) + 4)
        
        if hue < 0:
            hue+=360

        if cmax == 0: 
            saturation = 0; 
        else:
            saturation = (diff / cmax) * 100

        value = cmax * 100

        return hue, saturation, value            

    def __call__(self, x, y):
        r, g, b = map(int, self.rgb_image[int(y), int(x)])
        h, s, v = map(int, self.hsv_image[int(y), int(x)])        
        return f'r={r}, g={g}, b={b} : h={h}, s={s}, v={v}'



image = np.array(Image.open('frimage.png'))
image = image[:,:,:3]

# image = np.array(Image.open('niceimage.jpg'))


print(image.shape)
fig, ax = plt.subplots(1, 2, figsize=(14, 14))
im = ax[0].imshow(image, interpolation='none')
ax[0].format_coord = ImageViewer(im, ax)
plt.title('CieLab L hist')
plt.show()