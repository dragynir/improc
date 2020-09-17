import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt




class Transforms(object):
    def __init__(self):
        pass


    XYZ_M = np.array([
                        [0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]
                        ])
    CIELAB_Xn = 95.04
    CIELAB_Yn = 100.0
    CIELAB_Zn = 108.8


    @staticmethod
    def transform_hsv(image, h, s, v):
        '''
            http://beesbuzz.biz/code/16-hsv-color-transforms
            h - hue shift angle in degrees
            s - scalar from 1.0
            v - scalar from 1.0 
        '''
        image = tf.cast(image, tf.float32)
        r, g, b = tf.split(image, num_or_size_splits=3, axis=-1)

        # r * 0.299 + g * 0.587 + b * 0.114
        # r * 0.2126 + g * 0.7152 + b * 0.0722 

        vsu = v*s*tf.math.cos(h * np.pi / 180.0)
        vsw = v*s*tf.math.sin(h * np.pi / 180.0)

        nr = (0.299*v + 0.701*vsu + 0.168*vsw) * r + \
            (0.587*v - 0.587*vsu + 0.330*vsw) * g + \
            (0.114*v - 0.114*vsu - 0.497*vsw) * b

        ng = (0.299*v - 0.299*vsu - 0.328*vsw) * r + \
            (0.587*v + 0.413*vsu + 0.035*vsw) * g + \
            (0.114*v - 0.114*vsu + 0.292*vsw) * b

        nb = (0.299*v - 0.300*vsu + 1.25*vsw) * r + \
            (0.587*v - 0.588*vsu - 1.05*vsw) * g + \
            (0.114*v + 0.886*vsu - 0.203*vsw) * b

        image = tf.concat([nr, ng, nb], axis=-1)
        image = tf.clip_by_value(image, 0.0, 255.0)

        # 

        return tf.cast(image, tf.uint32)

    
    @staticmethod
    def rgb_image_to_cielab_image(image):
        image = tf.cast(image, tf.float32)
        xyz = tf.linalg.matmul(image, tf.constant(Transforms.XYZ_M))

    
    @staticmethod
    def cielab_f(x):
        if x > (6/29)**3:
            return x**(1/3)
        return (1/3)*((29/6)**2) * x + 4/29

    @staticmethod
    def rgb_to_cielab(pixel):
        pixel = np.reshape(np.array(pixel), (3, 1))

        x, y, z = np.matmul(Transforms.XYZ_M, pixel).squeeze()

        L = 116.0 * Transforms.cielab_f(y/Transforms.CIELAB_Yn) - 16

        a = 500.0 * (Transforms.cielab_f(x/Transforms.CIELAB_Xn) - \
                    Transforms.cielab_f(y/Transforms.CIELAB_Yn))

        b = 200.0 * (Transforms.cielab_f(y/Transforms.CIELAB_Yn) - \
                    Transforms.cielab_f(z/Transforms.CIELAB_Zn))
        return int(L), round(a, 3), round(b, 3)




    @staticmethod
    def rgb_to_hsv(pixel):
        pixel = np.array(pixel) / 255
        r, g, b = pixel

        cmin = np.min(pixel)
        cmax = np.max(pixel)
        diff = cmax - cmin
        
        hue, saturation = -1, -1
        if diff == 0:
            hue = 0
        elif cmax == r:
            hue = (60 * ((g - b) / diff) + 360) % 360
        elif cmax == g:
            hue = (60 * ((b - r) / diff) + 120) % 360
        elif cmax == b:
            hue = (60 * ((r - g) / diff) + 240) % 360
        
        if cmax == 0: 
            saturation = 0; 
        else:
            saturation = (diff / cmax) * 100

        value = cmax * 100

        return int(hue), int(saturation), int(value)       


if __name__ == '__main__':
    image = np.array(Image.open('res\\niceimage.jpg'))
    im_tr = Transforms.transform_hsv(image, 0, 1.3, 1.5)


    fig, ax = plt.subplots(1, 2, figsize=(14, 14))
    ax[0].imshow(image)
    ax[1].imshow(im_tr.numpy())
    plt.show()