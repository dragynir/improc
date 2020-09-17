import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt




class Transforms(object):
    def __init__(self):
        pass



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

        return tf.cast(image, tf.uint32)


if __name__ == '__main__':
    image = np.array(Image.open('res\\niceimage.jpg'))
    im_tr = Transforms.transform_hsv(image, 0, 1.3, 1.5)


    fig, ax = plt.subplots(1, 2, figsize=(14, 14))
    ax[0].imshow(image)
    ax[1].imshow(im_tr.numpy())
    plt.show()