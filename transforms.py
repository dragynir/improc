import tensorflow as tf
import numpy as np




class Transforms(object):
    def __init__(self):
        pass



    @staticmethod
    def transform_hsv(image, h, s, v):
        '''

            h - hue shift angle in degrees
            s - scalar from 1.0
            v - scalar from 1.0 
        '''

        tf.split(image, [], [])



image = np.array(Image.open('res\\niceimage.jpg'))

Transforms.transform