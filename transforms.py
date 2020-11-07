import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import scipy
from scipy import ndimage, signal



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
    CIELAB_Zn = 108.88


    Sobel_Gx = np.array([
                        [[-1], [0], [1]],
                        [[-2], [0], [2]],
                        [[-1], [0], [1]]
    ])

    Sobel_Gy = np.array([
                        [[-1], [-2], [-1]],
                        [[0], [0], [0]],
                        [[1], [2], [1]]
    ])



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
    def cielab_L_component(image):
        image = tf.cast(image, tf.float32) / 255.0

        image = tf.transpose(image, [2, 0, 1])

        xyz = tf.tensordot(tf.cast(Transforms.XYZ_M, tf.float32), image, axes=((1), (0)))

        xyz = xyz * 100.0

        yd = xyz[1] / tf.cast(Transforms.CIELAB_Yn, tf.float32)

        temp = (1/3)*((29/6)**2) * yd + 4/29

        yf = tf.where(yd > (6/29)**3, tf.math.pow(yd, 1/3), temp)

        return 116.0 * yf - 16



    @staticmethod
    def rgb_to_cielab(pixel):
        pixel = np.array(pixel) / 255
        
        pixel = np.reshape(pixel, (3, 1))

        x, y, z = np.matmul(Transforms.XYZ_M, pixel).squeeze() * 100

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


    @staticmethod
    def rgb_to_gray(image):
        r, g, b = tf.split(image, num_or_size_splits=3, axis=-1)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray


    @staticmethod
    def sobel_filter(image, return_angles=False, return_dx_dy=False):    
        image = tf.cast(image, tf.float32)
        
        gray = Transforms.rgb_to_gray(image)

        gray = tf.expand_dims(gray, axis=0)

        gxf = tf.expand_dims(tf.constant(Transforms.Sobel_Gx, dtype=tf.float32), axis=2)
        gyf = tf.expand_dims(tf.constant(Transforms.Sobel_Gy, dtype=tf.float32), axis=2)

        gx = tf.nn.conv2d(gray, gxf, strides=[1, 1, 1, 1], padding='SAME')

        gy = tf.nn.conv2d(gray, gyf, strides=[1, 1, 1, 1], padding='SAME')

        if return_dx_dy:
            return gx.numpy().squeeze(), gy.numpy().squeeze()

        angles = tf.math.atan2(gy, gx)

        grad = tf.math.sqrt(tf.math.square(gx) + tf.math.square(gy))

        if return_angles:
            return tf.squeeze(tf.cast(grad, tf.uint8)), tf.squeeze(angles)

        return tf.squeeze(tf.cast(tf.repeat(grad, 3, axis=-1), tf.uint8))


    
    @staticmethod
    def local_features_detection(image, k=0.04, fertness=False):
        
        Ix, Iy = Transforms.sobel_filter(image, return_dx_dy=True)

        ixy = Ix * Iy
        ix2 = Ix * Ix
        iy2 = Iy * Iy

        ix2 = ndimage.gaussian_filter(ix2, sigma=2)
        iy2 = ndimage.gaussian_filter(iy2, sigma=2)
        ixy = ndimage.gaussian_filter(ixy, sigma=2)

        l, c = ixy.shape

        result = np.zeros((l, c))
        r = np.zeros((l, c))

        rmax = 0

        for i in range(l):
            for j in range(c):
                m = np.array([[ix2[i, j], ixy[i, j]], [ixy[i, j], iy2[i, j]]])
                detM = np.linalg.det(m)
                trM = np.trace(m)

                if fertness:
                    r[i, j] = detM / trM
                else:
                    r[i, j] = detM - k * np.power(trM, 2)

                if r[i, j] > rmax:
                    rmax = r[i, j]

        # non max suppresion
        for i in range(l - 1):
            for j in range(c - 1):
                if r[i, j] > 0.01 * rmax and r[i, j] > r[i - 1, j - 1] and r[i, j] > r[i - 1, j + 1] \
                        and r[i, j] > r[i + 1, j - 1] and r[i, j] > r[i + 1, j + 1]:
                    result[i, j] = 1

        result = Transforms.dilate(result[:, :, None], ksize=7)
        image = image.copy()

        # image[:, :, 0] = image[:, :, 0] + result * 40
        # image[:, :, 1] = image[:, :, 1] + result * 40

        image[:, :, 2][result > 0] = 245

        return image



    @staticmethod
    def canny_edge_detection(image, low_treshold=80, high_treshold=150):

        grad, angles = Transforms.sobel_filter(image, return_angles=True)

        M, N = grad.shape
        Z = tf.zeros_like(grad, dtype=tf.int32).numpy()

        angles = angles * 180. / np.pi

        angles = angles.numpy()
        grad = grad.numpy()
        angles[angles < 0] += 180

        for i in range(1, M-1):
            for j in range(1, N-1):
                q = 255
                r = 255
                #angle 0
                if (0 <= angles[i,j] < 22.5) or (157.5 <= angles[i,j] <= 180):
                    q = grad[i, j+1]
                    r = grad[i, j-1]
                #angle 45
                elif (22.5 <= angles[i,j] < 67.5):
                    q = grad[i+1, j-1]
                    r = grad[i-1, j+1]
                #angle 90
                elif (67.5 <= angles[i,j] < 112.5):
                    q = grad[i+1, j]
                    r = grad[i-1, j]
                #angle 135
                elif (112.5 <= angles[i,j] < 157.5):
                    q = grad[i-1, j-1]
                    r = grad[i+1, j+1]

                if (grad[i,j] >= q) and (grad[i,j] >= r):
                    Z[i,j] = grad[i,j]
                else:
                    Z[i,j] = 0

        
        Z = tf.constant(Z)

        high = tf.zeros_like(Z)
        middle = tf.zeros_like(Z)


        high = tf.where(Z > high_treshold, 255, high)
        middle_mask = tf.math.logical_and(Z < high_treshold , Z > low_treshold)
        middle = tf.where(middle_mask, 25, middle)

        double_treshold = high + middle

        kernel = tf.ones([3, 3, 1, 1], tf.int32)

        double_treshold = double_treshold[tf.newaxis, :, :, tf.newaxis]

        filt = tf.nn.conv2d(double_treshold, kernel, strides=[1, 1, 1, 1], padding="SAME")

        result = tf.zeros_like(double_treshold)

        result = tf.where(filt > 225, 255, result)

        return tf.squeeze(tf.repeat(tf.cast(result, tf.uint8), 3, axis=-1))



    @staticmethod
    def gabor_kernal(ksize, sigma, theta, lambd, gamma, psi):

        sigma_x = sigma
        sigma_y = sigma / gamma

        c, s = np.cos(theta), np.sin(theta)

        xmax = ksize[1] // 2
        ymax = ksize[0] // 2
        xmin = -xmax
        ymin = -ymax

        kernel = np.zeros([ymax - ymin + 1, xmax - xmin + 1])

        ex = -0.5 / sigma_x ** 2
        ey = -0.5 / sigma_y ** 2

        cscale = np.pi * 2 / lambd

        for iy, ix in np.ndindex(kernel.shape):
            xr = ix * c + iy * s
            yr = -ix * s + iy * c
            v = cscale * np.exp(ex * xr * xr + ey * yr * yr) * np.cos(cscale * xr + psi)
            kernel[iy, ix] = v

        return kernel


    @staticmethod
    def scale_min_max(image):

        min_v = tf.math.reduce_min(image)
        max_v = tf.math.reduce_max(image)

        return (image - min_v) / (max_v - min_v)

    @staticmethod
    def gabor_filter(image, theta):
        
        image = tf.cast(image, tf.float32)

        gray = Transforms.rgb_to_gray(image)
        gray = tf.expand_dims(gray, axis=0)


        gabor_kernel = Transforms.gabor_kernal(ksize=(7, 7), sigma=0.56 * 2,
                        theta=theta, lambd=2, gamma=0.1, psi=0)
                
        gabor_kernel = gabor_kernel[:, :, tf.newaxis, tf.newaxis]

        filt = tf.nn.conv2d(gray, gabor_kernel, strides=[1, 1, 1, 1], padding="SAME")

        filt = tf.squeeze(tf.repeat(filt, 3, axis=-1))

        filt = Transforms.scale_min_max(filt)

        print(tf.math.reduce_min(filt), tf.math.reduce_max(filt))

        return filt


    
    @staticmethod
    def gaussian_kernel(size: int, std: float):
         
        d = tfp.distributions.Normal(0.0, std)

        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

        gauss_kernel = tf.einsum('i,j->ij', vals, vals)

        return gauss_kernel / tf.reduce_sum(gauss_kernel)




    @staticmethod
    def gaussian_filter(image, std, size):

        image = tf.cast(image, tf.float32)

        image = tf.expand_dims(image, axis=0)

        r, g, b = tf.split(image, num_or_size_splits=3, axis=-1)

        gauss_kernel = Transforms.gaussian_kernel((size - 1)//2, std)

        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

        r = tf.nn.conv2d(r, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
        g = tf.nn.conv2d(g, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
        b = tf.nn.conv2d(b, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")

        image = tf.concat([r, g, b], axis=-1)

        return tf.squeeze(tf.cast(image, tf.uint8))


    # TODO fix
    @staticmethod
    def dilate(binary_image, ksize=3):

        binary_image = tf.cast(binary_image, dtype=tf.float32)

        kernel = tf.ones((ksize, ksize), dtype=tf.float32)

        kernel = kernel[:, :, tf.newaxis, tf.newaxis]

        binary_image = tf.expand_dims(binary_image, axis=0)

        tf.print(tf.shape(binary_image))

        conv_image = tf.nn.conv2d(binary_image, kernel, strides=[1, 1, 1, 1], padding="SAME")

        dilate_image = tf.where(conv_image >= 1, tf.ones_like(binary_image), 0)

        return tf.squeeze(dilate_image)

    @staticmethod
    def erode(binary_image, ksize=3):

        binary_image = tf.cast(binary_image, tf.float32)

        kernel = tf.ones((ksize, ksize), dtype=tf.float32)

        kernel = kernel[:, :, tf.newaxis, tf.newaxis]

        binary_image = tf.expand_dims(binary_image, axis=0)

        tf.print(tf.shape(binary_image))

        conv_image = tf.nn.conv2d(binary_image, kernel, strides=[1, 1, 1, 1], padding="SAME")

        erode_image = tf.where(conv_image == ksize * ksize, tf.ones_like(binary_image), 0)

        return tf.squeeze(erode_image)

    

    @staticmethod 
    def components_count(image, dilate_ksize=3, erode_ksize=3):
        # TODO remove small cells, select params

        binary_3c = Transforms.otsu_binarization(image) // 255

        binary = tf.slice(binary_3c, [0, 0, 0], [-1, -1, 1])

        binary = tf.cast(binary, dtype=tf.float32)

        binary = tf.math.logical_not(tf.cast(tf.cast(binary, tf.int32), tf.bool))
        
        # binary = tf.pad(binary, [[32, 32], [32, 32], [0, 0]], "CONSTANT")
        
        binary = Transforms.dilate(binary, ksize=3)

        binary = binary[:, :, tf.newaxis]

        binary = Transforms.erode(binary, ksize=7)

        binary = tf.cast(binary, tf.float32).numpy()

        M, N = binary.shape

        km = 0
        kn = 0
        cur = 1
        
        for i in range(1, M-1):
            for j in range(1, N-1):

                kn = j - 1

                if kn <= 0:
                    kn = 1
                    B = 0
                else:
                    B = binary[i, kn]

                km = i - 1
                if km <= 0:
                    km = 1
                    C = 0
                else:
                    C = binary[km, j]
                
                A = binary[i, j]

                if A == 0:
                    pass
                elif B == 0 and C == 0:
                    cur  = cur + 1
                    binary[i, j] = cur
                elif B != 0 and C == 0:
                    binary[i, j] = B
                elif B == 0 and C != 0:
                    binary[i, j] = C
                elif B != 0 and C != 0:
                    if B == C:
                        binary[i, j] = B
                    else:
                        binary[i, j] = B

                        binary[binary == C] = B  


        u_values = np.unique(binary)

        for i, v in enumerate(u_values):
            binary[binary == v] = i * 5
                  
        img = np.repeat(binary[..., None], 3, axis=-1)

        max_v = img.max()

        img[:, :, 1] = np.mod((img[:, :, 0] + 140), max_v)

        return len(u_values), tf.cast(img * 255, tf.uint8)
        # return 1, binary

    
    @staticmethod
    def otsu_binarization(image):

        image = tf.cast(image, tf.float32)

        gray = Transforms.rgb_to_gray(image)

        gray = tf.cast(gray, tf.int32)

        min_v = tf.math.reduce_min(gray)
        max_v = tf.math.reduce_max(gray)

        hist_size = max_v - min_v + 1
        hist = tf.histogram_fixed_width(
                gray, [min_v, max_v], nbins=hist_size, dtype=tf.int32, name=None
        )

        ind = tf.range(hist_size, dtype=tf.float32)
        hist = tf.cast(hist, dtype=tf.float32)

        n = tf.math.reduce_sum(hist)
        m = tf.math.reduce_sum(hist * ind)

        alpha1 = tf.math.cumsum(hist * ind) 
        beta1 = tf.math.cumsum(hist)

        w1 = beta1 / n
        a = (alpha1 / beta1) - (tf.repeat(m, hist_size) - alpha1) / (tf.repeat(n, hist_size) - beta1)
        sigma = w1 * (1 - w1) * a * a

        treshold = min_v + tf.cast(tf.math.argmax(sigma), tf.int32)

        r = tf.cast(gray > treshold, tf.uint8) * tf.constant(255, dtype=tf.uint8)

        return tf.repeat(r, 3, axis=-1)


if __name__ == '__main__':

    image = np.array(Image.open('res\cells.jpg'))

    image = image[:,:,:3]  

    count, im_tr = Transforms.components_count(image, dilate_ksize=3, erode_ksize=5)

    fig, ax = plt.subplots(1, 2, figsize=(14, 14))

    ax[0].imshow(image)

    ax[1].imshow(im_tr)
    plt.show()