from PIL import Image
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import matplotlib.pyplot as plt
import skimage.color
import skimage.filters
import skimage.util
import skimage.segmentation

"""
'19. 1. 22 T.Kato
!! CAUTION !! This code need scikit-Image beta (0.15β) 
"""
# >================ Function ================<
def reverse(img, rev=True):
    """
    0 - 255 converter
    :param img: Original Image
    :param rev: Reverse Flag. If rev is False, This code will do nothing.
    :return:reverse_img
    """
    w = img.shape[0]
    h = img.shape[1]
    R = 0
    Z = 255
    out = np.zeros((img.shape[0], img.shape[1]))
    if rev is False:
        R = 255
        Z = 0

    for i in range(w):
        for j in range(h):
            if img[i][j] != R:
                out[i][j] = R
            else:
                out[i][j] = Z
    return out


def easy_f_fill(img):
    """
    easy_flood_fill changer(because be proposed BB Box by SSD)
    :param img: Original Image (Superpixel image)
    :return: Background Region black out
    """
    # Upper side, Left 2 right
    # print(img.shape[0], img.shape[1])
    for i in range(0, img.shape[1], 20):
        cv2.floodFill(CV_out, mask, (i, 1), 255)
    # Left side, Up 2 down
    for j in range(0, img.shape[0], 20):
        cv2.floodFill(CV_out, mask, (1, j), 255)
    # Right side, Up 2 down
    for k in range(0, img.shape[0], 20):
        cv2.floodFill(CV_out, mask, (img.shape[1] - 1, k), 255)


def mask_image(img):
    """
    mask for output image. (masking area is not blue(255, 0, 0))
    :param img:
    :return:
    """
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != 29:  # Blue(255, 0, 0)のgray値
                img[i][j] = 255
            else:
                img[i][j] = 0

    return img


def filter_3(img):
    """
    Remove JUGGY EDGE by 3x3 kernel
    :param img: img, gray, np.array ((h, w, 1), dtype=np.uint8)
    :return: Remove JUGGY EDGE mat, ((h, w, 1), dtype=np.uint8)
    """
    ret_img = np.zeros((img.shape[0], img.shape[1]))
    kernel = np.zeros((3, 3))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            # window function
            Omega = np.array([[img[i - 1][j - 1], img[i][j - 1], img[i + 1][j - 1]],
                              [img[i - 1][j], img[i][j], img[i + 1][j]],
                              [img[i - 1][j + 1], img[i][j + 1], img[i + 1][j + 1]]])
            cnt = np.sum(Omega == kernel)
            if cnt >= 4:
                ret_img[i][j] = 0
            else:
                ret_img[i][j] = 255

    return ret_img


def filter_5(img):
    """
    Remove JUGGY EDGE by 5x5 kernel
    :param img: gray scale image , np.array ((h, w, 1), dtype=np.uint8)
    :return: Remove JUGGY EDGE mat, ((h, w, 1), dtype=np.uint8)
    """
    ret_img = np.zeros((img.shape[0], img.shape[1]))
    kernel = np.zeros((5, 5))
    for i in range(2, img.shape[0] - 2):
        for j in range(2, img.shape[1] - 2):
            # window function
            Omega = np.array(
                [[img[i - 2][j - 2], img[i - 1][j - 2], img[i][j - 2], img[i + 1][j - 2], img[i + 2][j - 2]],
                 [img[i - 2][j - 1], img[i - 1][j - 1], img[i][j - 1], img[i + 1][j - 1], img[i + 2][j - 1]],
                 [img[i - 2][j], img[i - 1][j], img[i][j], img[i + 1][j], img[i - 2][j]],
                 [img[i - 2][j + 1], img[i - 1][j + 1], img[i][j + 1], img[i + 1][j + 1], img[i + 2][j + 1]],
                 [img[i - 2][j + 2], img[i - 1][j + 2], img[i][j + 2], img[i + 1][j + 2], img[i + 2][j + 2]]])

            cnt = np.sum(Omega == kernel)
            if cnt >= 17:
                ret_img[i][j] = 0
            else:
                ret_img[i][j] = 255

    return ret_img


def match_counter(mat, kernel):
    """
    matching count up(mat, kernel)
    !! MUST BE THE SAME DATA TYPE BETWEEN MAT AND KERNEL !!
    :param mat: Be examined mat (Omega Function is Preferred)
    :param kernel: kernel's shape must be (channel, h, w).
    :return: flg. 0 is not match, 1 is match. Default is 0
    """
    flg = 0
    cnt = np.zeros(kernel.shape[0])
    for i in range(kernel.shape[0]):
        cnt[i] = np.sum(mat == kernel[i])

    if cnt.max() >= 8:
        flg = 1
    return flg


def filter_col(img):
    # Test Version
    kernel = np.array([[[1, 0, 1],
                        [0, 1, 0],
                        [0, 0, 0]],
                       [[1, 0, 0],
                        [0, 1, 1],
                        [0, 0, 0]],
                       [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]],
                       [[0, 0, 1],
                        [1, 1, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [1, 1, 0],
                        [0, 0, 1]],
                       [[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]],
                       [[0, 0, 0],
                        [0, 1, 1],
                        [1, 0, 0]],
                       [[0, 0, 0],
                        [0, 1, 0],
                        [1, 0, 1]]], dtype=np.bool)

    ret_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)

    # Any fix will for following dual loops
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            Omega = np.array([[img[i - 1][j - 1], img[i][j - 1], img[i + 1][j - 1]],
                              [img[i - 1][j], img[i][j], img[i + 1][j]],
                              [img[i - 1][j + 1], img[i][j + 1], img[i + 1][j + 1]]], dtype=bool)
            flg = match_counter(Omega, kernel)
            if flg == 1:
                ret_img[i][j] = True
            else:
                ret_img[i][j] = False
    return ret_img



# >=========== DEBUG MAIN =============<

# load(PIL)
img = skimage.util.img_as_float(plt.imread('./01.jpg'))

# SLIC (SuperPixelize)
a = skimage.segmentation.slic(img, n_segments=18)
out = skimage.color.label2rgb(a, img, kind='avg')

# PIL -> CV2 Convert
tmp = np.asarray(out * 255, dtype=np.uint8)
# BGR -> RGB Convert
CV_out = tmp[:, :, ::-1].copy()

# masking/flood filling
mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
easy_f_fill(img)
gr_img = cv2.cvtColor(CV_out, cv2.COLOR_BGR2GRAY)
R_C = mask_image(gr_img)

# 5x5 kernel processing
ret_ = filter_5(R_C)
a = ret_.astype(np.bool)

# skeletonize by skimage β0.15
skeleton = skeletonize(a)

# > ============= Checking ============= <

# Original
plt.imshow(img)
plt.show()

# After SLIC
plt.imshow(out)
plt.show()

# After Binarization (CV2 image show)
cv2.imshow('Binarization', R_C)
cv2.waitKey(0)
cv2.destroyAllWindows()

# After Smoothing (CV2 image show)
cv2.imshow('Smoothing Binary', ret_)
cv2.waitKey(0)
cv2.destroyAllWindows()

# After Skeleton image
plt.imshow(skeleton)
plt.show()
