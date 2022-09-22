import numpy as np
from torch import sum
import scipy.optimize as opt
import cv2

# Image Paths
# real_truth_path = 'data/affine_reg.png'
ebsd_path = 'Sequence/EBSD_09-07-29_1415-13_Map1-2-3-4_corr-BC-IPF-GB_crop.png'
orig_cl_path = 'Sequence/cl.png'

# Transformation boundaries
sup_translation_y = None
sup_rotation = 90
sup_x_skew = None
sup_y_skew = None

# Colour codes
black = [0, 0, 0]
white = [255, 255, 255]

# Problems:
# 1. Each transformation need to be checked both ways
#    e.g. rotate counter clockwise and anti clockwise,  shift y from bottom and from top ect.


# Loads cl and ebsd images by path
def load_images():
    ebsd_img = cv2.imread(ebsd_path)
    # truth_img = cv2.imread(real_truth_path)
    cl_img = cv2.imread(orig_cl_path)
    return ebsd_img, cl_img


# Generates transformations within given boundaries and step size for translation, rotation and skew transformations
def generate_transformations(translate_x_bounds, translate_y_bounds, rotate_bounds, skew_x_bounds, skew_y_bounds):
    # Unpack boundaries
    min_trans_x, max_trans_x, step_trans_x = translate_x_bounds
    min_trans_y, max_trans_y, step_trans_y = translate_y_bounds
    min_rot, max_rot, step_rot = rotate_bounds
    min_skew_x, max_skew_x, step_skew_x = skew_x_bounds
    min_skew_y, max_skew_y, step_skew_y = skew_y_bounds

    n = (max_trans_x-min_trans_y)/ step_trans_x
    ret_arr = []

    for i in range(n):
        new_transformations = fix_translation_x(translate_x_bounds, translate_y_bounds, rotate_bounds, skew_x_bounds, skew_y_bounds)
        ret_arr.append(new_transformations)

    return ret_arr


def fix_translation_x(trans_x, translate_y_bounds, rotate_bounds, skew_x_bounds, skew_y_bounds):
    pass


def fix_translation_y():
    pass


def fix_rotation_x():
    pass


# Converts image array to a 1D binary array of {0,1}
def convert_img_to_bi(img):
    flat_arr = img.flatten()
    ret_arr = flat_arr[::3]
    # Replace black pixels with 1 and white with 1
    ret_arr[ret_arr == 0] = 1
    ret_arr[ret_arr == 255] = 0
    return ret_arr


def loss_func(trans_img, ebsd_img):
    bi_trans = convert_img_to_bi(trans_img)
    bi_ebsd = convert_img_to_bi(ebsd_img)
    return np.sum(bi_trans*bi_ebsd)


if __name__ == "__main__":
    # Load Images
    cl_img, ebsd_img = load_images()