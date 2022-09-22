import numpy as np
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

# Problems/ Questions:
# 1. Each transformation need to be checked both ways
#    e.g. rotate counterclockwise and anticlockwise,  shift y from bottom and from top ect.
# 2. How to convert trans values to matrix
# 3. Stop condition on enhancing  transformation values (number of iterations parameter?)
# 4. How to enhance (Set new boundaries to fit those values?)
# 5. Make sure to use numpy arrays when needed


# Loads cl and ebsd images by path
def load_images():
    ebsd = cv2.imread(ebsd_path)
    # truth_img = cv2.imread(real_truth_path)
    cl = cv2.imread(orig_cl_path)
    return ebsd, cl


# Generates transformations within given boundaries and step size for translation, rotation and skew transformations
def generate_transformations(translate_x_bounds, translate_y_bounds, rotate_bounds, skew_x_bounds, skew_y_bounds):
    # Unpack boundaries
    min_trans_x, max_trans_x, step = translate_x_bounds

    n = (max_trans_x-min_trans_x) // step
    ret_arr = []

    for i in range(n):
        new_transformations = fix_translation_x(min_trans_x + i*step, translate_y_bounds,
                                                rotate_bounds, skew_x_bounds, skew_y_bounds)
        ret_arr.extend(new_transformations)

    return ret_arr


def fix_translation_x(trans_x, translate_y_bounds, rotate_bounds, skew_x_bounds, skew_y_bounds):
    min_trans_y, max_trans_y, step = translate_y_bounds
    n = (max_trans_y - min_trans_y) // step
    ret_arr = []

    for i in range(n):
        new_transformations = fix_translation_y(trans_x, min_trans_y + i*step,
                                                rotate_bounds, skew_x_bounds, skew_y_bounds)
        ret_arr.extend(new_transformations)

    return ret_arr


def fix_translation_y(trans_x, trans_y, rotate_bounds, skew_x_bounds, skew_y_bounds):
    min_rot, max_rot, step = rotate_bounds
    n = (max_rot - min_rot) // step
    ret_arr = []

    for i in range(n):
        new_transformations = fix_rotation(trans_x, trans_y, min_rot + i*step, skew_x_bounds, skew_y_bounds)
        ret_arr.append(new_transformations)

    return ret_arr


def fix_rotation(trans_x, trans_y, rot, skew_x_bounds, skew_y_bounds):
    return [trans_x, trans_y, rot]


def fix_skew_x():
    pass


def fix_skew_y():
    pass


# Calculates the loss of each transformation in trans_arr
def calc_loss(trans_arr):
    ebsd_canny = get_canny_img(ebsd_img)
    cl_canny = get_canny_img(cl_img)

    loss_arr = []
    height, width, ch = cl_canny.shape
    (cX, cY) = (width // 2, height // 2)

    for trans in trans_arr:
        translate_x, translate_y, rotate = trans
        # Translation
        translation_mat = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
        img = cv2.warpAffine(cl_canny, translation_mat, (width, height))
        # Rotation
        rot_mat = cv2.getRotationMatrix2D((cX, cY), rotate, 1.0)
        trans_img = cv2.warpAffine(img, rot_mat, (width, height))
        # Calc loss
        loss_val = loss_func(trans_img, ebsd_canny)
        loss_arr.append(loss_val)

    return np.array(loss_arr)


# Sums the multiplication of the transposed canny and ebsd canny. The higher the better
def loss_func(trans_img, ebsd_img):
    bi_trans = convert_img_to_bi(trans_img)
    bi_ebsd = convert_img_to_bi(ebsd_img)
    return np.sum(bi_trans*bi_ebsd)


# Converts image array to a 1D binary array of {0,1}
def convert_img_to_bi(img):
    flat_arr = img.flatten()
    ret_arr = flat_arr[::3]
    # Replace black pixels with 1 and white with 1
    ret_arr[ret_arr == 0] = 1
    ret_arr[ret_arr == 255] = 0
    return ret_arr


def get_canny_img(img):
    return ''


if __name__ == "__main__":
    # Load Images
    cl_img, ebsd_img = load_images()
    trans_x_bounds = (0, 100, 20)
    trans_y_bounds = (0, 100, 20)
    rot_bounds = (0, 40, 20)
    skewx_bounds = (0, 0, 0)
    skewy_bounds = (0, 0, 0)

    # Steps:
    # 1. Generate transformation
    # 2. Calc loss
    # 3. Find 5(?) max losses
    # 4. Enhance transformation on best loss and repeat

    # trans = generate_transformations(trans_x_bounds, trans_y_bounds, rot_bounds, skewx_bounds, skewy_bounds)
    # print(trans)
