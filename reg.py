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

# Initial bounds
# Change later to (-sup, sup, step)
TRANS_X_BOUNDS = (-100, 100, 20)
TRANS_Y_BOUNDS = (-100, 100, 20)
ROT_BOUNDS = (-40, 40, 20)
SKEW_X_BOUNDS = (0, 0, 0)
SKEW_Y_BOUNDS = (0, 0, 0)

DIFF_VAL = 50000

# Colour codes
black = [0, 0, 0]
white = [255, 255, 255]

# Problems/ Questions/ Remarks:
# 1. Each transformation need to be checked both ways
#           e.g. rotate counterclockwise and anticlockwise,  shift y from bottom and from top ect.
# 2. How to convert trans values to matrix
# 3. Stop condition on enhancing  transformation values
#           * number of iterations as a parameter?
#           * when loss is close enough to loss of ebsd? what do img to I compare to? ebsd_canny has to many lines
# 4. How to enhance (Set new boundaries to fit those values?)
# 5. Make sure to use numpy arrays when needed
# 6. Not sure how to implement skew values, should I opt for guessing 4 points for perspective warp?
# 7. helper functions for generate transformation are repetitive but with different parameters, is there a way to
#           generalise them to one function?
# 8. More elegant way to get transformations by indexes in find_best_trans?


# Loads cl and ebsd images by path
def load_images():
    ebsd = cv2.imread(ebsd_path)
    # truth_img = cv2.imread(real_truth_path)
    cl = cv2.imread(orig_cl_path)
    return ebsd, cl


# Generates transformations within given boundaries and step size for translation, rotation and skew transformations
def generate_transformations(translate_x_bounds=TRANS_X_BOUNDS, translate_y_bounds=TRANS_Y_BOUNDS,
                             rotate_bounds=ROT_BOUNDS, skew_x_bounds=SKEW_X_BOUNDS, skew_y_bounds=SKEW_Y_BOUNDS):
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
    cl_canny = get_canny_img(cl_img)
    loss_arr = []
    height, width = cl_canny.shape
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
        loss_val = loss_func(trans_img)
        loss_arr.append(loss_val)

    return np.array(loss_arr)


# Sums the multiplication of the transposed canny and ebsd canny. The higher the better
def loss_func(trans_img):
    bi_trans = convert_canny_to_bi(trans_img)
    bi_ebsd = convert_canny_to_bi(ebsd_canny)
    return np.sum(bi_trans*bi_ebsd)


# Converts image array to a 1D binary array of {0,1}
def convert_canny_to_bi(img):
    flat_arr = img.flatten()
    # Replace black pixels with 1 and white with 1
    flat_arr[flat_arr == 0] = 1
    flat_arr[flat_arr != 1] = 0
    return flat_arr


#  Returns the borders detected in image using canny method
def get_canny_img(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    high_thresh, thresh_im = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh

    edges = cv2.Canny(grey_img, low_thresh, high_thresh)
    # cv2.imshow('2', edges)
    # cv2.waitKey()
    return edges


def calc_diff(img):
    return abs(loss_func(img) - loss_func(ebsd_img))


def find_best_trans(n=1):
    # Stop condition
    # if calc_diff(trans) < DIFF_VAL:
    #      return trans

    # Gets initial transformation options and their loss values
    transformations = generate_transformations()
    best_trans = calc_best_trans(transformations, n)
    return find_best_trans_helper(best_trans, 2)


def find_best_trans_helper(trans_arr, n, trans_x_bounds=TRANS_X_BOUNDS, trans_y_bounds=TRANS_Y_BOUNDS,
                           rot_bounds=ROT_BOUNDS, skewx_bounds=SKEW_X_BOUNDS, skewy_bounds=SKEW_Y_BOUNDS):
    # Need a stop condition
    # Stop if all children transformation are off
    # Possibly also a filter so that the return arr isn't too large

    # if calc_diff(trans_arr) <= DIFF_VAL:
    #   return trans_arr

    best_trans = []

    for trans in trans_arr:
        # Previous transformation data
        x, y, rot = trans
        prev_x_step = trans_x_bounds[2]
        prev_y_step = trans_y_bounds[2]
        prev_rot_step = rot_bounds[2]
        # New transformation boundaries
        x_step = prev_x_step//2 if prev_x_step//2 >= 1 else prev_x_step
        y_step = prev_y_step//2 if prev_y_step//2 >= 1 else prev_y_step
        rot_step = prev_rot_step//2 if prev_rot_step//2 >= 1 else prev_rot_step
        # x_step = trans_x_bounds[2]/2
        # y_step = trans_y_bounds[2]/2
        # rot_step = rot_bounds[2]/2

        # if x_step == prev_x_step and y_step == prev_y_step and rot_step == prev_rot_step:
        #   STOP
        if x_step != prev_x_step or y_step != prev_y_step or rot_step != prev_rot_step:
            trans_x_bounds = (x - x_step, x + x_step, x_step)
            trans_y_bounds = (y - y_step, y + y_step, y_step)
            rot_bounds = (rot - rot_step, rot + rot_step, rot_step)
            # New transformations
            # Currently has no stop condition so will not surpass this line
            new_trans = generate_transformations(trans_x_bounds, trans_y_bounds, rot_bounds, skewx_bounds, skewy_bounds)
            # Plan B stop condition: if step < 1 (pixel)
            new_best_trans = calc_best_trans(new_trans, n)
            print(new_best_trans)
            best_trans.append(new_best_trans)
    return best_trans


# Returns n transformations from trans_arr with the highest loss values
def calc_best_trans(trans_arr, n=1):
    trans_loss = calc_loss(trans_arr)
    top_ind = np.argsort(trans_loss)[-n:]
    best_trans = []
    # Remark 8- make more elegant
    for i in top_ind:
        best_trans.append(trans_arr[i])

    return best_trans


if __name__ == "__main__":
    # Load Images
    cl_img, ebsd_img = load_images()
    ebsd_canny = get_canny_img(ebsd_img)

    # Steps:
    # 1. Generate transformation
    # 2. Calc loss
    # 3. Find 5(?) max losses
    # 4. Enhance transformation on best loss and repeat

    # Find initial best transformations
    # Enhance each transformation; repeat process for each transformation by setting the boundaries accordingly;
    #

    bes = find_best_trans()
    print(len(bes))
