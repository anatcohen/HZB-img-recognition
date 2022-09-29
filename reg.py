import numpy as np
import cv2
import itertools
from hist_methods import quantise_img
import multiprocessing as mp

# Image Paths
truth_path = 'data/affine_reg.png'
ebsd_path = 'Sequence/EBSD_09-07-29_1415-13_Map1-2-3-4_corr-BC-IPF-GB_crop.png'
orig_cl_path = 'Sequence/cl.png'


# Transformation boundaries
SUP_TRANSLATE_X = 120
SUP_TRANSLATE_Y = 120
SUP_ROTATION = 15
SUP_X_SKEW = None
SUP_Y_SKEW = None

# Initial bounds
TRANS_X_BOUNDS = (-120, 120, 20)
TRANS_Y_BOUNDS = (-110, 120, 20)
ROT_BOUNDS = (-20, 20, 3)
SKEW_X_BOUNDS = (0, 0, 0)
SKEW_Y_BOUNDS = (0, 0, 0)

# Indices
MIN = 0
MAX = 1
STEP = 2

# Problems/Questions/Remarks:
# 6. Not sure how to implement skew values, should I opt for guessing 4 points for perspective warp?
# 8. More elegant way to get transformations by indexes in find_best_trans?
# 9. Change boundary constant to (-sup, sup, step) form
# 14. Is it better to filter best_trans by loss func on the go?
#       * Apply loss_func (which is slow) once on a very large array
#                       VS
#       * Apply every iteration?
# 15. Implement multiprocessing


# Loads cl and ebsd images by path
def load_images():
    ebsd = cv2.imread(ebsd_path)
    cl = cv2.imread(orig_cl_path)
    return ebsd, cl


#  Returns the edges detected in image using canny method
def create_canny_img(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(grey_img, (5, 5), 0)
    quant = quantise_img(img, 8)

    high_thresh, thresh_im = cv2.threshold(quant, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(img, low_thresh, high_thresh)
    return edges


# Returns canny images of ebsd and cl
def get_canny_images():
    # Canny Images
    ebsd_canny = create_canny_img(ebsd_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # ebsd_dil_canny = cv2.dilate(ebsd_canny, kernel, iterations=2)
    cl_canny = create_canny_img(cl_img)
    return ebsd_canny, cl_canny


# Returns canny image of cl after having had given transformation applied to it
def create_trans_img(trans):
    translate_x, translate_y, degree = trans
    # Translation
    translation_mat = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    img = cv2.warpAffine(cl_canny, translation_mat, (width, height))
    # Rotation
    rot_mat = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
    trans_img = cv2.warpAffine(img, rot_mat, (width, height))
    return trans_img


# Converts image array to a 1D binary array of {0,1}
def convert_canny_to_bi(img):
    flat_arr = img.flatten()
    # Replace black pixels with 1 and white with 1
    flat_arr[flat_arr == 0] = 1
    flat_arr[flat_arr != 1] = 0
    return flat_arr


# Generates transformations within given boundaries and step size for translation, rotation and skew transformations
def generate_transformations(translate_x_bounds=TRANS_X_BOUNDS, translate_y_bounds=TRANS_Y_BOUNDS,
                             rotate_bounds=ROT_BOUNDS, skew_x_bounds=SKEW_X_BOUNDS, skew_y_bounds=SKEW_Y_BOUNDS):
    # Unpack boundaries
    min_trans_x, max_trans_x, trans_x_step = translate_x_bounds
    min_trans_y, max_trans_y, trans_y_step = translate_y_bounds
    min_rot, max_rot, rot_step = rotate_bounds

    trans_x_range = list(range(min_trans_x, max_trans_x + trans_x_step, trans_x_step))
    trans_y_range = list(range(min_trans_y, max_trans_y + trans_y_step, trans_y_step))
    rot_range = list(range(min_rot, max_rot, rot_step))
    a = [trans_x_range, trans_y_range, rot_range]

    return list(itertools.product(*a))


# Finds the best transformation for the registration according to the LOSS function
def find_best_trans(n=1):
    best_trans = find_best_trans_helper(n)
    ret_trans = calc_best_trans(best_trans)
    return ret_trans


def find_best_trans_helper(n=1, trans_x_bounds=TRANS_X_BOUNDS, trans_y_bounds=TRANS_Y_BOUNDS, rot_bounds=ROT_BOUNDS,
                           scale_x_bounds=SKEW_X_BOUNDS, scale_y_bounds=SKEW_Y_BOUNDS):
    prev_trans_x_step = trans_x_bounds[STEP]
    prev_trans_y_step = trans_y_bounds[STEP]
    prev_rot_step = rot_bounds[STEP]
    prev_scale_x_step = scale_x_bounds[STEP]
    prev_scale_y_step = scale_y_bounds[STEP]

    trans_arr = generate_transformations(trans_x_bounds, trans_y_bounds, rot_bounds, scale_x_bounds, scale_y_bounds)
    best_trans = calc_best_trans(trans_arr, n)

    if prev_trans_x_step <= 1 and prev_trans_y_step <= 1 and prev_rot_step <= 1 and prev_scale_x_step <= 1 \
            and prev_scale_y_step <= 1:
        return best_trans

    ret_trans = []
    ret_trans.extend(best_trans)

    for trans in best_trans:
        new_trans_x_bound, new_trans_y_bound, new_rot_bounds = get_new_boundaries(trans, prev_trans_x_step,
                                                                                  prev_trans_y_step, prev_rot_step,
                                                                                  prev_scale_x_step, prev_scale_y_step)
        new_trans = find_best_trans_helper(n, new_trans_x_bound, new_trans_y_bound, new_rot_bounds, scale_x_bounds,
                                           scale_y_bounds)
        ret_trans.extend(new_trans)

    ret_trans = np.array(ret_trans)
    return np.unique(ret_trans, axis=0)


def get_new_boundaries(trans, trans_x_step, trans_y_step, rot_step, scale_x, scale_y):
    x, y, deg = trans
    new_trans_x_inf = max(x-trans_x_step, -SUP_TRANSLATE_X)
    new_trans_x_sup = min(x+trans_x_step, SUP_TRANSLATE_X)
    new_trans_x_bounds = (new_trans_x_inf, new_trans_x_sup, trans_x_step//2)

    new_trans_y_inf = max(y-trans_y_step, -SUP_TRANSLATE_Y)
    new_trans_y_sup = min(y+trans_y_step, SUP_TRANSLATE_Y)
    new_trans_y_bounds = (new_trans_y_inf, new_trans_y_sup, trans_y_step//2)

    new_rot_inf = max(deg-rot_step, -SUP_ROTATION)
    new_rot_sup = min(deg+rot_step, SUP_ROTATION)
    new_rot_bounds = (new_rot_inf, new_rot_sup, trans_y_step//2)

    # new_skew_x_sup = ''
    # new_skew_x_inf = ''
    # new_skew_y_sup = ''
    # new_skew_y_inf = ''
    return new_trans_x_bounds, new_trans_y_bounds, new_rot_bounds


# def old_find_best_trans(n=1):
#     # Gets initial transformation options and their loss values
#     init_trans = generate_transformations()
#     best_trans = find_best_trans_helper(init_trans, n)
#     best_trans = calc_best_trans(best_trans)
#     return best_trans

#
# def old_find_best_trans_helper(trans_arr, n, trans_x_step=TRANS_X_BOUNDS[2], trans_y_step=TRANS_Y_BOUNDS[2],
#                            rot_step=ROT_BOUNDS[2], skew_x_step=SKEW_X_BOUNDS[2], skew_y_step=SKEW_Y_BOUNDS[2]):
#     if trans_x_step <= 1 and trans_y_step <= 1 and rot_step <= 1:
#         return trans_arr
#
#     best_trans = []
#
#     # New transformation steps
#     new_trans_x_step = trans_x_step//2 if trans_x_step//2 >= 1 else 1
#     new_trans_y_step = trans_y_step//2 if trans_y_step//2 >= 1 else 1
#     new_rot_step = rot_step//2 if rot_step//2 >= 1 else 1
#
#     for trans in trans_arr:
#         # Previous transformation data
#         x, y, rot = trans
#         # ******* Problems here *******
#         #       1.  x+_ step limits the search span
#         #       2. new_rot_step goes out of bounds
#         trans_x_bounds = (x - new_trans_x_step, x + new_trans_x_step, new_trans_x_step)
#         trans_y_bounds = (y - new_trans_y_step, y + new_trans_y_step, new_trans_y_step)
#         rot_bounds = (rot - new_rot_step, rot + new_rot_step, new_rot_step)
#         # New transformations
#         new_trans = generate_transformations(trans_x_bounds, trans_y_bounds, rot_bounds, skew_x_step, skew_y_step)
#         best_trans.extend(new_trans)
#
#     best_trans = calc_best_trans(best_trans, n)
#     return find_best_trans_helper(best_trans, n, new_trans_x_step, new_trans_y_step, new_rot_step, skew_x_step, skew_y_step)


# Calculates the loss of each transformation in trans_arr

# Returns loss val of each transformation in input array
def loss_func(trans_arr):
    # results = pool.map(loss_func_helper, [trans for trans in trans_arr])
    # return results
    return np.array([loss_func_helper(trans) for trans in trans_arr])


# Sums the multiplication of the transposed canny and ebsd canny. The higher, the better
def loss_func_helper(trans):
    # trans_img = create_trans_img(trans)
    # bi_trans = convert_canny_to_bi(trans_img)
    # bi_ebsd = convert_canny_to_bi(ebsd_canny)
    # return np.sum(bi_trans*bi_ebsd)
    trans_img = create_trans_img(trans)
    intersect_canny = trans_img & ebsd_canny
    bi_inter_canny = convert_canny_to_bi(intersect_canny)
    return np.sum(bi_inter_canny)


# Returns n transformations from trans_arr with the highest loss values
def calc_best_trans(trans_arr, n=1):
    trans_loss = loss_func(trans_arr)
    top_ind = np.argsort(trans_loss)[-n:]
    best_trans = []
    # Remark 8- make more elegant
    for i in top_ind:
        best_trans.append(trans_arr[i])

    return best_trans


if __name__ == "__main__":
    # Load Images
    ebsd_img, cl_img = load_images()
    truth = cv2.imread(truth_path)
    ebsd_canny, cl_canny = get_canny_images()

    cv2.imshow('', ebsd_canny)
    cv2.waitKey(0)

    # Image dimensions
    height, width = cl_canny.shape
    (cX, cY) = (width // 2, height // 2)

    # pool = mp.Pool(mp.cpu_count() - 1)
    transformation = find_best_trans(3)
    print(transformation)
    # pool.close()

    # Results: (-120, -120, -15)
    #
    # tran_mat = np.float32([[1, 0, -120], [0, 1, -120]])
    # img = cv2.warpAffine(cl_img, tran_mat, (width, height))
    # rot_mat = cv2.getRotationMatrix2D((cX, cY), -15, 1.0)
    # tran = cv2.warpAffine(img, rot_mat, (width, height))

    # super_impose = cv2.addWeighted(ebsd_img, 0.4, tran, 0.5, 0)
    # cv2.imshow('super impose', super_impose)

    # canny_img = cv2.warpAffine(cl_canny, tran_mat, (width, height))
    # canny_tran = cv2.warpAffine(canny_img, rot_mat, (width, height))
    # inter = ebsd_canny & canny_tran
    # black = np.array([0, 0, 0])
    # ebsd_img[inter > 0] = black
    # cv2.imshow('', ebsd_img)
    # cv2.imshow('canny', ebsd_canny)
    # cv2.waitKey(0)

    # cv2.imwrite('C:/Users/Admin/Documents/Helmholtz/Prep Code/init_res.png', tran)

    # Cl canny is global and not recognised in multiprocessing
