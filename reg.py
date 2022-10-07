import numpy as np
import cv2
from cv2 import warpAffine, getRotationMatrix2D
from itertools import product
from scipy.optimize import dual_annealing, basinhopping
from multiprocessing import Pool
import multiprocessing as mp

# Image Paths
truth_path = 'data/affine_reg.png'
ebsd_path = 'Sequence/EBSD_09-07-29_1415-13_Map1-2-3-4_corr-BC-IPF-GB_crop.png'
orig_cl_path = 'Sequence/cl.png'

# Transformation boundaries
INF_TRANSLATE_X = -200
SUP_TRANSLATE_X = 200

INF_TRANSLATE_Y = -200
SUP_TRANSLATE_Y = 200

INF_ROTATION = -15
SUP_ROTATION = 15

INF_SCALE_X = 0.8
SUP_SCALE_X = 1.2

INF_SCALE_Y = 0.8
SUP_SCALE_Y = 1.2

# Initial bounds
TRANS_X_BOUNDS = (INF_TRANSLATE_X, SUP_TRANSLATE_X, 5)
TRANS_Y_BOUNDS = (INF_TRANSLATE_Y, SUP_TRANSLATE_Y, 5)
ROT_BOUNDS = (INF_ROTATION, SUP_ROTATION, 5)
SCALE_X_BOUNDS = (INF_SCALE_X, SUP_SCALE_X, 0.2)
SCALE_Y_BOUNDS = (INF_SCALE_Y, SUP_SCALE_Y, 0.2)

MIN_STEP = 1
MIN_SCALE_STEP = 0.1

# Indices
STEP_IND = 2

# Problems/Questions/Remarks:
# 8. More elegant way to get transformations by indexes in find_best_trans?
# 17. plot losses
# 20. Add resizing option for when original images aren't the same size


# Loads cl and ebsd images by path
def load_images():
    ebsd = cv2.imread(ebsd_path)
    cl = cv2.imread(orig_cl_path)
    return ebsd, cl


#  Returns the edges detected in image using canny method
def create_canny_img(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(grey_img, (5, 5), 0)

    high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(img, low_thresh, high_thresh)

    # Thicken the borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    return edges


# Returns canny images of ebsd and cl
def get_canny_images():
    # Canny Images
    ebsd_canny = create_canny_img(ebsd_img)
    cl_canny = create_canny_img(cl_img)
    return ebsd_canny, cl_canny


# Returns canny image of cl after having had given transformation applied to it
# def create_trans_img(trans):
def create_trans_img(trans, cl_canny, dim):
    # def create_trans_img(trans):
    width, height, cX, cY = dim
    translate_x, translate_y, degree, scale_x, scale_y = trans

    # Translation
    translation_mat = np.float32([[scale_x, 0, translate_x], [0, scale_y, translate_y]])
    img = warpAffine(cl_canny, translation_mat, (width, height))

    # Rotation
    rot_mat = getRotationMatrix2D((cX, cY), degree, 1.0)
    trans_img = warpAffine(img, rot_mat, (width, height))
    return trans_img


# Generates transformations within given boundaries and step size for translation, rotation and scale transformations
def generate_transformations(translate_x_bounds=TRANS_X_BOUNDS, translate_y_bounds=TRANS_Y_BOUNDS,
                             rotate_bounds=ROT_BOUNDS, scale_x_bounds=SCALE_X_BOUNDS, scale_y_bounds=SCALE_Y_BOUNDS):
    # Unpack boundaries
    min_trans_x, max_trans_x, trans_x_step = translate_x_bounds
    min_trans_y, max_trans_y, trans_y_step = translate_y_bounds
    min_rot, max_rot, rot_step = rotate_bounds
    min_scale_x, max_scale_x, scale_x_step = scale_x_bounds
    min_scale_y, max_scale_y, scale_y_step = scale_y_bounds

    trans_x_range = list(range(min_trans_x, max_trans_x + trans_x_step, trans_x_step))
    trans_y_range = list(range(min_trans_y, max_trans_y + trans_y_step, trans_y_step))
    rot_range = list(range(min_rot, max_rot, rot_step))
    scale_x_range = np.arange(min_scale_x, max_scale_x + scale_x_step, scale_x_step)
    scale_x_range = scale_x_range[scale_x_range != 0]
    scale_y_range = np.arange(min_scale_y, max_scale_y + scale_y_step, scale_y_step)
    scale_y_range = scale_y_range[scale_y_range != 0]
    a = [trans_x_range, trans_y_range, rot_range, scale_x_range, scale_y_range]

    return list(product(*a))


# Finds the best transformation for the registration according to the LOSS function
def find_best_trans(n=1):
    best_trans = optimiser(n)
    print(best_trans)
    ret_trans = calc_best_trans(best_trans)
    return ret_trans


def optimiser(n=1, trans_x_bounds=TRANS_X_BOUNDS, trans_y_bounds=TRANS_Y_BOUNDS, rot_bounds=ROT_BOUNDS,
              scale_x_bounds=SCALE_X_BOUNDS, scale_y_bounds=SCALE_Y_BOUNDS):
    prev_trans_x_step = trans_x_bounds[STEP_IND]
    prev_trans_y_step = trans_y_bounds[STEP_IND]
    prev_rot_step = rot_bounds[STEP_IND]
    prev_scale_x_step = scale_x_bounds[STEP_IND]
    prev_scale_y_step = scale_y_bounds[STEP_IND]

    trans_arr = generate_transformations(trans_x_bounds, trans_y_bounds, rot_bounds, scale_x_bounds, scale_y_bounds)
    best_trans = calc_best_trans(trans_arr, n)

    if prev_trans_x_step <= MIN_STEP and prev_trans_y_step <= MIN_STEP and prev_rot_step <= MIN_STEP and \
            prev_scale_x_step <= MIN_SCALE_STEP and prev_scale_y_step <= MIN_SCALE_STEP:
        return best_trans

    ret_trans = []
    ret_trans.extend(best_trans)

    for trans in best_trans:
        new_trans_x_bound, new_trans_y_bound, new_rot_bounds, new_scale_x_bounds, new_scale_y_bounds = \
            get_new_boundaries(trans, prev_trans_x_step, prev_trans_y_step, prev_rot_step, prev_scale_x_step,
                               prev_scale_y_step)
        new_trans = optimiser(n, new_trans_x_bound, new_trans_y_bound, new_rot_bounds, new_scale_x_bounds,
                              new_scale_y_bounds)
        ret_trans.extend(new_trans)

    ret_trans = np.array(ret_trans)
    return np.unique(ret_trans, axis=0)


# Defines new boundaries for next generation of possible transformations
def get_new_boundaries(trans, trans_x_step, trans_y_step, rot_step, scale_x_step, scale_y_step):
    x, y, deg, scale_x, scale_y = trans
    new_trans_x_inf = max(x-trans_x_step, INF_TRANSLATE_X)
    new_trans_x_sup = min(x+trans_x_step, SUP_TRANSLATE_X)
    new_trans_x_step = max(MIN_STEP, trans_x_step//2)
    new_trans_x_bounds = (new_trans_x_inf, new_trans_x_sup, new_trans_x_step)

    new_trans_y_inf = max(y-trans_y_step, INF_TRANSLATE_Y)
    new_trans_y_sup = min(y+trans_y_step, SUP_TRANSLATE_Y)
    new_trans_y_step = max(MIN_STEP, trans_y_step//2)
    new_trans_y_bounds = (new_trans_y_inf, new_trans_y_sup, new_trans_y_step)

    new_rot_inf = max(deg-rot_step, INF_ROTATION)
    new_rot_sup = min(deg+rot_step, SUP_ROTATION)
    new_rot_step = max(MIN_STEP, rot_step//2)
    new_rot_bounds = (new_rot_inf, new_rot_sup, new_rot_step)

    new_scale_x_inf = max(scale_x-scale_x_step, INF_SCALE_X)
    new_scale_x_sup = min(scale_x+scale_x_step, SUP_SCALE_X)
    new_scale_x_bounds = (new_scale_x_inf, new_scale_x_sup, scale_x_step/2)

    new_scale_y_inf = max(scale_y-scale_y_step, INF_SCALE_Y)
    new_scale_y_sup = min(scale_y+scale_y_step, SUP_SCALE_Y)
    new_scale_y_bounds = (new_scale_y_inf, new_scale_y_sup, scale_y_step/2)

    return new_trans_x_bounds, new_trans_y_bounds, new_rot_bounds, new_scale_x_bounds, new_scale_y_bounds


# Returns loss val of each transformation in input array
def get_arr_loss(trans_arr):
    # Zipping the global arguments for multiprocessing to work
    cl_arr = [cl_canny]*len(trans_arr)
    ebsd_arr = [ebsd_canny]*len(trans_arr)
    dim = [(width, height, cX, cY)]*len(trans_arr)
    args = list(zip(trans_arr, cl_arr, ebsd_arr, dim))

    with Pool(mp.cpu_count() - 1) as pool:
        results = pool.starmap(loss_func, args)
    return results
    # return np.array([loss_func(trans) for trans in trans_arr])


# Sums the multiplication of the transposed canny and ebsd canny. The higher, the better
def loss_func(trans, cl_canny, ebsd_canny, dim):
# def loss_func(trans):
    trans_canny = create_trans_img(trans, cl_canny, dim)
    # trans_canny = create_trans_img(trans)
    inter = ebsd_canny & trans_canny
    flat_arr = inter.flatten()
    flat_arr[flat_arr != 0] = 1
    loss = np.sum(flat_arr)
    return loss*-1

# Simulated Annealing Optimiser LOSS function
# def loss_func(pts):
#     p1_r, p1_c, p2_r, p2_c, p3_r, p3_c, p4_r, p4_c = pts
#     coord1 = np.array([[p1_r, p1_c], [p2_r, p2_c], [p3_r, p3_c], [p4_r, p4_c]], np.float32)
#     # coord2 = [[q1_r, q1_c], [q2_r, q2_c], [q3_r, q3_c], [q4_r, q4_c]]
#     coord2 = np.array([[0, 0], [0, width], [height, 0], [height, width]], np.float32)
#
#     mat = cv2.getPerspectiveTransform(coord1, coord2)
#     trans_img = cv2.warpPerspective(cl_canny, mat, cl_canny.shape)
#     bi_trans = convert_canny_to_bi(trans_img)
#     loss = np.sum(bi_ebsd*bi_trans)
#     if abs(sum(bi_ebsd) - loss) < 10:
#         loss = 0
#     return loss*-1


# Returns n transformations from trans_arr with the highest loss values
def calc_best_trans(trans_arr, n=1):
    trans_loss = get_arr_loss(trans_arr)
    top_ind = np.argsort(trans_loss)[:n]
    best_trans = []
    # Remark 8- make more elegant
    for i in top_ind:
        best_trans.append((trans_arr[i]))

    return best_trans


# Three different ways to test registration results: 1. show_trans- displays registration
#                                                    2. show_super_impose- superimposes registration & ebsd
#                                                    3. show_traces- Displays intersection between registration canny
#                                                                    and ebsd canny
def test(trans, show_trans=True, show_super_impose=True, show_traces=True):
    trans_x, trans_y, deg, scale_x, scale_y = trans

    loss = loss_func(trans, cl_canny, ebsd_canny, dim)
    print('loss: ' + str(loss))

    tran_mat = np.float32([[scale_x, 0, trans_x], [0, scale_y, trans_y]])
    img = cv2.warpAffine(cl_canny, tran_mat, (width, height))
    rot_mat = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    if show_trans:
        tran = cv2.warpAffine(img, rot_mat, (width, height))
        cv2.imshow('', tran)
        cv2.waitKey(0)
    if show_super_impose:
        img1 = cv2.warpAffine(cl_img, tran_mat, (width, height))
        tran1 = cv2.warpAffine(img1, rot_mat, (width, height))
        super_impose = cv2.addWeighted(ebsd_img, 0.4, tran1, 0.5, 0)
        cv2.imshow('superimpose_lim.png', super_impose)
    if show_traces:
        inter = tran & ebsd_canny
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        inter = cv2.dilate(inter, kernel, iterations=2)
        ebsd_img[inter > 0] = [0, 0, 0]
        # cv2.imwrite('overlap.png', ebsd_img)
        cv2.imshow('overlap.png', ebsd_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    # Load Images
    ebsd_img, cl_img = load_images()
    ebsd_canny, cl_canny = get_canny_images()

    # Image dimensions
    height, width = cl_canny.shape
    (cX, cY) = (width // 2, height // 2)
    dim = (width, height, cX, cY)

    # Truth loss: -144333
    # transformation = find_best_trans(2)
    # print(transformation)

    # test([28. , -33. ,   6. ,   1.2,   1.2])
