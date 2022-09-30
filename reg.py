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
SUP_TRANSLATE_Y = 110
SUP_ROTATION = 15
INF_SCALE_X = 0.8
SUP_SCALE_X = 1.5
INF_SCALE_Y = 0.8
SUP_SCALE_Y = 1.5

# Initial bounds
TRANS_X_BOUNDS = (-120, 120, 10)
TRANS_Y_BOUNDS = (-110, 120, 10)
ROT_BOUNDS = (-15, 15, 3)
SCALE_X_BOUNDS = (0.8, 1.2, 0.2)
SCALE_Y_BOUNDS = (0.8, 1.2, 0.2)

# Indices
MIN = 0
MAX = 1
STEP = 2

# Problems/Questions/Remarks:
# 8. More elegant way to get transformations by indexes in find_best_trans?
# 9. Change boundary constant to (-sup, sup, step) form
# 14. Is it better to filter best_trans by loss func on the go?
#       * Apply loss_func (which is slow) once on a very large array
#                       VS
#       * Apply every iteration?
# 15. Implement multiprocessing
# 16. Stop condition for scale value? <= 0.01?


# Loads cl and ebsd images by path
def load_images():
    ebsd = cv2.imread(ebsd_path)
    cl = cv2.imread(orig_cl_path)
    return ebsd, cl


#  Returns the edges detected in image using canny method
def create_canny_img(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(grey_img, (5, 5), 0)
    # quant = quantise_img(img, 8)

    # high_thresh, thresh_im = cv2.threshold(quant, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(img, low_thresh, high_thresh)
    return edges


# Returns canny images of ebsd and cl
def get_canny_images():
    # Canny Images
    ebsd_canny = create_canny_img(ebsd_img)
    cl_canny = create_canny_img(cl_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ebsd_canny = cv2.dilate(ebsd_canny, kernel, iterations=2)
    return ebsd_canny, cl_canny
    # cl_dil_canny = cv2.dilate(cl_canny, kernel, iterations=2)
    # return ebsd_dil_canny, cl_dil_canny


# Returns canny image of cl after having had given transformation applied to it
def create_trans_img(trans):
    translate_x, translate_y, degree, scale_x, scale_y = trans
    # Translation
    translation_mat = np.float32([[scale_x, 0, translate_x], [0, scale_y, translate_y]])
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
    scale_x_range = np.arange(min_scale_x, max_scale_x, scale_x_step)
    scale_y_range = np.arange(min_scale_y, max_scale_y, scale_y_step)
    a = [trans_x_range, trans_y_range, rot_range, scale_x_range, scale_y_range]

    return list(itertools.product(*a))


# Finds the best transformation for the registration according to the LOSS function
def find_best_trans(n=1):
    best_trans = find_best_trans_helper(n)
    ret_trans = calc_best_trans(best_trans)
    return ret_trans


def find_best_trans_helper(n=1, trans_x_bounds=TRANS_X_BOUNDS, trans_y_bounds=TRANS_Y_BOUNDS, rot_bounds=ROT_BOUNDS,
                           scale_x_bounds=SCALE_X_BOUNDS, scale_y_bounds=SCALE_Y_BOUNDS):
    prev_trans_x_step = trans_x_bounds[STEP]
    prev_trans_y_step = trans_y_bounds[STEP]
    prev_rot_step = rot_bounds[STEP]
    prev_scale_x_step = scale_x_bounds[STEP]
    prev_scale_y_step = scale_y_bounds[STEP]

    trans_arr = generate_transformations(trans_x_bounds, trans_y_bounds, rot_bounds, scale_x_bounds, scale_y_bounds)
    best_trans = calc_best_trans(trans_arr, n)

    if prev_trans_x_step <= 1 and prev_trans_y_step <= 1 and prev_rot_step <= 1 and prev_scale_x_step <= 0.01 \
            and prev_scale_y_step <= 0.01:
        return best_trans

    ret_trans = []
    ret_trans.extend(best_trans)

    for trans in best_trans:
        new_trans_x_bound, new_trans_y_bound, new_rot_bounds, new_scale_x_bounds, new_scale_y_bounds = \
            get_new_boundaries(trans, prev_trans_x_step, prev_trans_y_step, prev_rot_step, prev_scale_x_step,
                               prev_scale_y_step)
        new_trans = find_best_trans_helper(n, new_trans_x_bound, new_trans_y_bound, new_rot_bounds, new_scale_x_bounds,
                                           new_scale_y_bounds)
        ret_trans.extend(new_trans)

    ret_trans = np.array(ret_trans)
    return np.unique(ret_trans, axis=0)


def get_new_boundaries(trans, trans_x_step, trans_y_step, rot_step, scale_x_step, scale_y_step):
    x, y, deg, scale_x, scale_y = trans
    new_trans_x_inf = max(x-trans_x_step, -SUP_TRANSLATE_X)
    new_trans_x_sup = min(x+trans_x_step, SUP_TRANSLATE_X)
    new_trans_x_bounds = (new_trans_x_inf, new_trans_x_sup, trans_x_step//2)

    new_trans_y_inf = max(y-trans_y_step, -SUP_TRANSLATE_Y)
    new_trans_y_sup = min(y+trans_y_step, SUP_TRANSLATE_Y)
    new_trans_y_bounds = (new_trans_y_inf, new_trans_y_sup, trans_y_step//2)

    new_rot_inf = max(deg-rot_step, -SUP_ROTATION)
    new_rot_sup = min(deg+rot_step, SUP_ROTATION)
    new_rot_bounds = (new_rot_inf, new_rot_sup, trans_y_step//2)

    new_scale_x_inf = max(scale_x-scale_x_step, INF_SCALE_X)
    new_scale_x_sup = min(scale_x+scale_x_step, SUP_SCALE_X)
    new_scale_x_bounds = (new_scale_x_sup, new_scale_x_inf, scale_x_step/2)

    new_scale_y_inf = max(scale_y-scale_y_step, INF_SCALE_Y)
    new_scale_y_sup = min(scale_y+scale_y_step, SUP_SCALE_Y)
    new_scale_y_bounds = (new_scale_y_sup, new_scale_y_inf, scale_y_step/2)

    return new_trans_x_bounds, new_trans_y_bounds, new_rot_bounds, new_scale_x_bounds, new_scale_y_bounds


# Returns loss val of each transformation in input array
def loss_func(trans_arr):
    # results = pool.map(loss_func_helper, [trans for trans in trans_arr])
    # return results
    return np.array([loss_func_helper(trans) for trans in trans_arr])


# Sums the multiplication of the transposed canny and ebsd canny. The higher, the better
def loss_func_helper(trans):
    trans_img = create_trans_img(trans)
    bi_trans = convert_canny_to_bi(trans_img)
    bi_ebsd = convert_canny_to_bi(ebsd_canny)
    return np.sum(bi_ebsd*bi_trans)


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
    ebsd_canny, cl_canny = get_canny_images()

    # Image dimensions
    height, width = cl_canny.shape
    (cX, cY) = (width // 2, height // 2)

    # pool = mp.Pool(mp.cpu_count() - 1)
    transformation = find_best_trans(1)
    print(transformation)
    # pool.close()

    # trans_arr = generate_transformations((-120, 120, 1), (-120, 120, 1), (-10, 10, 1))
    # best = calc_best_trans(trans_arr)
    # print(best)

    # Results: (-120, -110, 0, 0.7. 0.7)
    # tran_mat = np.float32([[0.7, 0, -120], [0, 0.7, -110]])
    # img = cv2.warpAffine(cl_img, tran_mat, (width, height))
    # rot_mat = cv2.getRotationMatrix2D((cX, cY), 0, 1.0)
    # tran = cv2.warpAffine(img, rot_mat, (width, height))
    #
    # super_impose = cv2.addWeighted(ebsd_img, 0.4, tran, 0.5, 0)
    # cv2.imshow('super impose', super_impose)
    #
    # black = np.array([0, 0, 0])
    # ebsd_img[inter > 0] = black
    # cv2.imshow('', ebsd_img)
    # cv2.waitKey(0)