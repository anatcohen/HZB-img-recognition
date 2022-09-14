import numpy as np
from torch import cumsum, abs, sum
import hist_methods
import scipy.optimize as opt
import cv2

# Problems/Questions:
# 1. Will one frequent colour be enough?
# 2. Find how to get the right range of frequent colours
# 3. Find most accurate method


def calc_emd(p):
    x = ref_his - p
    y = cumsum(x, dim=0)
    return abs(y).sum()


def old_idea():
    # def find_best_transformation(points, step, min_emd=None, min_trans=None):
    #     # General idea:
    #     # Fix three points at a time by order:
    #     # 1. Fix nw corner
    #     # 2. Fix ne corner
    #     # 3. Fix sw corner
    #     # 4. Fix se corner
    #     # 5. Calc emd
    #     # 6. Repeat 4 after and find min emd
    #     # 7. Repeat 3 and find min emd
    #     # 8. Repeat 2 and find min emd
    #     # 9. Repeat 1 and find min emd
    #
    #     nw = points[0]  # north-west, north-east, south-west, south-east corners
    #
    #     if nw >= height:
    #         return min_trans
    #
    #     new_emd, new_mat = ne_transformation(points, step, min_emd, min_trans)
    #
    #     if new_emd < min_emd:
    #         min_emd = new_emd
    #         min_mat = new_mat
    #
    #     points[0] += step
    #     return find_best_transformation(points, step, min_emd, min_mat)

    # def ne_transformation(points, step, min_emd, min_mat):
    #     if points[1] > width:
    #         return min_emd, min_mat
    #
    #     new_emd, new_mat = sw_transformation(points, min_emd, min_mat)
    #
    #     if new_emd < min_emd:
    #         min_emd = new_emd
    #         min_mat = new_mat
    #
    #     points[1] += step
    #
    #     return ne_transformation(points, step, min_emd, min_mat)
    #
    #
    # def sw_transformation(points, step, min_emd, min_mat):
    #     # if points[2] <
    #     new_emd, new_mat = se_transformation(points, step, min_emd, min_mat)
    #     if new_emd < min_emd:
    #         min_emd = new_emd
    #         min_mat = new_mat
    #
    #     points[2] += step
    #     return sw_transformation()
    #
    #
    # def se_transformation(points, step, min_emd, min_mat):
    #     se = points[4]
    #     if se > width:
    #         return min_mat
    #
    #     trans_mat = cv2.getPerspectiveTransform(*points)
    #     warp_img = cv2.warpPerspective(cl_img, trans_mat, (width, height))
    #     mask_img = ''
    #     trans_his = fixed_hist.get_colours_by_coords(warp_img, mask_img)
    #     new_emd = calc_emd(trans_his)
    #
    #     if min_emd is None or new_emd < min_emd:
    #         min_emd = new_emd
    #         min_mat = trans_mat
    #
    #     points[4] += step
    #     return se_transformation(points, step, min_emd, min_mat)
    pass


def loss_func(points):
    trans_mat = cv2.getPerspectiveTransform(*points)
    trans_cl = cv2.warpPerspective(cl_img, trans_mat, (width, height))
    trans_his = hist_methods.get_hist(trans_cl, ebsd_img, ref_freq_col, ref_freq_col - 50, ref_freq_col + 50)
    return calc_emd(trans_his)


def find_best_transformation():
    trans = opt.minimize(calc_emd, [[0, 0], [0, 1024], [1024, 0], [1024, 1024]])
    print(trans)


if __name__ == "__main__":
    # Load Images
    cl_img, ebsd_img, quant_image = hist_methods.load_images()
    # Get reference histogram and reference frequent colours
    ref_freq_col, ref_his = hist_methods.get_basic_data()
    height, width, ch = cl_img.shape

    find_best_transformation()

