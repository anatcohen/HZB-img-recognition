# Input: cl, ebsd images
# Get initial histogram
# Create loss function

import numpy as np
import cv2
from pyemd import emd
import fixed_hist

# Problems/ Questions:
# What is the distance matrix in pyemd?


def best_transformation(mask_img, corners, min_emd, trans_mat, step):
    # Stop condition

    # north-west, north-east, south-west, south-east corners
    nw, ne, sw, se = corners

    trans_matrix = cv2.getAffineTransform(pts1, pts2)
    trans_img = cv2.warpAffine(cl_img, trans_matrix, (cols, rows))
    trans_his = fixed_hist.get_colours_by_coords(trans_img, mask_img)

    new_emd = calc_emd(trans_his)

    # Replace min emd
    if new_emd < min_emd:
        min_emd = new_emd
        trans_mat = trans_matrix

    # New points in range
    if nw in [rows, cols] and ne in [rows, cols] and sw in [rows, cols] and se in [rows, cols]:
        # Update corners
        return best_transformation(corners, min_emd, step)

    # return best_transformation(pts1, pts2 + step, step)
    return trans_mat


# Cite Ofir Pele and Michael Werman's pyemd paper
# What is the distance matrix?
def calc_emd(his1):
    distance_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
    return emd(his1, his1, distance_matrix)


if __name__ == "__main__":
    # Load Images
    # cl_img, ebsd_img, quant_image = fixed_hist.load_images()
    # Get reference histogram and reference frequent colours
    ref_freq_col, ref_his = fixed_hist.get_basic_data()
    # mask_img = fixed_hist.get_mask_coordinates(ebsd_img, ref_freq_col - 30, ref_freq_col + 30)

    # rows, cols, ch = cl_img.shape
    # pts1 = np.float32([[0, 0], [rows, 0], [rows, cols], [0, cols]])
    # pts2 = np.float32([[0, 100], [rows - 100, 0], [rows, cols], [0, cols]])

    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # warp = cv2.warpPerspective(cl_img, M, (cols, rows))

    cl = cv2.imread('cl.png')
    ebsd = cv2.imread('ebsd.bmp')
    print(ref_freq_col)
    mask_img = fixed_hist.get_mask_coordinates(ebsd, ref_freq_col - 50, ref_freq_col + 50)
    coords = np.column_stack(np.where(mask_img > 0))

    cv2.imshow('', mask_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
