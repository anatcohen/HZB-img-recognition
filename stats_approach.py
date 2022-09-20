import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.stats import wasserstein_distance
import hist_methods as histogram


# Paths
ebsd_path = 'Sequence/EBSD_09-07-29_1415-13_Map1-2-3-4_corr-BC-IPF-GB_crop.png'
real_truth_path = 'data/affine_reg.png'
orig_cl_path = 'Sequence/cl.png'


def create_trans_x_data(n=100):
    ret_arr = []
    step = height/(2*n)

    for i in range(n):
        trans_x_mat = np.float32([[1, 0, 1], [0, 1, -step*(i+1)]])
        trans_x_img = cv2.warpAffine(cl, trans_x_mat, (width, height))
        ret_arr.append(trans_x_img)

    return ret_arr


def get_histogram_by_colour(data, colour):
    ret_arr = []
    for img in data:
        img_hist = histogram.get_hist(cl, ebsd, colour, colour, colour)
        ret_arr.append(img_hist)
    return ret_arr


def get_wd(his_arr, orig_his):
    ret = []




if __name__ == "__main__":
    # get_basic_data()
    ebsd, truth, cl, quant = histogram.load_images()
    height, width, ch = cl.shape
    # freq_colours = histogram.get_n_freq_colours(quant, 6)
    trans_x = create_trans_x_data()
