import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import seaborn as sns

# Image Paths
# ebsd_path = 'Sequence/EBSD_09-07-29_1415-13_Map1-2-3-4_corr-BC-IPF-GB_crop.png'
# orig_cl_path = 'Sequence/cl.png'
# truth_path = 'data/affine_reg.png'

# orange, yellow, green, light blue, dark blue, purple, pink
col_bound = np.array([[0, 20], [21, 37], [38, 80], [81, 100], [101, 133], [134, 140], [141, 179]])


# def plot_colour_scatter():
#     fig = plt.figure(figsize=(10, 7))
#     ax = plt.axes(projection="3d")
#
#     scatter_img_height, scatter_img_width, _ = ebsd_img.shape
#     b, g, r = cv2.split(ebsd_img)
#     b = np.ravel(b)
#     g = np.ravel(g)
#     r = np.ravel(r)
#
#     cl_b, cl_g, cl_r = cv2.split(truth_img)
#     cl_b = np.ravel(cl_b)
#     cl_g = np.ravel(cl_g)
#     cl_r = np.ravel(cl_r)
#
#     color = np.array(list(zip(cl_r, cl_g, cl_b))) / 255.0
#
#     ax.scatter3D(b, g, r, c=color)
#     ax.set_xlabel('Blue')
#     ax.set_ylabel('Green')
#     ax.set_zlabel('Red')
#
#     plt.show()


# Creates a histogram of 6 colour distribution side by side
def side_by_side(hsv_ebsd, truth_img, bin_num=20):
    row_num = 2
    col_num = round(len(col_bound)/row_num)
    fig, ax = plt.subplots(nrows=row_num, ncols=col_num, figsize=(15, 9))

    for i in range(len(col_bound)):
        plot_inf_fig(hsv_ebsd, truth_img, i, ax[i//col_num][i%col_num], bin_num)
        fig.tight_layout()
    return fig


# Returns a figure of 5 side by side histograms to plot in the GUI
def get_fig(ebsd_path, cl_path):
    ebsd_img = cv2.imread(ebsd_path)
    truth_img = cv2.imread(cl_path)

    hsv_ebsd = cv2.cvtColor(ebsd_img, cv2.COLOR_BGR2HSV)

    fig = side_by_side(hsv_ebsd, truth_img)
    return fig


# Returns an individual histogram by colour and it's mapping
def get_ind_figs(ebsd_path, colour_ind):
    ebsd_img = cv2.imread(ebsd_path)
    hsv_ebsd = cv2.cvtColor(ebsd_img, cv2.COLOR_BGR2HSV)

    map1, map2 = get_col_map(ebsd_img, hsv_ebsd, colour_ind)
    return map1, map2


# Plots an individual histogram depending on colour
def plot_inf_fig(hsv_ebsd, truth_img, col_ind, ax, bin_num=20):
    truth = truth_img.copy()
    inf = np.array([col_bound[col_ind][0], 0, 0])
    sup = np.array([col_bound[col_ind][1], 255, 255])
    mask = cv2.inRange(hsv_ebsd, inf, sup)
    truth[mask == 0] = [0, 0, 0]
    grey, _, _ = cv2.split(truth)
    grey = grey[grey != 0]

    n, bins, patches = ax.hist(grey, edgecolor='black', linewidth=1, bins=bin_num, density=True)

    for j, bar in enumerate(patches):
        x = ((bins[j] + bins[j + 1]) / 2) / 255
        bar.set_facecolor((x, x, x))

    colour = np.uint8([[[(col_bound[col_ind][1]+col_bound[col_ind][0])//2, 100, 255]]])

    colour = cv2.cvtColor(colour, cv2.COLOR_HSV2RGB)
    handle = lines.Line2D([], [], c=colour[0][0]/255, lw=15)
    ax.legend(handles=[handle])
    ax.set_xlabel('Luminescence Colour')
    ax.set_ylabel('Frequency')
    return ax


# Creates mask for input colour
def get_col_map(ebsd, hsv_ebsd, colour_ind):
    img1 = ebsd.copy()
    img2 = ebsd.copy()

    inf = np.array([col_bound[colour_ind][0], 0, 0])
    sup = np.array([col_bound[colour_ind][1], 255, 255])

    mask = cv2.inRange(hsv_ebsd, inf, sup)
    img1[mask > 0] = [0, 0, 0]
    img2[mask == 0] = [0, 0, 0]
    ebsd[mask > 0] = [0, 0, 0]

    return img1, img2
