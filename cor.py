import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import hist_methods
import seaborn as sns

# Image Paths
ebsd_path = 'Sequence/EBSD_09-07-29_1415-13_Map1-2-3-4_corr-BC-IPF-GB_crop.png'
orig_cl_path = 'Sequence/cl.png'
truth_path = 'data/affine_reg.png'


def plot_colour_scatter():
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    scatter_img_height, scatter_img_width, _ = ebsd_img.shape
    b, g, r = cv2.split(ebsd_img)
    b = np.ravel(b)
    g = np.ravel(g)
    r = np.ravel(r)

    cl_b, cl_g, cl_r = cv2.split(truth_img)
    cl_b = np.ravel(cl_b)
    cl_g = np.ravel(cl_g)
    cl_r = np.ravel(cl_r)

    color = np.array(list(zip(cl_r, cl_g, cl_b))) / 255.0

    ax.scatter3D(b, g, r, c=color)
    ax.set_xlabel('Blue')
    ax.set_ylabel('Green')
    ax.set_zlabel('Red')

    plt.show()


def cl_dist_by_colour(truth, colour, j, bin_num, compare=False):
    if compare:
        y, _, _ = cv2.split(truth)
        y = y[y != 0]
        plt.hist(y, bin_num, alpha=0.5, label='Full Image')
        plt.legend(loc='upper right')

    mask = cv2.inRange(quant, colour, colour)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=2)
    truth[mask == 0] = [0, 0, 0]

    grey, _, _ = cv2.split(truth)
    grey = grey[grey != 0]

    n, bins, patches = plt.hist(grey, edgecolor='black', linewidth=1, bins=bin_num)

    for i, bar in enumerate(patches):
        x = ((bins[i] + bins[i + 1]) / 2) / 255
        bar.set_facecolor((x, x, x))

    handle = [lines.Line2D([], [], c=np.flip(colour)/255, lw=15)]

    plt.legend(handles=handle)
    plt.xlabel('Luminescence Colour')
    plt.ylabel('Frequency')
    name = 'Histograms/Final/full_colour_' + str(j) if compare else 'Histograms/Final/colour_' + str(j)
    plt.savefig(name)
    # plt.show()


def get_colours_in_mask(mask, img, display=False):
    img[mask == 0] = [0, 0, 0]
    coord = np.any(img != [0, 0, 0], axis=-1)
    colours = img[coord]

    # Display colours
    if display:
        un = np.unique(colours, axis=0)/255
        sns.palplot(un)
        plt.show()

    return colours


# Creates a histogram of 5 colour distribution side by side
def side_by_side(bin_num, compare=False):
    truth0 = truth_img.copy()
    truth1 = truth_img.copy()
    truth2 = truth_img.copy()
    truth3 = truth_img.copy()
    truth4 = truth_img.copy()

    mask0 = cv2.inRange(quant, freq_colours[0], freq_colours[0])
    mask1 = cv2.inRange(quant, freq_colours[1], freq_colours[1])
    mask2 = cv2.inRange(quant, freq_colours[2], freq_colours[2])
    mask3 = cv2.inRange(quant, freq_colours[3], freq_colours[3])
    mask4 = cv2.inRange(quant, freq_colours[4], freq_colours[4])

    truth0[mask0 == 0] = [0, 0, 0]
    truth1[mask1 == 0] = [0, 0, 0]
    truth2[mask2 == 0] = [0, 0, 0]
    truth3[mask3 == 0] = [0, 0, 0]
    truth4[mask4 == 0] = [0, 0, 0]

    grey0, _, _ = cv2.split(truth0)
    grey1, _, _ = cv2.split(truth1)
    grey2, _, _ = cv2.split(truth2)
    grey3, _, _ = cv2.split(truth3)
    grey4, _, _ = cv2.split(truth4)

    grey0 = grey0[grey0 != 0]
    grey1 = grey1[grey1 != 0]
    grey2 = grey2[grey2 != 0]
    grey3 = grey3[grey3 != 0]
    grey4 = grey4[grey4 != 0]

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=5, figsize=(30, 10))

    handle0 = lines.Line2D([], [], c=np.array(freq_colours[0]) / 255, lw=15)
    handle1 = lines.Line2D([], [], c=np.array(freq_colours[1]) / 255, lw=15)
    handle2 = lines.Line2D([], [], c=np.array(freq_colours[2]) / 255, lw=15)
    handle3 = lines.Line2D([], [], c=np.array(freq_colours[3]) / 255, lw=15)
    handle4 = lines.Line2D([], [], c=np.array(freq_colours[4]) / 255, lw=15)

    n_0, bins_0, patches_0 = ax0.hist(grey0, edgecolor='black', linewidth=1, bins=bin_num, density=True)
    n_1, bins_1, patches_1 = ax1.hist(grey1, edgecolor='black', linewidth=1, bins=bin_num, density=True)
    n_2, bins_2, patches_2 = ax2.hist(grey2, edgecolor='black', linewidth=1, bins=bin_num, density=True)
    n_3, bins_3, patches_3 = ax3.hist(grey3, edgecolor='black', linewidth=1, bins=bin_num, density=True)
    n_4, bins_4, patches_4 = ax4.hist(grey4, edgecolor='black', linewidth=1, bins=bin_num, density=True)

    for i, bar in enumerate(patches_0):
        x = ((bins_0[i] + bins_0[i + 1]) / 2) / 255
        bar.set_facecolor((x, x, x))
    for i, bar in enumerate(patches_1):
        x = ((bins_1[i] + bins_1[i + 1]) / 2) / 255
        bar.set_facecolor((x, x, x))
    for i, bar in enumerate(patches_2):
        x = ((bins_2[i] + bins_2[i + 1]) / 2) / 255
        bar.set_facecolor((x, x, x))
    for i, bar in enumerate(patches_3):
        x = ((bins_3[i] + bins_3[i + 1]) / 2) / 255
        bar.set_facecolor((x, x, x))
    for i, bar in enumerate(patches_4):
        x = ((bins_4[i] + bins_4[i + 1]) / 2) / 255
        bar.set_facecolor((x, x, x))

    if compare:
        y0, _, _ = cv2.split(truth0)
        y1, _, _ = cv2.split(truth1)
        y2, _, _ = cv2.split(truth2)
        y3, _, _ = cv2.split(truth3)
        y4, _, _ = cv2.split(truth4)

        y0 = y0[y0 != 0]
        y1 = y0[y0 != 1]
        y2 = y0[y0 != 2]
        y3 = y0[y0 != 3]
        y4 = y0[y0 != 4]

        ax0.hist(y0, bin_num, alpha=0.5)
        ax1.hist(y1, bin_num, alpha=0.5)
        ax2.hist(y2, bin_num, alpha=0.5)
        ax3.hist(y3, bin_num, alpha=0.5)
        ax4.hist(y4, bin_num, alpha=0.5)

    ax0.legend(handles=[handle0])
    ax1.legend(handles=[handle1])
    ax2.legend(handles=[handle2])
    ax3.legend(handles=[handle3])
    ax4.legend(handles=[handle4])

    fig.tight_layout()
    plt.xlabel('Luminescence Colour')
    plt.ylabel('Frequency')
    name = 'Histograms/Final/full_colour' if compare else 'Histograms/Final/colour'
    plt.savefig(name)
    plt.show()


# Creates mask for input colour
def create_mask(colour):
    mask = cv2.inRange(quant, colour, colour)
    mask = cv2.erode(mask, kernel, iterations=2)
    return mask


if __name__ == "__main__":
    ebsd_img = cv2.imread(ebsd_path)
    truth_img = cv2.imread(truth_path)
    quant = hist_methods.quantise_img(ebsd_img, 30)

    freq_colours = hist_methods.get_n_freq_colours(quant, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    j = 4
    colour = freq_colours[j]
    cl_dist_by_colour(truth_img, colour, 0, 20)

    # un = np.unique(freq_colours, axis=0)/255
    # sns.palplot(un)
    # plt.show()

    mask = create_mask(colour)
    ebsd_img[mask > 0] = [0, 0, 0]
    cv2.imwrite('Histograms/Final/mask' + str(j) + '.png', ebsd_img)

    # bins = 20
    # for i, col in enumerate(freq_colours):
    #     cl_dist_by_colour(truth_img, col, i, bins, True)
    # cl_dist_by_colour(truth_img, freq_colours[0], 0, bins, True)
    # side_by_side(20)
