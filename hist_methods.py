import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import pandas as pd


# Paths
# ebsd_path = 'data/EBSD_09-07-29_1415-13_Map1 2 3 4_corr-BC   IPF   GB_crop.tif'
ebsd_path = 'Sequence/EBSD_09-07-29_1415-13_Map1-2-3-4_corr-BC-IPF-GB_crop.png'
real_truth_path = 'data/affine_reg.png'
orig_cl_path = 'data/CL1415-13_8K_1280nm_8kV_SS33_3000x_900V_1mm_40us_1024x1024.bmp'

white = [225, 225, 225]
black = [0, 0, 0]


# Reduces the amount of colours in image to k colours
def quantise_img(image, k):
    # source: https://www.analyticsvidhya.com/blog/2021/07/colour-quantization-using-k-means-clustering-and-opencv/
    i = np.float32(image).reshape(-1, 3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    ret, label, center = cv2.kmeans(i, k, None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img


# Returns a list of n most frequent colours in img
def get_n_freq_colours(image, n=1):
    colours, count = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    n = min([len(colours), n])
    n_colours = []

    while len(n_colours) < n:
        ind = count.argmax()
        freq_colour = np.array(colours[ind])
        # Deleting freq colour from colours and count
        colours = colours.flatten()
        colours = np.delete(colours, [ind*3, ind*3 + 1, ind*3 + 2])
        colours = np.reshape(colours, (-1, 3))
        count = np.delete(count, ind)
        if not np.array_equal(freq_colour, black):
            n_colours.append(freq_colour)

    return np.array(n_colours)


def create_histogram(arr, bin_num, colour):
    n, bins, patches = plt.hist(arr, edgecolor='black', linewidth=1, bins=bin_num)

    for i, bar in enumerate(patches):
        x = ((bins[i] + bins[i + 1])/2)/255
        bar.set_facecolor((x, x, x))

    handle = lines.Line2D([], [], c=colour, lw=15)
    plt.legend(handles=[handle])
    plt.show()


def get_mask_coordinates(image, inf, sup):
    # Fix range
    # inf = (np.vectorize(lambda i: max(0, i))(inf))
    # sup = (np.vectorize(lambda i: min(i, 255))(sup))

    mask = cv2.inRange(image, inf, sup)
    coords = np.column_stack(np.where(mask > 0))
    return coords


# IMPORTANT: So far returns 1D array with every element being R,B,G values of pixel all at once
#            eg. [1,2,3] = [[1,1,1], [2,2,2], [3,3,3]]
#            * bc cl is a greyscale, every pixel should have equal R,B,G values
#            * could cause issues later if cl pixels don't work like that

# Gets colours of specific coordinates in a greyscale image
def get_colours_by_coords(img, coords):
    flat_img = img.flatten()
    flat_coords = coords.flatten()
    col = flat_coords[::2]
    row = flat_coords[1::2]
    flat_coords = col*row
    return flat_img[flat_coords]


# Returns histogram and freq colours
def get_hist(cl_img, ebsd_img, freq_colour, min_colour, max_colour):
    coordinates = get_mask_coordinates(ebsd_img, min_colour, max_colour)
    img_hist = get_colours_by_coords(cl_img, coordinates)
    img_hist = img_hist[img_hist != 0]
    # create_histogram(img_hist, 255, freq_colour/255)
    return img_hist


# Returns basic data; frequent colour and histogram
def get_basic_data():
    ebsd_img, truth_img, quant_img = load_images()
    freq_colour = get_n_freq_colours(quant_img)[0]
    return freq_colour, get_hist(truth_img, ebsd_img, freq_colour, freq_colour - 30, freq_colour + 30)


# Loads ebsd, true registration images and also creates quantised image of ebsd
def load_images():
    ebsd_img = cv2.imread(ebsd_path)
    truth_img = cv2.imread(real_truth_path)
    cl_img = cv2.imread(orig_cl_path)
    # quant_img = quantise_img(ebsd_img, 13)
    quant_img = cv2.imread('quant13.png')
    return ebsd_img, truth_img, cl_img, quant_img


if __name__ == "__main__":
    # get_basic_data()
    ebsd, truth, cl, quant = load_images()
    freq_colours = get_n_freq_colours(quant, 6)

    for j in range(6):
        colour = freq_colours[j]
        coordinates = get_mask_coordinates(quant, colour, colour)
        cl_hist = get_colours_by_coords(cl, coordinates)
        truth_hist = get_colours_by_coords(truth, coordinates)
        cl_hist = cl_hist[cl_hist != 0]
        truth_hist = truth_hist[truth_hist != 0]

        fig, axes = plt.subplots(1, 2)
        n_bins = 150
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        handle = lines.Line2D([], [], c=np.array(colour)/255, lw=15)

        n_0, bins_0, patches_0 = ax0.hist(cl_hist, edgecolor='black', linewidth=1, bins=n_bins)
        ax0.set_title('Unregistered Cl')
        n_1, bins_1, patches_1 = ax1.hist(truth_hist, edgecolor='black', linewidth=1, bins=n_bins)
        ax1.set_title('Registered Cl')

        for i, bar in enumerate(patches_0):
            x = ((bins_0[i] + bins_0[i + 1]) / 2) / 255
            bar.set_facecolor((x, x, x))
        for i, bar in enumerate(patches_1):
            x = ((bins_1[i] + bins_1[i + 1]) / 2) / 255
            bar.set_facecolor((x, x, x))

        ax0.legend(handles=[handle])
        ax1.legend(handles=[handle])

        fig.tight_layout()
        name = 'Histograms/colour' + str(j+1) + '.png'
        plt.savefig(name)
        # plt.show()


