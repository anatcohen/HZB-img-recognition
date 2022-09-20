import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.stats import wasserstein_distance

# Paths
ebsd_path = 'Sequence/EBSD_09-07-29_1415-13_Map1-2-3-4_corr-BC-IPF-GB_crop.png'
real_truth_path = 'data/affine_reg.png'
orig_cl_path = 'Sequence/cl.png'

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


def side_by_side_random():
    random1 = cv2.imread('data/random1.png')
    random2 = cv2.imread('data/random2.png')
    random3 = cv2.imread('data/random3.png')
    random4 = cv2.imread('data/random4.png')
    # height, width, ch = cl.shape

    # af_1 = np.float32([[1, 0, 1], [0, 1, height - 1000]])
    # af_2 = np.float32([[1, 0, 1], [0, 1, height - 900]])
    # af_3 = np.float32([[1, 0, 1], [0, 1, height - 800]])
    # af_4 = np.float32([[1, 0, 1], [0, 1, height - 700]])
    # af_5 = np.float32([[1, 0, 1], [0, 1, height - 600]])

    # random1 = cv2.warpAffine(cl, af_1, (width, height))
    # random2 = cv2.warpAffine(cl, af_2, (width, height))
    # random3 = cv2.warpAffine(cl, af_3, (width, height))
    # random4 = cv2.warpAffine(cl, af_4, (width, height))
    # random4 = cv2.warpAffine(orig_img, af_5, (width, height))

    for j in range(6):
        colour = freq_colours[j]
        coordinates = get_mask_coordinates(quant, colour, colour)

        cl_hist = get_colours_by_coords(cl, coordinates)
        random1_hist = get_colours_by_coords(random1, coordinates)
        random2_hist = get_colours_by_coords(random2, coordinates)
        random3_hist = get_colours_by_coords(random3, coordinates)
        random4_hist = get_colours_by_coords(random4, coordinates)
        truth_hist = get_colours_by_coords(truth, coordinates)

        cl_hist = cl_hist[cl_hist != 0]
        random1_hist = random1_hist[random1_hist != 0]
        random2_hist = random2_hist[random2_hist != 0]
        random3_hist = random3_hist[random3_hist != 0]
        random4_hist = random4_hist[random4_hist != 0]
        truth_hist = truth_hist[truth_hist != 0]

        fig, axes = plt.subplots(1, 6)
        n_bins = 20
        fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=6, figsize=(30, 10))
        handle = lines.Line2D([], [], c=np.array(colour)/255, lw=15)

        n_0, bins_0, patches_0 = ax0.hist(cl_hist, edgecolor='black', linewidth=1, bins=n_bins, density=True)
        ax0.set_title('Original Cl')
        n_1, bins_1, patches_1 = ax1.hist(random1_hist, edgecolor='black', linewidth=1, bins=n_bins, density=True)
        ax1.set_title('Random 1 Cl')
        n_2, bins_2, patches_2 = ax2.hist(random2_hist, edgecolor='black', linewidth=1, bins=n_bins, density=True)
        ax2.set_title('Random 2 Cl')
        n_3, bins_3, patches_3 = ax3.hist(random3_hist, edgecolor='black', linewidth=1, bins=n_bins, density=True)
        ax3.set_title('Random 3 Cl')
        n_4, bins_4, patches_4 = ax4.hist(random4_hist, edgecolor='black', linewidth=1, bins=n_bins, density=True)
        ax4.set_title('Random 4 Cl')
        n_5, bins_5, patches_5 = ax5.hist(truth_hist, edgecolor='black', linewidth=1, bins=n_bins, density=True)
        ax5.set_title('Registered Cl')

        wd0 = str("{:.2f}".format(wasserstein_distance(cl_hist, truth_hist)))
        wd1 = str("{:.2f}".format(wasserstein_distance(random1_hist, truth_hist)))
        wd2 = str("{:.2f}".format(wasserstein_distance(random2_hist, truth_hist)))
        wd3 = str("{:.2f}".format(wasserstein_distance(random3_hist, truth_hist)))
        wd4 = str("{:.2f}".format(wasserstein_distance(random4_hist, truth_hist)))
        wd5 = str("{:.2f}".format(wasserstein_distance(truth_hist, truth_hist)))

        ax0.set(xlabel="Wasserstein Distance: " + wd0)
        ax1.set(xlabel="Wasserstein Distance: " + wd1)
        ax2.set(xlabel="Wasserstein Distance: " + wd2)
        ax3.set(xlabel="Wasserstein Distance: " + wd3)
        ax4.set(xlabel="Wasserstein Distance: " + wd4)
        ax5.set(xlabel="Wasserstein Distance: " + wd5)

        for i, bar in enumerate(patches_0):
            x = ((bins_0[i] + bins_0[i + 1]) / 2) / 255
            bar.set_facecolor((x, x, x))
        for i, bar in enumerate(patches_1):
            x = ((bins_1[i] + bins_1[i + 1]) / 2) / 255
            bar.set_facecolor((x, x, x))
        for i, bar in enumerate(patches_2):
            x = ((bins_1[i] + bins_1[i + 1]) / 2) / 255
            bar.set_facecolor((x, x, x))
        for i, bar in enumerate(patches_3):
            x = ((bins_1[i] + bins_1[i + 1]) / 2) / 255
            bar.set_facecolor((x, x, x))
        for i, bar in enumerate(patches_4):
            x = ((bins_1[i] + bins_1[i + 1]) / 2) / 255
            bar.set_facecolor((x, x, x))
        for i, bar in enumerate(patches_5):
            x = ((bins_1[i] + bins_1[i + 1]) / 2) / 255
            bar.set_facecolor((x, x, x))

        ax0.legend(handles=[handle])
        ax1.legend(handles=[handle])
        ax2.legend(handles=[handle])
        ax3.legend(handles=[handle])
        ax4.legend(handles=[handle])
        ax5.legend(handles=[handle])

        fig.tight_layout()
        # name = 'Transpose Histograms/Translation X/' + str(j+1) + '.png'
        name = 'Random Histograms/colour' + str(j+1) + '.png'
        plt.savefig(name)
        # plt.show()


if __name__ == "__main__":
    # get_basic_data()
    ebsd, truth, cl, quant = load_images()
    freq_colours = get_n_freq_colours(quant, 6)
    side_by_side_random()

