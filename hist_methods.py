import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as lines


# Paths
# ebsd_path = 'data/EBSD_09-07-29_1415-13_Map1 2 3 4_corr-BC   IPF   GB_crop.tif'
ebsd_path = 'Sequence/EBSD_09-07-29_1415-13_Map1-2-3-4_corr-BC-IPF-GB_crop.png'
real_truth_path = 'data/fix_affine_reg.png'


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


# Finds the most frequent colour in image
# To get second colour 1. get max 2. remove from array 3. find new max
def most_freq_colour(image):
    colours, count = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    return np.array(colours[count.argmax()])


def create_histogram(arr, bin_num, colour):
    n, bins, patches = plt.hist(arr, edgecolor='black', linewidth=1, bins=bin_num)
    for i, bar in enumerate(patches):
        x = ((bins[i] + bins[i + 1])/2)/255
        bar.set_facecolor((x, x, x))

    handle = lines.Line2D([], [], c=colour, lw=15)
    plt.legend(handles=[handle])
    plt.show()

    return zip(n, bins, patches)


def get_mask_coordinates(image, inf, sup):
    # Fix range
    inf = (np.vectorize(lambda i: max(0, i))(inf))
    sup = (np.vectorize(lambda i: min(i, 255))(sup))

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
    # his = create_histogram(img_hist, 40, freq_colour/255)
    return img_hist


# Returns basic data; frequent colour and histogram
def get_basic_data():
    ebsd_img, truth_img, quant_img = load_images()
    freq_colour = most_freq_colour(quant_img)
    return freq_colour, get_hist(truth_img, ebsd_img, freq_colour, freq_colour - 30, freq_colour + 30)


# Loads ebsd, true registration images and also creates quantised image of ebsd
def load_images():
    ebsd_img = cv2.imread(ebsd_path)
    truth_img = cv2.imread(real_truth_path)
    # quant_img = quantise_img(ebsd_img, 13)
    quant_img = cv2.imread('quant13.png')
    return ebsd_img, truth_img, quant_img


if __name__ == "__main__":
    get_basic_data()
