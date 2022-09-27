import numpy as np
import cv2
import matplotlib.pyplot as plt
import hist_methods as hist

# Image Paths
ebsd_path = 'Sequence/EBSD_09-07-29_1415-13_Map1-2-3-4_corr-BC-IPF-GB_crop.png'
orig_cl_path = 'Sequence/cl.png'
truth_path = 'data/affine_reg.png'


def load_images():
    ebsd = cv2.imread(ebsd_path)
    cl = cv2.imread(orig_cl_path)
    return ebsd, cl


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


def plot_colour_scatter_by_mask(mask):
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    ebsd_img[mask > 0] = [0, 0, 0]
    truth_img[mask > 0] = [0, 0, 0]

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


if __name__ == "__main__":
    ebsd_img, cl_img = load_images()
    truth_img = cv2.imread(truth_path)
    # print(ebsd_img)
    quant = cv2.imread('quant13.png')
    freq_colour = hist.get_n_freq_colours(quant, 1)[0]
    mask = cv2.inRange(quant, freq_colour, freq_colour)
    plot_colour_scatter_by_mask(mask)
