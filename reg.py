# Input: cl, ebsd images
# Get initial histogram
# Create loss function

import numpy as np
import cv2
import fixed_hist

# Image paths
cl_path = ''
ebsd_path = ''

# Load Images
cl_img = cv2.imread(cl_path)
ebsd_img = cv2.imread(ebsd_path)

# Reference histogram and frequent colours
ref_his, ref_freq_colours = fixed_hist.get_data()
