import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from src.functions import *


# if __name__ == '__main__':
#     # data_dir = os.chdir('C:/Users/Milica/OneDrive - Universitaet Bern/EHH 2021/pngall')
#     data_dir = os.getcwd() + r"\data"
#     # data_list = [file for file in os.listdir(data_dir) if file.endswith(('.dcm', '.DCM'))]
#     # data_path = data_dir + "\\" + data_list[0]
#
#     # dcm = dcmread(data_path)
#     # img = dcm.pixel_array
#     # img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
#
#     file_path = data_dir + "\\" + "crv.png"
#     img = cv2.imread(file_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
#
#     plt.figure()
#     plt.imshow(img, cmap=plt.cm.gray)
#     plt.show()
#
#     pixel_values = np.float32(img.reshape((-1, 1)))
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#     k = 4
#     _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#
#     centers = np.uint8(centers)
#     labels = labels.flatten()
#
#     seg_img = centers[labels.flatten()]
#     seg_img = seg_img.reshape(img.shape)
#
#     plt.figure()
#     plt.imshow(seg_img)
#     plt.show()

TOP_VIEW_FILE = "top_view.dcm"
SIDE_VIEW_FILE = "side_view.dcm"
TOP_VIEW_ATLAS = "top_view_atlas.png"
SIDE_VIEW_ATLAS = "side_view_atlas.png"
TOP_VIEW_REG = "top_view_reg.png"
SIDE_VIEW_REG = "side_view_reg.png"
PLOT_OK = True

# Load files
os.chdir(r"C:\\Users\\Milica\\Develop\\sad-quant")
data_dir = os.getcwd() + "\\data"

# Load x-rays
dcm = get_image(data_dir, TOP_VIEW_FILE, "dcm")
top_view, _ = preprocess(dcm)

dcm = get_image(data_dir, SIDE_VIEW_FILE, "dcm")
side_view, _ = preprocess(dcm)

if PLOT_OK:
    plt.figure()
    plt.imshow(top_view, 'gray')
    plt.title("Top view")
    plt.show()

    plt.figure()
    plt.imshow(side_view, 'gray')
    plt.title("Side view")
    plt.show()

# Load atlases
top_view_atlas = get_image(data_dir, TOP_VIEW_ATLAS, "png")
side_view_atlas = get_image(data_dir, SIDE_VIEW_ATLAS, "png")

if PLOT_OK:
    plt.figure()
    plt.imshow(top_view_atlas, 'gray')
    plt.title("Top view")
    plt.show()

    plt.figure()
    plt.imshow(side_view_atlas, 'gray')
    plt.title("Side view")
    plt.show()

top_view_atlas = register(top_view, top_view_atlas)
cv2.imwrite(data_dir + "\\" + TOP_VIEW_REG, top_view_atlas)

side_view_atlas = register(side_view, side_view_atlas)
cv2.imwrite(data_dir + "\\" + SIDE_VIEW_REG, side_view_atlas)

