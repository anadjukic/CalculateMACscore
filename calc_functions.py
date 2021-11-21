from PIL import Image
import cv2
import numpy as np
import pydicom as pyd
import copy
from matplotlib import pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

MAX = 255

# CLAHE parameters
CLIP = 2.5  # 1.0
KC = 5  # 8

# Mask parameters
BG_THR = 20

# Adaptive threshold
BIAS = -3  # -9
ADA_DIL = 3

# Edge detection
CU = 1.05  # 1.1
CL = 0.5  # 0.75
GRAD = True


def enhance_contrast(image, method="clahe"):
    clahe = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(KC, KC))
    return clahe.apply(image)


def create_mask(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # find outermost contour > eliminate it
    whole_foot = copy.copy(image)
    # binary thresholding, eliminating dark background
    whole_foot[whole_foot < BG_THR] = 0
    whole_foot[whole_foot > 0] = MAX
    foot_edge = whole_foot - cv2.morphologyEx(
        whole_foot, cv2.MORPH_ERODE, kernel, iterations=1
    )
    foot_edge[foot_edge < 0] = 0
    foot_edge = cv2.morphologyEx(
        foot_edge, cv2.MORPH_DILATE, kernel, iterations=5
    )

    # localizing the bone tissue using adaptive thresholding
    h, w = image.shape
    # calculating the adaptive kernel size as a function of image dims
    k = round((h + w) / 2 / 10) + (round((h + w) / 2 / 10) % 2 == 0)
    ada_thr = cv2.adaptiveThreshold(
        image, MAX, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, k, BIAS
    )
    # # erode for elimination of artery parts - unnecessary
    # ada_thr_dil = cv2.morphologyEx(
    #     ada_thr, cv2.MORPH_ERODE, kernel, iterations=5
    # )
    # dilation for covering the interior edges
    ada_thr_dil = cv2.morphologyEx(
        ada_thr, cv2.MORPH_DILATE, kernel, iterations=ADA_DIL
    )

    # safely remove low intensity content, as arteries are brighter
    low_int = np.zeros(image.shape, np.uint8)
    low_int[image < 0.22 * MAX] = MAX

    return np.uint8(
        np.float32(ada_thr_dil) + np.float32(foot_edge) + np.float32(low_int)
    )


def edge_detection(image):

    base, _ = cv2.threshold(image, 0, MAX, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    upper = CU * base
    lower = CL * base
    return cv2.Canny(image, lower, upper, L2gradient=GRAD)


