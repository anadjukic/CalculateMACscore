from PIL import Image
import cv2
import numpy as np
import pydicom as pyd
import copy
# from matplotlib import pyplot as plt
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename

MAX = 255

def calculate_mac()

# img_dcm = pyd.dcmread(filename)
# img = img_dcm.pixel_array
# dx = img_dcm.ImagerPixelSpacing[0]  # in mm
# dy = img_dcm.ImagerPixelSpacing[1]  # in mm
#
# plt.figure()
# plt.imshow(img)
# plt.show()

imgP = Image.open(filename).convert("L")
img = np.array(imgP)

dx = 0.125
dy = 0.125

clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(25, 25))
# clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(11, 11))
img_proc = clahe.apply(img)
# img_proc = img

plt.figure()
plt.imshow(img)
# plt.show()

plt.figure()
plt.imshow(img_proc)
# plt.show()

# img = cv2.imread(
#     r"C:\Users\Administrator\Desktop\sample.jpg", cv2.IMREAD_GRAYSCALE
# )

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

whole_foot = copy.copy(img)
whole_foot[whole_foot < 20] = 0
whole_foot[whole_foot > 0] = MAX
foot_edge = whole_foot - cv2.morphologyEx(whole_foot, cv2.MORPH_ERODE, kernel,
                                          iterations=1)
# foot_edge = cv2.morphologyEx(whole_foot, cv2.MORPH_DILATE, kernel,
#                              iterations=1) - whole_foot
foot_edge[foot_edge < 0] = 0
foot_edge = cv2.morphologyEx(foot_edge, cv2.MORPH_DILATE, kernel, iterations=5)

plt.figure()
plt.imshow(whole_foot)
# plt.show()

plt.figure()
plt.imshow(foot_edge)
# plt.show()

h, w = img.shape
k = round((h + w) / 2 / 10) + (round((h + w) / 2 / 10) % 2 == 0)
ada_thr = cv2.adaptiveThreshold(
    img, MAX, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, k, -9  # 251,
    # -3  # 191, -3
)

# ada_thr = cv2.adaptiveThreshold(
#     img, MAX, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0
# )

ada_thr_dil = cv2.morphologyEx(ada_thr, cv2.MORPH_ERODE, kernel, iterations=5)
ada_thr_dil = cv2.morphologyEx(ada_thr, cv2.MORPH_DILATE, kernel, iterations=5)

low_int = np.zeros(img.shape, np.uint8)
low_int[img < 0.22 * MAX] = MAX

mask = np.uint8(
    np.float32(ada_thr_dil) + np.float32(foot_edge) + np.float32(low_int)
)

plt.figure()
plt.imshow(ada_thr)
# plt.show()

a = 1.1
b = 0.75
upper, _ = cv2.threshold(img_proc, 0, MAX, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
upper = a * upper
lower = b * upper
edge = cv2.Canny(img_proc, lower, upper, L2gradient=True)

plt.figure()
plt.imshow(edge)
# plt.show()

max_img = np.ones(mask.shape, np.uint8) * 255
mask_neg = np.uint8((max_img - mask) / MAX)
artery_cand = mask_neg * img

plt.figure()
plt.imshow(artery_cand)
# plt.show()

artery_area = edge - mask
artery_area[artery_area < 0] = 0
print(np.unique(artery_area))

plt.figure()
plt.imshow(artery_area)

# artery_area = cv2.morphologyEx(
#     artery_area, cv2.MORPH_DILATE, kernel, iterations=3
# )

cnts, _ = cv2.findContours(
    artery_area, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
)

valid_contours = np.zeros(img.shape, np.uint8)
for c in cnts:
    area = cv2.contourArea(c)
    if area > 50:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # a = np.sqrt((box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2)
        # b = np.sqrt((box[0][0]-box[3][0])**2 + (box[0][1]-box[3][1])**2)
        if min(a, b) / max(a, b) < 0.5:
            cv2.drawContours(valid_contours, [c], 3, MAX, 3)  #,
            # thickness=-1

plt.figure()
plt.imshow(valid_contours)
plt.show()
