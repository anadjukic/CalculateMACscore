import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from src.functions import *
from src.calc_functions import *


TOP_VIEW_FILE = "top_view.dcm"
SIDE_VIEW_FILE = "side_view.dcm"
TOP_VIEW_ATLAS = "top_view_atlas.png"
SIDE_VIEW_ATLAS = "side_view_atlas_1.png"
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

top_view_reg = register(top_view, top_view_atlas)
# cv2.imwrite(data_dir + "\\" + TOP_VIEW_REG, top_view_atlas)

side_view_reg = register(side_view, side_view_atlas)
# cv2.imwrite(data_dir + "\\" + SIDE_VIEW_REG, side_view_atlas)

## --------------------------------------------------------------------------------------
# TOP VIEW
view = top_view_reg
values, counts = np.unique(view, return_counts=True)
top_view_areas = values[np.argpartition(-counts, kth=4)[:4]]

# cv2.imwrite('D_top_view.png', top_view)
# cv2.imwrite('D_top_view_reg.png', top_view_reg)

view = top_view
non_artery_area = create_mask(view)
if PLOT_OK:
    plt.figure()
    plt.imshow(non_artery_area, 'gray')
    plt.show()

edges = edge_detection(view)
if PLOT_OK:
    plt.figure()
    plt.imshow(edges, 'gray')
    plt.show()

cv2.imwrite('edges_top_view.png', edges)

non_artery_area[non_artery_area > 0] = 255
artery_area = 255 - non_artery_area
artery_edges = edges * artery_area

cv2.imwrite('artery_area_top_view.png', artery_area)

if PLOT_OK:
    plt.figure()
    plt.imshow(artery_area, 'gray')
    plt.show()

    plt.figure()
    plt.imshow(artery_edges, 'gray')
    plt.show()

# Arteries
a1 = artery_edges.copy()
a2 = artery_edges.copy()
a3 = artery_edges.copy()

a1[top_view_reg != top_view_areas[1]] = 0
a2[top_view_reg != top_view_areas[2]] = 0
a3[top_view_reg != top_view_areas[3]] = 0

a = (a1 | a2 | a3) * 255
cv2.imwrite('arteries_top_view.png', a)

plt.figure()
plt.imshow(a, 'gray')
plt.show()

###########
# SIDE_VIEW

view = side_view_reg
values, counts = np.unique(view, return_counts=True)
side_view_areas = values[np.argpartition(-counts, kth=3)[:3]]

cv2.imwrite('D_side_view.png', side_view)
cv2.imwrite('D_side_view_reg.png', side_view_reg)

view = side_view
non_artery_area = create_mask(view)
if PLOT_OK:
    plt.figure()
    plt.imshow(non_artery_area, 'gray')
    plt.show()

edges = edge_detection(view)
if PLOT_OK:
    plt.figure()
    plt.imshow(edges, 'gray')
    plt.show()

cv2.imwrite('edges_side_view.png', edges)

non_artery_area[non_artery_area > 0] = 255
artery_area = 255 - non_artery_area
artery_edges = edges * artery_area

cv2.imwrite('artery_area_side_view.png', artery_area)

if PLOT_OK:
    plt.figure()
    plt.imshow(artery_area, 'gray')
    plt.show()

    plt.figure()
    plt.imshow(artery_edges)
    plt.show()

# Arteries
a1 = artery_edges.copy()
a2 = artery_edges.copy()

a1[side_view_reg != side_view_areas[1]] = np.uint8(0)
a2[side_view_reg != side_view_areas[2]] = np.uint8(0)

a = (a1 | a2) * 255
cv2.imwrite('arteries_side_view.png', a)

plt.figure()
plt.imshow(a, 'gray')
plt.show()

# regions = [a1, a2]
# ind = 0
# for r in regions:
#     ind += 1
#     cnts, _ = cv2.findContours(r, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
#     MAX = 255
#     valid_contours = np.zeros(r.shape, np.uint8)
#     l = 0
#     for c in cnts:
#         area = cv2.contourArea(c)
#     # if area > 0:
#         if cv2.arcLength(c, True) > 10:
#             ((cx, cy), (a, b), angle) = cv2.minAreaRect(c)
#         # box = cv2.boxPoints(rect)
#         # box = np.int0(box)
#         # a = np.sqrt((box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2)
#         # b = np.sqrt((box[0][0]-box[3][0])**2 + (box[0][1]-box[3][1])**2)
#             if min(a, b) / max(a, b) < 0.5:
#                 cv2.drawContours(valid_contours, [c], 0, MAX, -1)
#                 l += np.sqrt(a**2 + b**2) / 2
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     crvici = cv2.morphologyEx(
#         valid_contours, cv2.MORPH_DILATE, kernel, iterations=1
#     )
#     from PIL import Image
#     temp = Image.fromarray(crvici)
#     path = r"C:\Users\Milica\Desktop\crv" + str(ind) + ".jpg"
#     # temp.save(temp, path)
#     plt.figure()
#     plt.imshow(crvici)
#     plt.show()
