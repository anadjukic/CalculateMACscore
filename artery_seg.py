from PIL import Image
import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

MAX = 255

# impath = r"C:\Users\Administrator\Desktop\sample.jpg"
impath = r"C:\Users\Administrator\Desktop\sample3.png"
imgP = Image.open(impath).convert("L")
img = np.array(imgP)

clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(9,9))
clh = clahe.apply(img)

plt.figure()
plt.imshow(img)
plt.show()

# img = cv2.imread(
#     r"C:\Users\Administrator\Desktop\sample.jpg", cv2.IMREAD_GRAYSCALE
# )

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

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
plt.show()

plt.figure()
plt.imshow(foot_edge)
plt.show()

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

mask = ada_thr_dil + foot_edge

plt.figure()
plt.imshow(ada_thr)
plt.show()

a = 1.0
b = 0.6
upper, _ = cv2.threshold(clh, 0, MAX, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
upper = a * upper
lower = b * upper
edge = cv2.Canny(clh, lower, upper, L2gradient=True)

plt.figure()
plt.imshow(edge)
plt.show()

artery = edge - mask
artery[artery < 0] = 0

plt.figure()
plt.imshow(artery)
plt.show()

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
# adap1 = cv2.morphologyEx()