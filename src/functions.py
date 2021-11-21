from pydicom import dcmread
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy

PLOT_OK = True


def get_image(data_dir, file_name, file_type="dcm"):
    if file_name:
        file_path = data_dir + "\\" + file_name
        if file_type == "dcm":
            img = dcmread(file_path)
        else:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    return -1


def rescale_image(img):
    return ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')


def crop(img, margin):
    return img[margin[0]:(img.shape[0] - margin[0]), margin[1]:(img.shape[1] - margin[1])]


def preprocess(dcm):
    spacing = dcm.ImagerPixelSpacing
    img = dcm.pixel_array
    img = rescale_image(img)
    margin = [round(x * 0.05) for x in img.shape]
    img = crop(img, margin)

    return img, spacing


def register(fixed, moving):
    # res = copy.copy(fixed)
    # res[res > 25] = 255
    # res[res < 25] = 0
    _, res = cv2.threshold(fixed, 25, 255, 0)

    if PLOT_OK:
        plt.figure()
        plt.imshow(res, 'gray')
        plt.show()

    # Find contours
    cnts = cv2.findContours(np.uint8(res), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Find maximum area contour
    areas = np.array([cv2.contourArea(cnts[i]) for i in range(len(cnts))])
    cnt = cnts[np.argmax(areas)]

    # Get rotated rectangle from outer contour
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))

    # Draw rotated rectangle on copy of img as result
    if PLOT_OK:
        res = cv2.cvtColor(fixed.copy(), cv2.COLOR_GRAY2RGB)
        cv2.drawContours(res, [box], 0, (0, 0, 255), -1)

        plt.figure()
        plt.imshow(res)
        plt.show()

    # Get angle from rotated rectangle
    k = rect[-1] > 45
    angle = -rect[-1] + k * 90

    # Resize atlas to rectangle size
    moving = cv2.resize(moving, (int(rect[1][k]), int(rect[1][1-k])))

    if PLOT_OK:
        plt.figure()
        plt.imshow(moving)
        plt.show()

    # Rotate atlas to rectangle orientation
    matrix = cv2.getRotationMatrix2D(center=[0, moving.shape[0]], angle=angle, scale=1)
    moving = cv2.warpAffine(src=moving, M=matrix, dsize=fixed.shape[1::-1])

    if PLOT_OK:
        plt.figure()
        plt.imshow(moving)
        plt.show()

    return moving
