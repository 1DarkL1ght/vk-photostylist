import cv2
import numpy as np
import time

image = cv2.imread("src/vk_photostylist/cv2_modules/cat.jpg")

def detect_edges(channel: np.ndarray | cv2.UMat):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return sobel

def findSignificantContours(
    edgeImg: np.ndarray | cv2.UMat,
):
    contours, heirarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    level1 = []
    for i, data in enumerate(heirarchy[0]):
        if data[3] == -1:
            data = np.insert(data, 0, [i])
            level1.append(data)
    significant = []
    tooSmall = edgeImg.size * 10 / 100
    for tupl in level1:
        contour = contours[tupl[0]]
        area = cv2.contourArea(contour)
        if area > tooSmall:
            significant.append([contour, area])

    significant.sort(key=lambda x: x[1])
    return [x[0] for x in significant]


blurred = cv2.GaussianBlur(image, (5, 5), 0)
edgeImg = np.max(
    np.array(
        [
            detect_edges(blurred[:,:, 0]),
            detect_edges(blurred[:,:, 1]),
            detect_edges(blurred[:,:, 2]),
        ],
    ),
    axis=0,
)
mean = np.mean(edgeImg)
edgeImg[edgeImg <= mean] = 0

edgeImg_8u = np.asarray(edgeImg, np.uint8)
significant_contour = findSignificantContours(image, edgeImg_8u)

# epsilon = 0.10 * cv2.arcLength(significant_contour, True)
# # or epsilon = 3, so slighter contour corrections
# approx = cv2.approxPolyDP(significant_contour, epsilon, True)
# significant_contour = approx
print(significant_contour)

mask = edgeImg.copy()
mask[mask > 0] = 0
cv2.fillPoly(mask, significant_contour, 255)
# Invert mask
mask = np.logical_not(mask)

#Finally remove the background
image[mask] = 0


cv2.imshow("cat", image)
cv2.waitKey(0)
