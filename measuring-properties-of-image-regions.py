import cv2
import numpy as np
from imutils import contours
import imutils
from matplotlib import pyplot as plt


# type the name of the image here
img = cv2.imread('image.jpg')
# cv2.imshow('original image', img)


img_copy = img.copy()
# img_copy2 = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
edged = cv2.Canny(gray, 50, 200)
# edged = cv2.dilate(edged, None, iterations=1)
# edged = cv2.erode(edged, None, iterations=1)

im = edged
# cv2.imshow('im', im)

# for colored objects close to the background
m1 = img[:, :, 0]
_, m1_thresh = cv2.threshold(m1, 102, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

m2 = img[:, :, 1]
_, m2_thresh = cv2.threshold(m2, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

m3 = img[:, :, 2]
_, m3_thresh = cv2.threshold(m3, 160, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

image_sum = m1_thresh + m2_thresh + m3_thresh + im

cntrs = cv2.findContours(image_sum, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntrs = imutils.grab_contours(cntrs)
(cntrs, _) = contours.sort_contours(cntrs)

draw = cv2.drawContours(image_sum, cntrs, -1, (255, 250, 255), cv2.FILLED)

# plt.subplot(224)
# plt.imshow(I_out, cmap='gray', interpolation='nearest')

# cv2.imshow('I_out', draw)

orig = img.copy()


# midpoint of two coordinates
def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

def area(cnts):
    # for image name
    j = 0
    area_array = []

    for cnt in cnts:
        # Ignore undersize contour
        # if cv2.contourArea(cnt) < 100:
        #     continue

        j += 1

        # Finding the coordinates and properties of a bounding rectangle
        x_coor, y_coor, w, h = cv2.boundingRect(cnt)

        # Finding the area in coordinates
        rgn = image_sum[y_coor:y_coor + h, x_coor:x_coor + w]

        imgs = img[y_coor:y_coor + h, x_coor:x_coor + w]

        # counting nonzero pixels
        n = cv2.countNonZero(rgn)
        # print("area of the border")
        # area of the rectangle
        area_rect = h * w

        # area of the region
        area_reg = n

        # print("area for " + str(j) + ": ", area_reg)
        # print(n)
        area_array.append(n)

        # creating images of contours
        cv2.imwrite(str(j) + '.jpg', imgs)
    return area_array


def boundingBox(cnts):
    pixelsPerMetric = None
    j = 0
    prop_array = []

    for cnt in cnts:
        # Ignore undersize contour
        if cv2.contourArea(cnt) < 100:
            continue
        j += 1
        # Finding the coordinates and properties of a bounding rectangle
        x_coor, y_coor, w, h = cv2.boundingRect(cnt)
        prop = (x_coor, y_coor, w, h)

        prop_array.append(prop)

        cv2.rectangle(orig, (x_coor, y_coor), (x_coor+w, y_coor+h), (0, 0, 255), 1)

    return prop_array, orig


def eccentricity(cnts):
    j = 0
    eccentricity_array = []
    for cnt in cnts:
        a = ()
        # Ignore undersize contour
        if cv2.contourArea(cnt) < 100:
            continue
        j += 1

        # because the ellipse requires 5 points
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, orientation) = ellipse

            min_axis_length = minor_axis_length(axes, j)
            maj_axis_length = major_axis_length(axes, j)

            # calculate eccentricity
            eccentricity = np.sqrt(1 - (min_axis_length / maj_axis_length) ** 2)

            eccentricity_array.append((eccentricity, min_axis_length, maj_axis_length))

            # print("eccentricity for " + str(j) + ": ", eccentricity)
    return eccentricity_array


def major_axis_length(axes, j):
    # major axis length
    major_axis_length = max(axes)
    # print("major axis length for " + str(j) + ": ", major_axis_length)
    return major_axis_length


def minor_axis_length(axes, j):
    # minor axis length
    minor_axis_length = min(axes)
    # print("minor axis length for " + str(j) + ": ", minor_axis_length)
    return minor_axis_length

area_array = area(cntrs)
prop_array, image = boundingBox(cntrs)
eccentricity_array = eccentricity(cntrs)

plt.figure(), plt.title('img'), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.figure(), plt.title('m1_thresh'), plt.imshow(m1_thresh, cmap='gray', interpolation='nearest')
plt.figure(), plt.title('m2_thresh'), plt.imshow(m2_thresh, cmap='gray', interpolation='nearest')
plt.figure(), plt.title('m3_thresh'), plt.imshow(m3_thresh, cmap='gray', interpolation='nearest')
plt.figure(), plt.title('edged'), plt.imshow(im, cmap='gray', interpolation='nearest')
plt.figure(), plt.title('sum'), plt.imshow(image_sum, cmap='gray', interpolation='nearest')
plt.figure(), plt.title('image'), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.show()
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
