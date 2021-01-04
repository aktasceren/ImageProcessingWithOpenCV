import math
import time
import numpy as np
import pyautogui
import imutils
import cv2
from matplotlib import pyplot as plt
from imutils import contours
from PIL import Image

time.sleep(3)

image = pyautogui.screenshot()
image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
cv2.imwrite("img.jpg", image)

image = cv2.imread("img.jpg")

output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
(cnts, _) = contours.sort_contours(cnts)


def boundingBox(cnts):
    j = 0
    center = []

    for cnt in cnts:

        if len(cnt) > 20:
            j += 1

            # Finding the coordinates and properties of a bounding rectangle
            x_coor, y_coor, w, h = cv2.boundingRect(cnt)

            # center points and radius of contours
            x = int(x_coor + w / 2)
            y = int(y_coor + h / 2)
            r = w / 2

            # calculating the circularity
            circularity = 4 * math.pi * (cv2.contourArea(cnt)) / (2 * math.pi * r) ** 2
            if 0.9 <= circularity <= 1:
                imgs = image[y_coor:y_coor + h, x_coor:x_coor + w]
                cv2.imwrite(str(j) + '.jpg', imgs)
                r = int(r)
                cv2.circle(output, (x, y), r, (255,255,0), 3)
                cv2.circle(output, (x, y), 3, (255, 255, 255), -1)
                center.append((x, y))
    return center



centers = boundingBox(cnts)

cv2.imwrite("screenshot.jpg", output)

# cv2.namedWindow("window", cv2.WINDOW_FREERATIO)
# cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.imshow("window", output)

im = Image.open("screenshot.jpg")
im.show()

time.sleep(2)

# location of full screen button
pyautogui.moveTo(1873, 985)

pyautogui.click()

time.sleep(2)

for c in centers:
    pyautogui.moveTo(c[0], c[1], 1)

cv2.waitKey(0)
cv2.destroyAllWindows()