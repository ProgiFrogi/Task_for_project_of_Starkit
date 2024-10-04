import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2



def contours_detection(path) -> None:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blurred = cv2.blur(img, (1, 1))
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    ret, s_otsu = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.imshow(img)
    plt.show()

    plt.imshow(hsv[:, :, 1], cmap='gray')
    plt.show()

    plt.imshow(s_otsu)
    plt.show()

    cl_ker = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(s_otsu, cv2.MORPH_CLOSE, cl_ker)

    op_ker = np.ones((29, 29), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, op_ker)

    plt.imshow(closed)
    plt.show()

    plt.imshow(opened)
    plt.show()
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    filtered = []
    for c in contours:
        l = cv2.arcLength(c, True)
        if ( l > 500):
            approx = cv2.approxPolyDP(c, l*0.01, True)

            filtered.append(approx)

    # show contours with image
    cv2.drawContours(img, filtered, -1, (0, 255, 0), 3)
    plt.imshow(img)
    plt.show()



if __name__ == '__main__':
    contours_detection('images/example5.png')