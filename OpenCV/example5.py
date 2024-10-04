import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2



def contours_detection(path) -> None:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.blur(img, (1, 1))
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    # ret, g_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, s_otsu = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret, v_otsu = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(s_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    plt.imshow(img)
    plt.show()

    # plt.imshow(gray, cmap='gray')
    # plt.show()

    # plt.imshow(hsv[:, :, 0], cmap='gray')
    # plt.show()
    #
    plt.imshow(hsv[:, :, 1], cmap='gray')
    plt.show()
    #
    # plt.imshow(hsv[:, :, 2], cmap='gray')
    # plt.show()

    # plt.imshow(g_otsu)
    # plt.show()

    plt.imshow(s_otsu)
    plt.show()

    # plt.imshow(v_otsu)
    # plt.show()

    filtered = []
    for c in contours:
        if (cv2.arcLength(c, True) > 2000):
            filtered.append(c)


    # show contours with image
    cv2.drawContours(img, filtered, -1, (0, 255, 0), 3)
    plt.imshow(img)
    plt.show()

    # # show len of contours
    # for cnt in contours:
    #     print(cv2.arcLength(cnt, True))




if __name__ == '__main__':
    contours_detection('images/example5.png')