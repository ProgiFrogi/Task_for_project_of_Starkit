import numpy as np
import cv2
import copy

# now we try detect example2.png
def camera_detector(img) -> None:
    w_img, h_img, _ = img.shape

    cv2.namedWindow('mask')  # Создаем окно, тк мы работаем в лупе, чтобы не пересоздавать его постоянно

    def nothing(x):
        pass
    cv2.createTrackbar('lh','mask',8,255, nothing)
    cv2.createTrackbar('ls', 'mask', 20, 255, nothing)
    cv2.createTrackbar('lv', 'mask', 63, 255, nothing)
    cv2.createTrackbar('hh', 'mask', 27, 255, nothing)
    cv2.createTrackbar('hs', 'mask', 98, 255, nothing)
    cv2.createTrackbar('hv', 'mask', 196, 255, nothing)
    cv2.createTrackbar('l', 'mask', 590, 20000, nothing)
    cv2.createTrackbar('h', 'mask', 20000, 20000, nothing)

    while (True):
        frame = cv2.resize(copy.deepcopy(img), (h_img//2, w_img//2))


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lh = cv2.getTrackbarPos('lh', 'mask')
        ls = cv2.getTrackbarPos('ls', 'mask')
        lv = cv2.getTrackbarPos('lv', 'mask')
        hh = cv2.getTrackbarPos('hh', 'mask')
        hs = cv2.getTrackbarPos('hs', 'mask')
        hv = cv2.getTrackbarPos('hv', 'mask')
        mask = cv2.inRange(hsv, (lh, ls, lv), (hh, hs, hv))
        mask = 255 - mask

        kernel = np.ones((4, 4), np.uint8)
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('morph', morph)

        cv2.imshow('mask', mask)

        connectivity = 1
        # Perform the operation
        output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)

        # Get results
        # The nuber of labels
        num_labels = output[0]
        # label matrix
        labels = output[1]
        # stat matrix
        stata = output[2]

        filtred = np.zeros_like(morph)

        l = cv2.getTrackbarPos('l', 'mask')
        h = cv2.getTrackbarPos('h', 'mask')
        for i in range(1, num_labels):
            a = stata[i, cv2.CC_STAT_AREA]
            al = stata[i, cv2.CC_STAT_LEFT]
            at = stata[i, cv2.CC_STAT_TOP]
            aw = stata[i, cv2.CC_STAT_WIDTH]
            ah = stata[i, cv2.CC_STAT_HEIGHT]
            if (a >= l and a < h):
                print(a)
                filtred[np.where(labels == i)] = 255
                cv2.putText(frame, str(a / (aw*ah))[:4], (al, at), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (al, at), (al + aw, at + ah), (0, 255, 0), 2)
        # print('=================')
        cv2.imshow('filtred', filtred)
        cv2.imshow('frame', frame)
        cv2.waitKey(80)


if __name__ == '__main__':
    img = cv2.imread('images/example2.png')
    camera_detector(img)