import cv2
import numpy as np


def open_image(text='Result') -> None:
    img = cv2.imread('Img/abibaboba.jpg')
    img = cv2.Canny(img, 10, 200)
    help(cv2.Canny)

    cv2.imshow(text, img)
    cv2.waitKey(0)

def open_video() -> None:
    cap = cv2.VideoCapture('video/happybirthday.mp4')

    while True:
        success, img = cap.read()
        cv2.imshow('Result', img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

def open_camera() -> None:
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            img,  # Наше изображение
            (0, 0, 150),  # Начало диапазона
            (255, 255, 255)  # Конец
        )

        cv2.imshow('Result', mask)
        print(img.shape)


        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

def tool_camera_detector_setting_hsv() -> None:
    cv2.namedWindow('mask')  # Создаем окно, тк мы работаем в лупе, чтобы не пересоздавать его постоянно

    def nothing(x):
        pass

    cv2.createTrackbar('lh','mask',0,255, nothing)
    cv2.createTrackbar('ls', 'mask', 0, 255, nothing)
    cv2.createTrackbar('lv', 'mask', 0, 255, nothing)
    cv2.createTrackbar('hh', 'mask', 255, 255, nothing)
    cv2.createTrackbar('hs', 'mask', 255, 255, nothing)
    cv2.createTrackbar('hv', 'mask', 255, 255, nothing)


    cam = cv2.VideoCapture(0)

    while (True):
        success, frame = cam.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lh = cv2.getTrackbarPos('lh', 'mask')
        ls = cv2.getTrackbarPos('ls', 'mask')
        lv = cv2.getTrackbarPos('lv', 'mask')
        hh = cv2.getTrackbarPos('hh', 'mask')
        hs = cv2.getTrackbarPos('hs', 'mask')
        hv = cv2.getTrackbarPos('hv', 'mask')

        mask = cv2.inRange(hsv, (lh, ls, lv), (hh, hs, hv))

        cv2.imshow('mask', mask)
        cv2.waitKey(80)

def camera_detector() -> None:
    cv2.namedWindow('mask')  # Создаем окно, тк мы работаем в лупе, чтобы не пересоздавать его постоянно

    def nothing(x):
        pass
    cv2.createTrackbar('lh','mask',0,255, nothing)
    cv2.createTrackbar('ls', 'mask', 0, 255, nothing)
    cv2.createTrackbar('lv', 'mask', 0, 255, nothing)
    cv2.createTrackbar('hh', 'mask', 255, 255, nothing)
    cv2.createTrackbar('hs', 'mask', 255, 255, nothing)
    cv2.createTrackbar('hv', 'mask', 255, 255, nothing)
    cv2.createTrackbar('l', 'mask', 0, 20000, nothing)
    cv2.createTrackbar('h', 'mask', 0, 20000, nothing)
    cam = cv2.VideoCapture(0)

    while (True):
        success, frame = cam.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lh = cv2.getTrackbarPos('lh', 'mask')
        ls = cv2.getTrackbarPos('ls', 'mask')
        lv = cv2.getTrackbarPos('lv', 'mask')
        hh = cv2.getTrackbarPos('hh', 'mask')
        hs = cv2.getTrackbarPos('hs', 'mask')
        hv = cv2.getTrackbarPos('hv', 'mask')
        mask = cv2.inRange(hsv, (lh, ls, lv), (hh, hs, hv))

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

        filtred = np.zeros_like(mask)

        l = cv2.getTrackbarPos('l', 'mask')
        h = cv2.getTrackbarPos('h', 'mask')
        for i in range(1, num_labels):
            a = stata[i, cv2.CC_STAT_AREA]
            al = stata[i, cv2.CC_STAT_LEFT]
            at = stata[i, cv2.CC_STAT_TOP]
            if (12000 >= l and a < 21000):
                print(a)
                filtred[np.where(labels == i)] = 255
                # cv2.putText(frame, str(a), (al, at), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        print('=================')
        cv2.imshow('filtred', filtred)
        cv2.imshow('frame', frame)
        cv2.waitKey(80)


if __name__ == '__main__':
    camera_detector()
