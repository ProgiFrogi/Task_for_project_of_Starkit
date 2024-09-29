import numpy as np
import cv2

def camera_detector(path) -> None:
    cv2.namedWindow('mask')

    cam = cv2.VideoCapture(path)

    _, background = cam.read()
    background = 0
    def nothing(x):
        pass
    cv2.createTrackbar('l', 'mask', 590, 20000, nothing)
    cv2.createTrackbar('h', 'mask', 20000, 20000, nothing)

    while (True):
        success, frame = cam.read()
        if (success == False):
            cam.release()
            cam = cv2.VideoCapture(path)

            continue

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        background = cv2.addWeighted(background, 0.97, hsv_frame, 0.03, 0)

        diff = cv2.absdiff(background, hsv_frame)
        foreground_mask = cv2.inRange(diff, (0, 20, 20), (255, 255, 255))

        # Different window for exploration
        # cv2.imshow("background", background)
        # cv2.imshow("diff0", diff[:, :, 0])
        # cv2.imshow("diff1", diff[:, :, 1])
        # cv2.imshow("diff2", diff[:, :, 2])
        cv2.imshow("foreground", foreground_mask)

        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('morph', morph)

        connectivity = 10
        # Perform the operation
        output = cv2.connectedComponentsWithStats(foreground_mask, connectivity, cv2.CV_32S)
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
        # # print('=================')
        cv2.imshow('filtred', filtred)
        cv2.imshow('frame', frame)
        cv2.waitKey(10)


if __name__ == '__main__':
    camera_detector('videos/example3.mp4')