import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2

def plot_dist(channel):
    fig, ax = plt.subplots()
    # plt.figure(figsize=(4, 6))
    ax.hist(channel.ravel(), 25, [0,256])

    fig.canvas.draw()
    dist = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()

    return dist

def nothing(x):
    pass

def camera_detector(path) -> None:
    cv2.namedWindow('frame')

    cam = cv2.VideoCapture(path)

    cv2.createTrackbar('hs', 'frame', 256, 512, nothing)
    cv2.createTrackbar('ss', 'frame', 256, 512, nothing)
    cv2.createTrackbar('vs', 'frame', 256, 512, nothing)


    while (True):
        success, frame = cam.read()
        if (success == False):
            cam.release()
            cam = cv2.VideoCapture(path)

            continue

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        hs = cv2.getTrackbarPos('hs', 'frame')
        ss = cv2.getTrackbarPos('ss', 'frame')
        vs = cv2.getTrackbarPos('vs', 'frame')

        hsv_frame[:, :, 0] = cv2.add(hsv_frame[:, :, 0], hs - 256)
        hsv_frame[:, :, 1] = cv2.add(hsv_frame[:, :, 1], ss - 256)
        hsv_frame[:, :, 2] = cv2.add(hsv_frame[:, :, 2], vs - 256)

        dist_0 = plot_dist(hsv_frame[::30, ::30, 0])
        dist_1 = plot_dist(hsv_frame[::30, ::30, 1])
        dist_2 = plot_dist(hsv_frame[::30, ::30, 2])

        cv2.imshow('frame', cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR))

        cv2.imshow('dist_0', dist_0)
        cv2.imshow('dist_1', dist_1)
        cv2.imshow('dist_2', dist_2)
        cv2.waitKey(5)


if __name__ == '__main__':
    camera_detector('videos/example3.mp4')