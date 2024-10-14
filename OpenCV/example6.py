import glob
import math

import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2



def make_photo():
    cam = cv2.VideoCapture(0)

    i = 0

    while(True):
        success, frame = cam.read()

        if (success == False):
            print('Cannot read camera frame')

            break

        key = cv2.waitKey(20) & 0xFF

        if (key == ord('q')):
            break

        elif (key == 32): # space
            cv2.imwrite('calib/' + str(i) + '.jpg', frame)
            i += 1

        cv2.imshow('frame', frame)

def find_cornel():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

    # prepare objects points, like (0, 0, 0), (1, 0, 0), (2, 0, 0), ... (6, 5, 0)
    objp = np.zeros((7*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * 0.02

    # Arrays to stare object point and image points from all images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    images = glob.glob('calib/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,7), corners, ret)

            plt.imshow(img)
            plt.axis('off')
            plt.show()

            cv2.waitKey(50)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("ret", ret)
    print("\nmtx", mtx)
    print("\ndist", dist)
    print("\nvecs")

    for i in range(7):
        print(rvecs[i])
        print("")
    print("\ntvecs")

    for i in range(7):
        print(tvecs[i])
        print("")

    dict_calibrate = {
        "mtx" : mtx,
        "dist" : dist
    }

    return dict_calibrate

def chessboard_distance():
    A = np.array([[552.60918515,   0.,         355.6086743 ],
                  [  0.,         553.47490672, 275.06509388],
                  [  0.,           0.,           1.        ]])
    dist = np.array([-0.14397605,  0.55166717,  0.00519162,  0.0084168,  -1.01070363])

    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * 0.02

    objpoints = []
    imgpoints = []

    filename = 0

    cam = cv2.VideoCapture(filename)

    while (True):
        success, frame = cam.read()

        if (success == False):
            cam.release()
            cam = cv2.VideoCapture(filename)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        if (ret == True):
            s, rvec, tvec = cv2.solvePnP(objp, corners, A, dist, flags=0)

            if (s == False):
                continue

            srvec = "rvec: " + str(rvec[0][0])[:5] + " " + str(rvec[1][0])[:5] + " " + str(rvec[2][0])[:5]
            image = cv2.putText(frame, srvec, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
            stvec = "tvec: " + str(tvec[0][0])[:5] + " " + str(tvec[1][0])[:5] + " " + str(tvec[2][0])[:5]
            image = cv2.putText(frame, stvec, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            Rt = cv2.Rodrigues(rvec)
            Rt = np.transpose(Rt[0])

            sy = math.sqrt(Rt[0, 0] * Rt[0, 0] + Rt[1, 0] * Rt[1, 0])
            singular = sy < 1e-6

            if not singular:
                x = math.atan2(Rt[2, 1], Rt[2, 2]) * (180 / np.pi)
                y = math.atan2(-Rt[2, 0], sy) * (180 / np.pi)
                z = math.atan2(Rt[1, 0], Rt[0, 0]) * (180 / np.pi)
            else:
                x = math.atan2(-Rt[1, 2], Rt[1, 1]) * (180 / np.pi)
                y = math.atan2(-Rt[2, 0], sy) * (180 / np.pi)
                z = 0

            image = cv2.putText(frame, str(x)[:5] + " " +
                                       str(y)[:5] + " " +
                                       str(z)[:5], (50, 150), cv2.FONT_HERSHEY_DUPLEX,
                                .5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("frame", frame)

        key = cv2.waitKey(70) & 0xFF

        if (key == ord('q')):
            break


# A - matrix of intrinsic parameters
#  a - angle of rotation of the camera around vertical axis
# b - inclination angle of the camera
# x, y - coordinates of the object in the picture
# h - height of the camera above the ground

def pic2r(A, alpha, beta, x, y, h):
    fx = A[0, 0]
    fy = A[1, 1]
    cx = A[0, 2]
    cy = A[1, 2]

    y_ = h * math.tan(math.pi / 2 - beta - math.atan((y - cy) / fy))
    x_ = y_ * (x - cx) / fx

    R = np.array([[math.cos(alpha), -math.sin(alpha)],
                  math.sin(alpha), math.cos(alpha)])

    rotated = R @ np.array([x_, y_])

    return rotated[0], rotated[1]

def coordinate_of_object():
    A = np.array([[552.60918515, 0., 355.6086743],
                  [0., 553.47490672, 275.06509388],
                  [0., 0., 1.]])
    dist = np.array([-0.14397605, 0.55166717, 0.00519162, 0.0084168, -1.01070363])

    x, y = 0, 0
    cam = cv2.VideoCapture(0)

    while True:
        success, frame = cam.read()

        if success == False:
            print('Cannot read frame')
            break

        blurred = cv2.blur(frame, (7, 7))
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, (0, 0, 0), (255, 255, 255))

        cv2.imshow("mask", mask)

        connectivity = 4
        output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)

        num_labels = output[0]
        labels = output[1]
        stats = output[2]

        filtered = np.zeros_like(mask)

        obj_h = 0

        print(num_labels)

        for i in range(1, num_labels):
            a = stats[i, cv2.CC_STAT_AREA]
            t = stats[i, cv2.CC_STAT_TOP]
            l = stats[i, cv2.CC_STAT_LEFT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            if a >= 1500:
                cv2.rectangle(frame, (l, t), (l+w, t+h), (112, 123, 234), 4)

                x = l + w // 2
                y = t + h

                break

            xr, yr = pic2r(A, 0, (90 - 57) / 180 * math.pi, x, y, 30)

            image = cv2.putText(frame, str(xr)[:5] + " " + str(yr)[:5], (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("frame", frame)

            key = cv2.waitKey(90) & 0xFF
            if (key == ord('q')):
                break

if __name__ == '__main__':
    coordinate_of_object()
