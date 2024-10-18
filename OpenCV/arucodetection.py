import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_marker(marker_id, marker_size, directory, filename):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    cv2.imwrite(f'{directory}/{filename}', marker_image)
    plt.imshow(marker_image, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Hide axes
    plt.title(f'ArUco Marker {marker_id}')
    plt.show()

def detect_marker():
    cap = cv2.VideoCapture(0)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    while True:
        ret, frame = cap.read()

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

        new_frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        cv2.imshow('Found markers2', new_frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    for i in range(30, 40):
	    generate_marker(i, 500, './aruco', f'marker{i}_size500x500.png')