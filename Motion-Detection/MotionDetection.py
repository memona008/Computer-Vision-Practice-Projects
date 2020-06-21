from imutils.video import VideoStream
from collections import deque
import numpy as np
import argparse
import cv2
import time
import imutils
pts = deque(maxlen=64)

# opening webcam and getting videostream from there
cap = VideoStream(src=0).start()

time.sleep(2.0)

# get first two frames 
frame1 = cap.read()
frame2 = cap.read()


while True:
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=4)
    contours, hirarchy = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 700:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 255), 2)


    cv2.imshow('Motion Detector', frame1)
    cv2.imshow('Difference Frame', thresh)

    frame1 = frame2

    frame2 = cap.read()

    # Press 'esc' for quit
    if cv2.waitKey(40) == 27:
        break


cap.release()

# Destroy all windows
cv2.destroyAllWindows()
