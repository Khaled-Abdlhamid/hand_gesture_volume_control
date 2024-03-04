import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
pTime = 0
detector = htm.HandDetector(detection_con=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]
vol = 0
bar_vol = 400

while True:
    ret, frame = cap.read()

    frame = detector.find_hands(frame)
    landmarks = detector.find_position(frame, draw_landmark=[4, 8])  # getting the position and only drawing 4 and 8

    if len(landmarks) != 0:
        x1, y1 = landmarks[4][1], landmarks[4][2]  # getting the center of landmark 4 (tip of the thump)
        x2, y2 = landmarks[8][1], landmarks[8][2]  # getting the center of landmark 8 (tip of the index finger)
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(frame, (cx, cy), 10, (0,255,0), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        vol = np.interp(length, [30, 300], [min_vol, max_vol])  # converting the hand range to the volume range
        bar_vol = np.interp(length, [30, 300], [400, 150])
        volume.SetMasterVolumeLevel(vol, None)

        print(vol)
        if length < 30:
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

    cv2.rectangle(frame, (50, 150), (85, 400), (0,255,0), 3)
    cv2.rectangle(frame, (50, int(bar_vol)), (85, 400), (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f'FPS: {str(int(fps))}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
