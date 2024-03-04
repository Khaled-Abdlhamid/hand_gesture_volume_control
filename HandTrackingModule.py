import cv2
import mediapipe as mp


class HandDetector():
    def __init__(self, mode=False, max_hands=2, modelC=1, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.modelC = modelC
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.modelC, self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(img_rgb)

        if self.result.multi_hand_landmarks:
            for hand_lm in self.result.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

        return frame

    def find_position(self, frame, hand_number=0, draw_landmark=[]):
        """gets the position for one hand"""

        landmarks = []
        if self.result.multi_hand_landmarks:
            target_hand = self.result.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(target_hand.landmark):  # getting id of the landmark
                height, width, channel = frame.shape
                cx, cy = int(lm.x * width), int(lm.y * height)  # getting the position of each landmark
                landmarks.append([id, cx, cy])
                if id in draw_landmark:  #Drawing only the specified landmarks
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        return landmarks


