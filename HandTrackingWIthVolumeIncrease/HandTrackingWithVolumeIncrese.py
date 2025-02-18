import cv2
import mediapipe as mp
import time
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=mode, 
            max_num_hands=maxHands, 
            min_detection_confidence=detectionCon, 
            min_tracking_confidence=trackCon
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None  # Store hand detection results

    def detect_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        
        return img

    def find_position(self, img, handNo=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))  # Store landmarks
        
        return lm_list

class VolumeController:
    def __init__(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = interface.QueryInterface(IAudioEndpointVolume)
        self.vol_range = self.volume.GetVolumeRange()  # Get volume range
        self.min_vol = self.vol_range[0]  # Minimum volume (-65.25)
        self.max_vol = self.vol_range[1]  # Maximum volume (0.0)

    def set_volume(self, distance):
        """Map distance to volume range and set system volume."""
        vol = np.interp(distance, [20, 200], [self.min_vol, self.max_vol])
        self.volume.SetMasterVolumeLevel(vol, None)

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    vol_controller = VolumeController()

    pTime = 0  # Previous time for FPS calculation

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.detect_hands(img)
        lm_list = detector.find_position(img)

        if len(lm_list) >= 9:  # Ensure both thumb tip (4) and index finger tip (8) are detected
            x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb tip
            x2, y2 = lm_list[8][1], lm_list[8][2]  # Index finger tip

            # Draw circles at the detected points
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Draw line between fingers

            # Calculate distance between thumb and index finger
            distance = math.hypot(x2 - x1, y2 - y1)

            # Adjust system volume based on distance
            vol_controller.set_volume(distance)

            # Display Volume Level
            cv2.putText(img, f'Vol: {int(np.interp(distance, [20, 200], [0, 100]))}%', (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # FPS Calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime

        # Display FPS on frame
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)

        cv2.imshow("Hand Volume Control", img)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
