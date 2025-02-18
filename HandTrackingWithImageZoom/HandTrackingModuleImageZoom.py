import cv2
import mediapipe as mp
import numpy as np

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

    def detect_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        
        return img

    def find_position(self, img, hand_no=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
        return lm_list


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    img_to_zoom = cv2.imread("Add your image here")  # Load an image to zoom
    zoom_factor = 1.0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.detect_hands(img)
        lm_list = detector.find_position(img)

        if len(lm_list) >= 8:  # Ensure landmarks are detected
            thumb_x, thumb_y = lm_list[4][1], lm_list[4][2]  # Thumb tip
            index_x, index_y = lm_list[8][1], lm_list[8][2]  # Index tip
            
            # Calculate distance between thumb and index finger
            distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5
            zoom_factor = np.interp(distance, [30, 300], [1.0, 3.0])  # Map distance to zoom level

            # Resize image based on zoom factor
            height, width = img_to_zoom.shape[:2]
            new_size = (int(width * zoom_factor), int(height * zoom_factor))
            zoomed_img = cv2.resize(img_to_zoom, new_size)

            # Display Zoomed Image
            cv2.imshow("Zoomed Image", zoomed_img)

        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
