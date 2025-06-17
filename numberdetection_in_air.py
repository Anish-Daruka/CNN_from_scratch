
#Number detection in air
#touch the thumb and middle finger to start writing 
#do the same to stop writing ,once you stop ,CNN will byitself predict the number
#features->cropping the drawn number, padding it,resizing it to 28x28, normalizing it


import cv2
import mediapipe as mp
import numpy as np
import time
from model import predict  # Assumes you have a 'predict' function in model.py

class FingerTracker:
    def __init__(self, max_num_hands=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_num_hands)
        self.mp_draw = mp.solutions.drawing_utils
        self.previous_position = None  
        self.on = False  

    def get_index_tip(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        index_tip = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = img.shape
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                index_tip = (x, y)
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return index_tip

    def draw_on_canvas(self, canvas, tip_position, color=(255, 255, 255), thickness=20):
        if tip_position and self.previous_position:
            cv2.line(canvas, self.previous_position, tip_position, color, thickness)
        self.previous_position = tip_position

    def flip(self, img, scale=0.4):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = img.shape
                # Index tip and thumb tip
                middle_tip = np.array([
                    int(hand_landmarks.landmark[12].x * w),
                    int(hand_landmarks.landmark[12].y * h)
                ])
                index_tip = np.array([
                    int(hand_landmarks.landmark[8].x * w),
                    int(hand_landmarks.landmark[8].y * h)
                ])
                thumb_tip = np.array([
                    int(hand_landmarks.landmark[4].x * w),
                    int(hand_landmarks.landmark[4].y * h)
                ])
                # Index tip and index base (MCP)
                index_base = np.array([
                    int(hand_landmarks.landmark[5].x * w),
                    int(hand_landmarks.landmark[5].y * h)
                ])
                # Use index finger length as reference
                finger_length = np.linalg.norm(index_tip - index_base)
                threshold = finger_length * scale
                distance = np.linalg.norm(middle_tip - thumb_tip)
                if distance < threshold:
                    return True
        return False

    def preprocess_canvas(self, canvas):
        # Convert to grayscale
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Threshold to get binary image
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # Find non-zero (white) pixels
        coords = cv2.findNonZero(thresh)

        if coords is not None:
            # Get bounding box from non-zero pixels
            x, y, w, h = cv2.boundingRect(coords)
            # Crop to bounding box
            cropped = thresh[y:y+h, x:x+w]
        else:
            # If nothing is drawn, fallback to blank
            cropped = np.zeros((28, 28), dtype=np.uint8)

        # Add padding to make the cropped image square
        height, width = cropped.shape

        if height > width:
            e = int(height * 0.2)
            pad = (height - width) // 2
            cropped = cv2.copyMakeBorder(cropped, e, e, e + pad, e + height - width - pad, cv2.BORDER_CONSTANT, value=0)
        elif width > height:
            e = int(width * 0.2)
            pad = (width - height) // 2
            cropped = cv2.copyMakeBorder(cropped, pad + e, width - height - pad + e, e, e, cv2.BORDER_CONSTANT, value=0)

        # Resize to 28x28
        resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize to [0,1]
        normalized = resized.astype(np.float32) / 255.0

        # Enhance contrast if needed (optional tweak)
        normalized = np.clip(normalized * 2.0, 0.0, 1.0)

        # Reshape to (1, 1, 28, 28) for model input
        reshaped = normalized.reshape(1, 1, 28, 28)

        print("Processed image shape:", reshaped.shape)
        return reshaped



tracker = FingerTracker()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
img = cv2.flip(img, 1)
canvas = np.zeros_like(img)

# For displaying prediction text
prediction_text = ""
prediction_time = 0

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    tip = tracker.get_index_tip(img)

    if tip:
        cv2.circle(img, tip, 5, (0, 255, 0), -1)

    if tracker.flip(img):
        if tracker.on:
            prediction_input = tracker.preprocess_canvas(canvas)
            predicted = predict(prediction_input)
            print("Predicted:", predicted[0])
            prediction_text = str(predicted[0])
            prediction_time = time.time()

        canvas = np.zeros_like(img)
        tracker.on = not tracker.on
        tracker.previous_position = None
        cv2.waitKey(600)

    if tracker.on:
        tracker.draw_on_canvas(canvas, tip)

    # Combine original image and canvas
    combined = cv2.addWeighted(img, 1, canvas, 1, 0)

    # Display prediction if recent
    if time.time() - prediction_time < 2.0:
        cv2.putText(
            combined,
            f"Predicted: {prediction_text}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 255),
            4
        )

    cv2.imshow("Image", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()