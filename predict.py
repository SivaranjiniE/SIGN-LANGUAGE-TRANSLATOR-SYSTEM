import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            row = []

            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            for lm in hand_landmarks.landmark:
                row.append(lm.x - base_x)
                row.append(lm.y - base_y)

            # Predict sign
            prediction = model.predict([row])[0]

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display prediction
    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()