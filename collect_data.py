import cv2
import mediapipe as mp
import csv
import os

# Ask for the sign label
label = input("Enter sign label (A, B, C, etc.): ")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# CSV file location
csv_file = "dataset/landmarks.csv"

# Create header if file doesn't exist
if not os.path.exists(csv_file):
    header = []

    for i in range(21):
        header.append(f"x{i}")
        header.append(f"y{i}")

    header.append("label")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

count = 0
max_samples = 100

print(f"Collecting {max_samples} samples for '{label}'")

while count < max_samples:
    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            row = []

            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            for lm in hand_landmarks.landmark:
                 row.append(lm.x - base_x)
                 row.append(lm.y - base_y)

            row.append(label)

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            count += 1

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            cv2.putText(
                frame,
                f"Samples: {count}/{max_samples}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Collect Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Finished collecting data for '{label}'")
