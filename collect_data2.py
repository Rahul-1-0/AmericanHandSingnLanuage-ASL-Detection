import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Enter sign label (A-Z): ")

sample_count = 0
max_samples = 300   # change if you want more samples

with open("dataset.csv","a",newline="") as f:
    writer = csv.writer(f)

    while True:

        ret,frame = cap.read()
        frame = cv2.flip(frame,1)

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

                landmarks=[]

                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                landmarks.append(label)

                writer.writerow(landmarks)

                sample_count += 1

        cv2.putText(frame,f"Samples: {sample_count}",(10,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("Collecting Data",frame)

        if sample_count >= max_samples:
            break

        if cv2.waitKey(1)==27:
            break

cap.release()
cv2.destroyAllWindows()

print("Data collection completed!")