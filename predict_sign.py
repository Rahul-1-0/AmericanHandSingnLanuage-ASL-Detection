import cv2
import mediapipe as mp
import joblib

model = joblib.load("sign_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            prediction = model.predict([landmarks])

            cv2.putText(
                frame,
                prediction[0],
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3,
            )

    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()