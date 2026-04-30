import cv2
import mediapipe as mp
import joblib
import time

model = joblib.load("sign_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

word = ""
last_letter = ""
last_time = time.time()

delay = 1.5   # seconds between letter captures

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            prediction = model.predict([landmarks])[0]

            current_time = time.time()

            # add letter after delay
            if prediction != last_letter or (current_time - last_time) > delay:

                word += prediction
                last_letter = prediction
                last_time = current_time

            cv2.putText(frame, prediction, (50,80),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

    # show built word
    cv2.putText(frame, "Word: "+word, (50,150),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

    cv2.imshow("Sign Detection", frame)

    key = cv2.waitKey(1)

    # ESC to exit
    if key == 27:
        break

    # press C to clear word
    if key == ord('c'):
        word = ""

cap.release()
cv2.destroyAllWindows()