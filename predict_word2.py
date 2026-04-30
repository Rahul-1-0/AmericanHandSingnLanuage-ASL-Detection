import cv2
import mediapipe as mp
import joblib

model = joblib.load("sign_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

current_letter = ""
word = ""

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame,
                                   hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            landmarks=[]

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            prediction = model.predict([landmarks])[0]

            current_letter = prediction

            cv2.putText(frame,
                        "Detected: "+current_letter,
                        (50,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0,255,0),
                        3)

    cv2.putText(frame,
                "Word: "+word,
                (50,150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255,0,0),
                3)

    cv2.imshow("Sign Detection", frame)

    key = cv2.waitKey(1)

    # confirm letter
    if key == ord('c'):
        word += current_letter
        print("Added:", current_letter)

    # retake (ignore prediction)
    if key == ord('t'):
        print("Retake letter")

    # show result
    if key == ord('r'):
        print("Final Word:", word)

    # clear word
    if key == ord('x'):
        word = ""
        print("Word cleared")

    # exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()