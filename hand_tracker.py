import cv2
import mediapipe as mp

mp_hands= mp.solutions.hands
mp_drawing= mp.solutions.drawing_utils

hands= mp_hands.Hands(min_detection_confidence=0.7 , min_tracking_confidence=0.5)

cap= cv2.VideoCapture(0)

while cap.isOpened():
    success,frame = cap.read()
    if not success:
        continue

    frame= cv2.flip(frame ,1)
    rgb_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results= hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('ISL TRACKER', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()