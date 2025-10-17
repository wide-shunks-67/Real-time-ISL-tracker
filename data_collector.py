import cv2
import mediapipe as mp
import numpy as np
import csv
import os

data_directory= 'ISL_DATA'
os.makedirs(data_directory, exist_ok=True)

signs=["hello", "thank you", "A", "B", "Golu"]
for sign in signs:
    os.makedirs(os.path.join(data_directory, sign), exist_ok=True)

mp_hand= mp.solutions.hands
mp_drawing= mp.solutions.drawing_utils
hand= mp_hand.Hands(min_detection_confidence=0.7 , min_tracking_confidence=0.5)

cap= cv2.VideoCapture(0)

while cap.isOpened():
    success,frame =cap.read()
    if not success:
        continue
    
    frame=cv2.flip(frame,1)
    rgb_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results= hand.process(rgb_frame)


    # Display instructions on the frame
    cv2.putText(frame, "Press key to collect data for a sign...", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "h: hello | t: thankyou | a: A | b: B |g :GOLU| q: quit", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ... (MediaPipe processing and drawing landmarks is the same) ...
    if results.multi_hand_landmarks:
       for hand_landmarks in results.multi_hand_landmarks:
           mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hand.HAND_CONNECTIONS )
    
           key = cv2.waitKey(1) & 0xFF
            
            # Check which key was pressed and save data accordingly
           sign_to_save = None
           if key == ord('h'):
            sign_to_save = "hello"
           elif key == ord('t'):
            sign_to_save = "thank you"
           elif key == ord('a'):
            sign_to_save = "A"
           elif key == ord('b'):
            sign_to_save = "B"
           elif key == ord('g'):
            sign_to_save = "GOLU"

           if sign_to_save:
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)
                
                num_existing_files = len(os.listdir(os.path.join(data_directory, sign_to_save)))
                file_path = os.path.join(data_directory, sign_to_save, f'{sign_to_save}_{num_existing_files}.npy')
                np.save(file_path, landmarks)
                print(f"Saved data for '{sign_to_save}' to {file_path}")

    cv2.imshow('ISL_Data_Collector',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()