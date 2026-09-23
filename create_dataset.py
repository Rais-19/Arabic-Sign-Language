import os
import mediapipe as mp
import cv2
import pickle
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = r"C:\Users\LENOVO\Desktop\Computer Vision\Arabic sign language\data"

data = []
labels = []
total = sum([len(os.listdir(os.path.join(DATA_DIR, d))) for d in os.listdir(DATA_DIR)])
processed = 0
successful = 0
failed = 0

print(f"Total images to process: {total}\n")

for dir_ in sorted(os.listdir(DATA_DIR)):
    print(f"Processing {dir_}...")
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        try:
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            
            if img is None:
                failed += 1
                processed += 1
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            data_aux = []
            x_ = []
            y_ = []

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x_.append(hand_landmarks.landmark[i].x)
                        y_.append(hand_landmarks.landmark[i].y)

                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                        data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                data.append(data_aux)
                labels.append(dir_)
                successful += 1

            processed += 1
            if processed % 500 == 0:
                print(f"  {processed}/{total} | Saved: {successful} | Failed: {failed}")
                
        except Exception as e:
            failed += 1
            processed += 1
            continue

pickle.dump({'data': data, 'labels': labels}, open('data.pickle', 'wb'))

print(f"\n✓ Done!")
print(f"  Saved: {successful} | Failed: {failed}")