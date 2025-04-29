import os
import time
import cv2
import numpy as np
from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Load the model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Define your gesture classes (Match your trained gestures)
actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# Color list for visualization
colors = [(245,117,16)] * len(actions)

# Visualization function for confidence bars
def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# Detection variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8 

cap = cv2.VideoCapture(0)

# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0,40), (300,400), 255, 2)
        image, results = mediapipe_detection(cropframe, hands)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try: 
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action = actions[np.argmax(res)]
                print(predicted_action)
                predictions.append(np.argmax(res))

                # Only trigger action if it's a new gesture (prevent rapid repeats)
                if len(sentence) == 0 or predicted_action != sentence[-1]:
                    # Gesture-to-Action Mapping (with nircmd folder path)
                    if predicted_action == 'A':
                        os.system('notepad.exe')
                    elif predicted_action == 'B':
                        os.system('calc.exe')
                    elif predicted_action == 'C':
                        os.system('start chrome')
                    elif predicted_action == 'D':
                        os.system('snippingtool.exe')
                    elif predicted_action == 'E':
                        os.system('nircmd\\nircmd.exe changesysvolume 2000')
                    elif predicted_action == 'F':
                        os.system('nircmd\\nircmd.exe changesysvolume -2000')
                    elif predicted_action == 'G':
                        os.system('nircmd\\nircmd.exe mutesysvolume 2')
                    elif predicted_action == 'H':
                        os.system('rundll32.exe user32.dll,LockWorkStation')
                    elif predicted_action == 'I':
                        os.system('explorer.exe')
                    elif predicted_action == 'J':
                        os.system('nircmd\\nircmd.exe sendkeypress space')
                    
                    # Update sentence and accuracy
                    sentence.append(predicted_action)
                    accuracy.append(str(res[np.argmax(res)]*100))

                    # Cooldown delay to prevent over-triggering
                    time.sleep(1.5)  # 1.5 seconds pause after each action

                # Keep only the last detected gesture
                if len(sentence) > 1: 
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

                # Visualization
                frame = prob_viz(res, actions, frame, colors, threshold)
        except Exception as e:
            pass

        cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame,"Output: -"+' '.join(sentence)+''.join(accuracy), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
