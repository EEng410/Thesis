import os
import cv2
import mediapipe as mp
import csv
import numpy as np
# from Analyze import process_coord
import time
# from datapreprocessing import joint_angles_fast
from itertools import combinations
import pickle 

def joint_angles_fast(coords, combos):
    vecA = np.subtract(coords[combos[:, 1], :], coords[combos[:, 0], :])
    vecB = np.subtract(coords[combos[:, 1], :], coords[combos[:, 2], :])
    X = np.einsum('ij, ij -> i', vecA, vecB)/np.linalg.norm(vecA, 2, 1)/np.linalg.norm(vecB, 2, 1)
    Y = np.linalg.norm(np.cross(vecA, vecB), 2, 1)/np.linalg.norm(vecA, 2, 1)/np.linalg.norm(vecB, 2, 1)
    angles = np.arctan2(Y, X)
    return angles

# Prompt user for the file name
projName = input("Enter project name: ")

# Prompt user for input type
input_choice = 0
while input_choice != "1" and input_choice != "2":
    input_choice = input("Enter 1 for live feed and 2 for video: ")

# Establish input type
if input_choice == "1":
    input_type = 1
else: # 2
    input_type = input("Enter video file path: ")

# Create a directory with the project name
os.makedirs(projName, exist_ok=True)
os.makedirs(projName+"/Left", exist_ok=True)
os.makedirs(projName+"/Right", exist_ok=True)

# Initialize MediaPipe Hands
hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.25, static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open video capture
cap = cv2.VideoCapture(0)

# Get the video width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
normal_video = cv2.VideoWriter(os.path.join(projName, projName + 'gestures.mp4'), fourcc, 20.0,
                                        (width, height))

# Record start time
startTime = time.time()

# Initialize previous landmark positions
previous_landmarks = {}  # For left and right hands

# Set up combination vector so all the angles can be computed
landmarks = np.arange(21)
combos = np.array(list(combinations(landmarks, 3)))

# Get NMF projection matrix
# nmf_proj = np.linalg.pinv(np.loadtxt('nmf_basis.csv', delimiter=',', dtype=float))[:, 1::]
nmf_proj = np.loadtxt('nmf_proj.csv', delimiter=',', dtype=float).T
# Get classifier 
file = open("svm_nmf.pkl",'rb')
model = pickle.load(file)

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    if input_type != 0: # Video input -> flip again
        frame = cv2.flip(frame, 1)
    normal_frame = frame.copy()

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Generate tuples for joint locations on fingers
    # joint_tuples = [(2, 3, 4), (5, 6, 7), (6, 7, 8), (9, 10 , 12), (10, 11, 12), (13, 14, 15), (14, 15, 16), (17, 18, 19), (18, 19, 20)]

    # Generate tuples for joint locations of knuckles
    # knuckle_tuples = [(1, 2, 3), (0, 5, 6), (0, 9, 10), (0, 13, 14), (0, 17, 18)]

    prev_theta = 0 
    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        # Iterate over each detected hand
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Get landmarks for the hand
            hand_x = [landmark.x * frame.shape[1] for landmark in hand_landmarks.landmark]
            hand_y = [landmark.y * frame.shape[0] for landmark in hand_landmarks.landmark]

            # Calculate the bounding box coordinates
            min_x = min(hand_x)
            max_x = max(hand_x)
            min_y = min(hand_y)
            max_y = max(hand_y)

            # Draw the bounding box with the correct color argument placement
            cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)

            # Determine if it's the left or right hand
            handedness = results.multi_handedness[hand_idx].classification[0].label
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Array for storing each position instance (row)
            landmark_row = []

            # Display position of each landmark
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Get landmark position in 2D
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                # Get landmark position in 3D
                z = landmark.z
                # Save position to landmark_row
                landmark_row.append((x, y, z))
            
                # Display position as text
                # cv2.putText(frame, f"Landmark {idx+1}: ({x}, {y}, {z})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            coords = np.array(landmark_row)
            angles = joint_angles_fast(coords, combos)
            proj_angles = nmf_proj @ angles
            pred = model.predict(proj_angles.reshape(1, -1))
            cv2.putText(frame, str(pred), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Write frame to output videos based on handedness
    if results.multi_handedness:
        for handedness in results.multi_handedness:
            hand_label = handedness.classification[0].label
    
    # Original frame without landmarks
    normal_video.write(frame)

    # Show the frame
    cv2.imshow('MediaPipe Hands', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
normal_video.release()
cv2.destroyAllWindows()
hands.close()
