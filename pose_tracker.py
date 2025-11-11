import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import torch # Import PyTorch for structuring Phase 2 models

# --- PHASE 2: STRUCTURING FUTURE DEEP LEARNING MODELS ---
# The models are defined here conceptually to show the pipeline structure
# NOTE: These classes are placeholders and require training/implementation
# using a dataset like Human3.6M.

class ThreeDLiftingModel(torch.nn.Module):
    """
    Placeholder for the 2D-to-3D Lifting Model (Phase 2).
    This would typically be a Graph Convolutional Network (GCN) or an MLP.
    Input: Normalized 2D Keypoints (17 joints * 2 dimensions)
    Output: Estimated 3D Keypoints (17 joints * 3 dimensions)
    """
    def __init__(self, input_dim=34, output_dim=51):
        super(ThreeDLiftingModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, output_dim)
        print("Model 1: 3D Lifting Model initialized (Placeholder)")

    def forward(self, x):
        # In a real scenario, this would load trained weights and predict 3D
        return self.fc2(self.relu(self.fc1(x)))
    
class ActionRecognizer(torch.nn.Module):
    """
    Placeholder for the Action Recognition Model (Phase 3).
    This would typically be an LSTM/RNN for sequence modeling.
    Input: Sequence of 3D Keypoints over N frames.
    Output: Action Classification (e.g., 'Walking', 'Falling').
    """
    def __init__(self, input_size=51, hidden_size=64, num_layers=2, num_classes=5):
        super(ActionRecognizer, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        print("Model 2: Action Recognition Model initialized (Placeholder)")

    def forward(self, x):
        # In a real scenario, this would process the sequence for action prediction
        return self.fc(self.lstm(x)[0][:, -1, :])

# --- PHASE 1: REAL-TIME 2D DETECTION & DATA LOGGING ---

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Setup video capture (0 for default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam. Please check if it's connected or try a video file.")
    exit()

# Setup logging file
csv_file_path = '2d_keypoints_log.csv'
header = ['frame_id']
# MediaPipe detects 33 landmarks, we only log the essential 17 (Nose, Shoulders, Elbows, etc.)
for i in range(33):
    header.extend([f'x_{i}', f'y_{i}', f'z_{i}', f'v_{i}'])

try:
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        frame_counter = 0
        prev_time = 0

        # Initialize the Pose model (using the full version for accuracy)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            
            # Instantiate placeholders for Phase 2 models to show structure
            lifting_model = ThreeDLiftingModel()
            action_model = ActionRecognizer()

            print("\n--- Starting Real-Time Pose Tracker (Press 'q' to quit) ---")
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'
                    continue

                # 1. Pre-process Image (BGR to RGB for MediaPipe)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # 2. Make Detection
                results = pose.process(image)
                
                # 3. Post-process Image (RGB back to BGR for OpenCV display)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                keypoint_row = [frame_counter]
                
                # 4. Extract and Log Keypoints
                if results.pose_landmarks:
                    # Draw skeleton visualization
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
                    
                    # Extract normalized coordinates for logging
                    for landmark in results.pose_landmarks.landmark:
                        # Normalized coordinates (0.0 to 1.0) are ideal for ML input
                        keypoint_row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                else:
                    # Log zeros if no pose is detected to maintain CSV structure
                    keypoint_row.extend([0.0] * (33 * 4)) 

                # Write the keypoint data to CSV
                writer.writerow(keypoint_row)
                frame_counter += 1

                # 5. Display FPS and Instructions
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
                prev_time = curr_time
                
                cv2.putText(image, f'FPS: {int(fps)}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, 'Keypoints Logged. Press Q to Quit.', (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imshow('Real-Time 2D Pose Detector (P7 POC)', image)

                # Check for 'q' key to quit
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n--- Application Closed. Data saved to {csv_file_path} ---")