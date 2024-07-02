import numpy as np
import mediapipe as mp
import cv2

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    landmarks = []
    features = {
        "positions": [],
        "velocities": [],
        "accelerations": [],
        "rhythm_consistency": [],
        "timing": []
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            frame_landmarks = []
            for landmark in results.pose_landmarks.landmark:
                frame_landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks.append(frame_landmarks)

    cap.release()

    landmarks = np.array(landmarks)
    num_frames = landmarks.shape[0]
    num_landmarks = landmarks.shape[1]

    for i in range(1, num_frames):
        frame_velocities = []
        frame_accelerations = []
        for j in range(num_landmarks):
            prev_pos = landmarks[i-1][j]
            curr_pos = landmarks[i][j]
            velocity = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
            frame_velocities.append(velocity)

            if i > 1:
                prev_velocity = features["velocities"][-1][j]
                acceleration = velocity - prev_velocity
                frame_accelerations.append(acceleration)
        
        features["positions"].append(landmarks[i])
        features["velocities"].append(frame_velocities)
        if i > 1:
            features["accelerations"].append(frame_accelerations)

    # Calculate rhythm consistency and timing
    for i in range(1, len(features["velocities"])):
        rhythm_score = np.mean(features["velocities"][i]) - np.mean(features["velocities"][i-1])
        features["rhythm_consistency"].append(rhythm_score)

        timing_score = np.mean(features["positions"][i]) - np.mean(features["positions"][i-1])
        features["timing"].append(timing_score)

    return features
