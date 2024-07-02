# src/feature_extraction.py
import numpy as np
import mediapipe as mp
import cv2

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose #MediaPipeの姿勢推定モジュールをセットアップします。
    pose = mp_pose.Pose() #姿勢推定のインスタンスを作成します
    features = []

    while cap.isOpened():
        ret, frame = cap.read() #フレームを読み込みます。読み込みに成功した場合、retはTrueになり、frameにはフレームの画像データが格納されます。
        if not ret: 
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #OpenCVのBGR形式のフレームをRGB形式に変換します。
        results = pose.process(frame_rgb) #フレームのRGBデータに対して姿勢推定を行います。
        
        if results.pose_landmarks: #推定結果はresults.pose_landmarksに格納されます。
            frame_features = [] #results.pose_landmarksが存在する場合、各ランドマークの座標をリストframe_featuresに格納します。
            for landmark in results.pose_landmarks.landmark: #各ランドマークのx, y, z座標を抽出し、frame_featuresに追加します。
                frame_features.append([landmark.x, landmark.y, landmark.z])
            features.append(frame_features)

    cap.release()
    return np.array(features)
