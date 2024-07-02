# src/evaluate.py
from src.feature_extraction import extract_features_from_video
from src.preprocessing import preprocess_features

def evaluate_dance(video_path, model):
    features = extract_features_from_video(video_path)
    preprocessed_features = preprocess_features(features)
    predictions = model.predict(preprocessed_features)
    return predictions
