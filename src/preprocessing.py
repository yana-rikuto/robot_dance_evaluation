# src/preprocessing.py
import numpy as np

def preprocess_features(features):
    features_array = np.array(features)
    normalized_features = features_array / np.max(features_array)
    return normalized_features
