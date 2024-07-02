# # main.py
# from src.feature_extraction import extract_features_from_video
# from src.preprocessing import preprocess_features
# from src.model import build_model, train_model
# from src.evaluate import evaluate_dance
# import numpy as np

# def main():
#     video_path = 'data/robot_dance.mov'

#     # データの読み込みと特徴量抽出
#     features = extract_features_from_video(video_path)
#     preprocessed_features = preprocess_features(features)

#     # データの分割（例として訓練データとラベルを用意）
#     X_train = preprocessed_features[:-10]  # 訓練データ
#     y_train = np.random.rand(X_train.shape[0], 1)  # ダミーのラベル

#     # モデルの構築とトレーニング
#     input_shape = (preprocessed_features.shape[1], preprocessed_features.shape[2])
#     model = build_model(input_shape)
#     model = train_model(model, X_train, y_train)

#     # 評価の実行
#     predictions = evaluate_dance(video_path, model)
#     print("評価結果:", predictions)dfe

# if __name__ == "__main__":
#     main()

# from src.data_loader import load_video
# from src.feature_extraction import extract_features_from_video
# from src.evaluate import evaluate_dance

# video_path = 'data/robot_dance.mov'
# features = extract_features_from_video(video_path)
# evaluation_result = evaluate_dance(features)
# print("評価結果:", evaluation_result)

from src.data_loader import load_video
from src.feature_extraction import extract_features_from_video
from src.evaluate import evaluate_dance

video_path = 'data/robot_dance.mov'  # 動画ファイルのパスを指定してください
frames = load_video(video_path)
features = extract_features_from_video(video_path)
evaluate_dance(features)
