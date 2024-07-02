import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def evaluate_dance(features):
    # 各特徴量をNumpy配列に変換
    positions = np.array(features["positions"])
    velocities = np.array(features["velocities"])
    accelerations = np.array(features["accelerations"])
    rhythm_consistency = np.array(features["rhythm_consistency"])
    timing = np.array(features["timing"])

    # 最小の長さに合わせて特徴量を切り取る
    min_length = min(len(positions), len(velocities), len(accelerations), len(rhythm_consistency), len(timing))
    positions = positions[:min_length]
    velocities = velocities[:min_length]
    accelerations = accelerations[:min_length]
    rhythm_consistency = rhythm_consistency[:min_length]
    timing = timing[:min_length]

    # 各特徴量を適切な形状に整形
    X_positions = positions.reshape((min_length, -1))
    X_velocities = velocities.reshape((min_length, -1))
    X_accelerations = accelerations.reshape((min_length, -1))
    X_rhythm_consistency = rhythm_consistency.reshape((min_length, 1))
    X_timing = timing.reshape((min_length, 1))

    def build_and_evaluate_model(X, name):
        # 入力を適切な形状に再整形
        X = X.reshape((1, min_length, X.shape[1]))

        # LSTMモデルの構築
        model = Sequential()
        model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(LSTM(32))
        model.add(Dense(1, activation='sigmoid'))

        # モデルのコンパイル
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

        # モデルの学習
        model.fit(X, np.random.rand(1, min_length), epochs=10, verbose=1)

        # モデルの評価と結果の出力
        evaluation_result = model.predict(X)
        print(f"{name}の評価結果:", evaluation_result)

    # 各特徴量ごとにモデルを構築して評価
    build_and_evaluate_model(X_positions, "ポジション")
    build_and_evaluate_model(X_velocities, "速度")
    build_and_evaluate_model(X_accelerations, "加速度")
    build_and_evaluate_model(X_rhythm_consistency, "リズム一貫性")
    build_and_evaluate_model(X_timing, "タイミング")
