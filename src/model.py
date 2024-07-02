# src/model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=16):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model
