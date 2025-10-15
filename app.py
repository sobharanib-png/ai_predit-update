# AI-Based Weather Prediction using LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# Step 1: Load and prepare data
# -----------------------------
data = pd.read_csv("weather.csv")

# Ensure date is in datetime format and sorted
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Select feature (Temperature for prediction)
values = data['Temperature'].values.reshape(-1, 1)

# Scale data for neural network
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# -----------------------------
# Step 2: Prepare time-series data
# -----------------------------
def create_dataset(dataset, time_step=10):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, Y = create_dataset(scaled, time_step)

# Reshape for LSTM [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# -----------------------------
# Step 3: Split into train & test
# -----------------------------
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# -----------------------------
# Step 4: Build LSTM model
# -----------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# -----------------------------
# Step 5: Train model
# -----------------------------
model.fit(X_train, Y_train, batch_size=16, epochs=50, verbose=1)

# -----------------------------
# Step 6: Predictions
# -----------------------------
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Reverse scaling
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train_inv = scaler.inverse_transform([Y_train])
Y_test_inv = scaler.inverse_transform([Y_test])

# -----------------------------
# Step 7: Visualization
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(data['Date'][-len(test_predict):], Y_test_inv[0], label='Actual Temperature', color='blue')
plt.plot(data['Date'][-len(test_predict):], test_predict[:,0], label='Predicted Temperature', color='red')
plt.title("AI-Based Weather Prediction (LSTM)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()