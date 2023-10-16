

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Assuming you have the 'master_data' DataFrame
# and you've already performed data preprocessing

# Duplicate your master_data DataFrame to avoid tampering with the original
duplicated_master_data = master_data.copy()

# Extract the relevant column for time series forecasting (i.e., 'Profit_Margin')
data = duplicated_master_data[['Date of Travel', 'Profit_Margin']].copy()

# Group data by month and calculate the mean profit for each month
data['Date of Travel'] = pd.to_datetime(data['Date of Travel'])
data.set_index('Date of Travel', inplace=True)
monthly_data = data.resample('M').mean()

# Define a function to create time series samples with a given look-back window
def create_time_series_data(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Set the look-back window for creating time series samples
look_back = 12  # A look-back of 12 months (1 year)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(monthly_data)

# Create time series samples for training
X_train, y_train = create_time_series_data(scaled_data, look_back)

# Define the TensorFlow RNN model with a more complex architecture
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(200, activation='relu', return_sequences=True, input_shape=(look_back, 1)),
    tf.keras.layers.LSTM(200, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(100, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model without early stopping
model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2)

# Predict on the training data
y_train_pred = model.predict(X_train)

# Inverse transform the scaled predictions to the original scale
y_train_pred = scaler.inverse_transform(y_train_pred)

# Extend the date index for the prediction period
date_index_test = monthly_data.index[-len(y_train):]
date_index_test = pd.date_range(start=date_index_test[-1], periods=12, freq='M')  # Predict one year

# Predict on the test data
X_test = scaled_data[-look_back:]  # Use the last 'look_back' data points in the scaled data
y_test_pred = []

for _ in range(12):  # Predict one year
    prediction = model.predict(X_test.reshape(1, look_back, 1))
    y_test_pred.append(scaler.inverse_transform(prediction).flatten())
    X_test = np.append(X_test, prediction).flatten()[1:]  # Update X_test for the next prediction

y_test_pred = np.array(y_test_pred).flatten()

# Debug: Print the first few transformed predicted values
print("First few transformed predicted values:", y_test_pred[:5])

# Plot the actual and predicted profit margins for the forecasted period
plt.figure(figsize=(12, 6))
plt.plot(monthly_data.index, monthly_data.values, label='Actual Profit Margin')
plt.plot(date_index_test, y_test_pred, label='Predicted Profit Margin', linestyle='--')
plt.title('Profit Margin Forecasting')
plt.xlabel('Month')
plt.ylabel('Profit Margin')
plt.legend()
plt.show()
