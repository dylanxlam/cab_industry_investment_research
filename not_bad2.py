import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
X, y = create_time_series_data(scaled_data, look_back)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the TensorFlow RNN model
model = Sequential()
model.add(LSTM(100, input_shape=(look_back, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Predict on the test data
y_test_pred = model.predict(X_test)

# Inverse transform the scaled predictions to the original scale
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Calculate and print the Mean Squared Error
mse = mean_squared_error(y_test_actual, y_test_pred)
print(f"Mean Squared Error: {mse}")

# ... (previous code)

# Extend the date index for the prediction period
date_index_test = pd.date_range(start=monthly_data.index[-1], periods=len(y_test_pred), freq='M')

# Debug: Print the first few transformed predicted values
print("First few transformed predicted values:", y_test_pred[:5])

# Plot the actual and predicted profit margins for the forecasted period
plt.figure(figsize=(12, 6))
plt.plot(monthly_data.index, monthly_data.values, label='Actual Profit Margin')
plt.plot(date_index_test, y_test_pred, label='Predicted Profit Margin', linestyle='--')
plt.title('Profit Margin Forecasting')
plt.xlabel('Month')
plt.ylabel('Profit Margin')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.legend()
plt.show()
