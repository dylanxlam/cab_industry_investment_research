import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Duplicate your master_data DataFrame to avoid tampering with the original
duplicated_master_data = master_data_no_duplicates.copy()

# Extract the relevant column for time series forecasting (i.e., 'Profit')
data = duplicated_master_data[['Date of Travel', 'Profit']].copy()

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

# Create time series samples
X, y = create_time_series_data(monthly_data['Profit'].values.reshape(-1, 1), look_back)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the TensorFlow RNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(look_back, 1)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])


# Make predictions on the test data
y_test_pred = model.predict(X_test)

# Inverse transform the scaled predictions to the original scale
y_test_pred = y_test_pred  # No scaling was applied

# Prepare the date index for plotting
date_index_test = monthly_data.index[-len(y_test):]

# Plot the actual and predicted profit margins
plt.figure(figsize=(12, 6))
plt.plot(date_index_test, y_test, label='Actual Profit Margin')
plt.plot(date_index_test, y_test_pred, label='Predicted Profit Margin', linestyle='--')
plt.title('Profit Margin Forecasting')
plt.xlabel('Month')
plt.ylabel('Profit Margin')
plt.legend()
plt.show()
