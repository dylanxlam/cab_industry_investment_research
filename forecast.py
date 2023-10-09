import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Convert 'Date of Travel' to a datetime object
master_data['Date of Travel'] = pd.to_datetime(master_data['Date of Travel'], format='%Y-%m-%d')

# Set 'Date of Travel' as the index
master_data.set_index('Date of Travel', inplace=True)

# Select only the numeric columns for resampling
numeric_cols = ['KM Travelled', 'Price Charged', 'Cost of Trip', 'Price per KM', 'Profit', 'Income (USD/Month)']
master_data_numeric = master_data[numeric_cols]

# Resample data to a daily frequency (assuming your data is not already daily)
master_data_numeric = master_data_numeric.resample('D').mean()

# Extract the 'Profit_Margin' column as the target variable
profit_margin = master_data_numeric['Profit']

# Perform data exploration and visualization (optional)
# For example, you can plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(profit_margin)
plt.title('Profit Margin Over Time')
plt.xlabel('Date')
plt.ylabel('Profit Margin')
plt.show()

# Check for autocorrelation and partial autocorrelation
plot_acf(profit_margin, lags=30)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(profit_margin, lags=30)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Fit an ARIMA model
# You may need to adjust the order (p, d, q) based on ACF and PACF plots
p, d, q = 1, 1, 1
model = sm.tsa.ARIMA(profit_margin, order=(p, d, q))
model_fit = model.fit()

# Generate forecasts
forecast_steps = 30  # Adjust the number of forecasted days as needed
forecast = model_fit.forecast(steps=forecast_steps)

# Create a date range for the forecasted period
forecast_index = pd.date_range(start=profit_margin.index[-1], periods=forecast_steps)

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)

# Plot the original data and the forecast
plt.figure(figsize=(12, 6))
plt.plot(profit_margin, label='Original Data')
plt.plot(forecast_df, label='Forecast', linestyle='--', color='orange')
plt.title('Profit Margin Forecast')
plt.xlabel('Date')
plt.ylabel('Profit Margin')
plt.legend()
plt.show()

# Display the forecast DataFrame
print(forecast_df)

