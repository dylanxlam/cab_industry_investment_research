import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Function to plot time series data
def plot_time_series(data, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(f'Profit Margin Over Time ({title})')
    plt.xlabel('Date')
    plt.ylabel('Profit Margin')
    plt.show()

# Function to fit an ARIMA model
def fit_arima_model(data, order):
    model = sm.tsa.ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

# Function to generate a forecast
def generate_forecast(model_fit, steps, last_date):
    forecast = model_fit.forecast(steps=steps)
    forecast_index = pd.date_range(start=last_date, periods=steps)
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)
    return forecast_df

# Convert 'Date of Travel' to a datetime object
master_data['Date of Travel'] = pd.to_datetime(master_data['Date of Travel'], format='%Y-%m-%d')
pink_data = master_data[master_data['Company'] == 'Pink Cab']
yellow_data = master_data[master_data['Company'] == 'Yellow Cab']

# Set 'Date of Travel' as the index
pink_data.set_index('Date of Travel', inplace=True)
yellow_data.set_index('Date of Travel', inplace=True)

# Select only the numeric columns for resampling
numeric_cols = ['KM Travelled', 'Price Charged', 'Cost of Trip', 'Price per KM', 'Profit', 'Income (USD/Month)']
pink_data_numeric = pink_data[numeric_cols].resample('D').mean()
yellow_data_numeric = yellow_data[numeric_cols].resample('D').mean()

# Extract the 'Profit_Margin' column as the target variable
profit_margin_pink = pink_data_numeric['Profit']
profit_margin_yellow = yellow_data_numeric['Profit']

# Plot time series data
plot_time_series(profit_margin_pink, 'Pink')
plot_time_series(profit_margin_yellow, 'Yellow')

# Fit ARIMA models
p, d, q = 1, 1, 1
pink_model_fit = fit_arima_model(profit_margin_pink, (p, d, q))
yellow_model_fit = fit_arima_model(profit_margin_yellow, (p, d, q))

# Generate forecasts
forecast_steps = 30  # Adjust the number of forecasted days as needed
last_date_pink = profit_margin_pink.index[-1]
last_date_yellow = profit_margin_yellow.index[-1]

pink_forecast_df = generate_forecast(pink_model_fit, forecast_steps, last_date_pink)
yellow_forecast_df = generate_forecast(yellow_model_fit, forecast_steps, last_date_yellow)

# Plot the original data and the forecast for Pink Cab
plt.figure(figsize=(12, 6))
plt.plot(profit_margin_pink, label='Original Data')
plt.plot(pink_forecast_df, label='Forecast', linestyle='--', color='orange')
plt.title('Profit Margin Forecast (Pink Cab)')
plt.xlabel('Date')
plt.ylabel('Profit Margin')
plt.legend()
plt.show()

# Plot the original data and the forecast for Yellow Cab
plt.figure(figsize=(12, 6))
plt.plot(profit_margin_yellow, label='Original Data')
plt.plot(yellow_forecast_df, label='Forecast', linestyle='--', color='orange')
plt.title('Profit Margin Forecast (Yellow Cab)')
plt.xlabel('Date')
plt.ylabel('Profit Margin')
plt.legend()
plt.show()

# Display the forecast DataFrames
print("Pink Cab's Forecast:")
print(pink_forecast_df)
print("\nYellow Cab's Forecast:")
print(yellow_forecast_df)
