import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the Excel file
file_path = '/content/AIProject_final.xlsx'
weather_data = pd.read_excel(file_path, sheet_name='기상')

# Convert the date column to datetime format
weather_data['date_formatting'] = pd.to_datetime(weather_data['date_formatting'])

# Replace 0 with NaN in fuji_apple_price and forward-fill missing values
weather_data['fuji_apple_price'] = pd.to_numeric(weather_data['fuji_apple_price'], errors='coerce')
weather_data['fuji_apple_price'] = weather_data['fuji_apple_price'].replace(0, np.nan)
weather_data['fuji_apple_price'] = weather_data['fuji_apple_price'].fillna(method='ffill')

# Prepare data for Prophet
prophet_data = weather_data[['date_formatting', 'fuji_apple_price']].rename(columns={'date_formatting': 'ds', 'fuji_apple_price': 'y'}).dropna()

# Fit Prophet model
model = Prophet()
model.fit(prophet_data)

# Make future dataframe
future = model.make_future_dataframe(periods=7*12, freq='M')

# Predict the future prices
forecast = model.predict(future)

# Display the forecasted data
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Plot the actual and forecasted prices with a line plot
plt.figure(figsize=(12, 6))
plt.plot(prophet_data['ds'], prophet_data['y'], label='Actual Fuji Apple Prices', color='black')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Fuji Apple Prices', color='blue')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.ylim(10000, 40000)  # Set y-axis limit
plt.xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2025-12-31'))  # Set x-axis limit
plt.title('Actual and Forecasted Fuji Apple Prices (2019-2025)')
plt.legend()
plt.grid(True)

# Set y-axis ticks to display prices in 3000 intervals
y_ticks = np.arange(10000, 40001, 3000)
plt.yticks(y_ticks)

# Set x-axis ticks to display years in 1-year intervals
years = pd.date_range(start='2019', end='2026', freq='YS').year
plt.xticks(pd.date_range(start='2019-01-01', end='2026-01-01', freq='YS'), years)

plt.show()
