import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the Excel file
file_path = '/content/AIProject.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Filter out rows where 'fuji_apple_price' is 0
df_fuji = df[df['fuji_apple_price'] > 0]

# Set 'date_formatting' column to datetime format if not already
df_fuji['date_formatting'] = pd.to_datetime(df_fuji['date_formatting'])

# Define features and target variable for Fuji apples
X_fuji = df_fuji[['avg_temp', 'max_temp', 'min_temp', 'rainfall']]
y_fuji = df_fuji['fuji_apple_price']

# Split the data into training and testing sets
X_fuji_train, X_fuji_test, y_fuji_train, y_fuji_test = train_test_split(X_fuji, y_fuji, test_size=0.2, random_state=42)

# Train the regression model for Fuji apples
model_fuji = LinearRegression()
model_fuji.fit(X_fuji_train, y_fuji_train)

# Make predictions for Fuji apples
y_fuji_pred = model_fuji.predict(X_fuji_test)

# Evaluate the model for Fuji apples
mse_fuji = mean_squared_error(y_fuji_test, y_fuji_pred)
r2_fuji = r2_score(y_fuji_test, y_fuji_pred)

print(f'Fuji Apple - Mean Squared Error: {mse_fuji}')
print(f'Fuji Apple - R^2 Score: {r2_fuji}')

# Combine actual and predicted values into a DataFrame
results_fuji = pd.DataFrame({'Actual Fuji': y_fuji_test, 'Predicted Fuji': y_fuji_pred})

# Extract Year and Month from 'date_formatting' and add to DataFrame
results_fuji['YearMonth'] = df_fuji.loc[results_fuji.index, 'date_formatting'].apply(lambda x: x.strftime('%Y-%m'))

# Calculate monthly average for actual and predicted prices
avg_results_fuji = results_fuji.groupby('YearMonth').mean()

# Ensure all months are included by reindexing with a complete date range
full_date_range = pd.date_range(start=results_fuji['YearMonth'].min(), end=results_fuji['YearMonth'].max(), freq='M').strftime('%Y-%m')
avg_results_fuji = avg_results_fuji.reindex(full_date_range).interpolate()

# Plot the results
plt.figure(figsize=(16, 8))

# Plot actual and predicted prices
plt.plot(avg_results_fuji.index, avg_results_fuji['Actual Fuji'], label='Actual Fuji Apple Prices', marker='o')
plt.plot(avg_results_fuji.index, avg_results_fuji['Predicted Fuji'], label='Predicted Fuji Apple Prices', linestyle='--', marker='x')

plt.xlabel('Year-Month(2019 ~ 2023years)')
plt.ylabel('Price(won)')
plt.title('Actual vs Predicted Fuji Apple Prices Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
