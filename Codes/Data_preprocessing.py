import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
	   'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
	            '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10'],
	   'temperature': [20, 21, 19, 18, 22, 23, 21, 20, 19, 18],
	   'rainfall': [100, 80, 90, 110, 70, 60, 85, 95, 105, 115],
	   'apple_price': [3000, 3200, 3100, 3050, 3300, 3400, 3250, 3150, 3100, 3000]
}

df = pd.DataFrame(data)

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

X = df[['year', 'month', 'day', 'dayofweek', 'temperature', 'rainfall']]
y = df['apple_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
  
df['predicted_price'] = model.predict(X)

plt.figure(figsize=(14, 7))

plt.subplot(3, 1, 1)
plt.plot(df['date'], df['temperature'], label='Temperature', color='tab:red')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper left')
plt.title('Temperature, Rainfall, and Predicted Apple Prices Over Time')

plt.subplot(3, 1, 2)
plt.plot(df['date'], df['rainfall'], label='Rainfall', color='tab:blue')
plt.ylabel('Rainfall (mm)')
plt.legend(loc='upper left')

plt.subplot(3, 1, 3)
plt.plot(df['date'], df['apple_price'], label='Actual Apple Price', color='tab:green')
plt.plot(df['date'], df['predicted_price'], label='Predicted Apple Price', color='tab:orange', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Apple Price (KRW)')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
df = pd.read_excel('AIProject.xlsx')
df.head()
place = ['geochang', 'chungju', 'pohang']

for i in place:
   path = i+'.csv'
   df = pd.read_csv(path, )

   # 'date' 열을 날짜 형식으로 변환
   df['date'] = pd.to_datetime(df['date'])
  
   # 'date' 열을 기준으로 연도별로 분리하여 저장
   for year in range(2013, 2024):
       year_df = df[df['date'].dt.year == year]
       year_df.to_csv(f'{i}_{year}.csv', index=False)

import os.path

def getData(location, year):
   yy = str(year)
   firstpath = location+'_'+yy+'.csv'

   price10 = None
   price11 = None
   # price12 = None

   if os.path.isfile("price.csv"):
       applePrice = pd.read_csv("price.csv")
       size = applePrice.shape
       for i, col_name in enumerate(applePrice.columns[1:]):
           if col_name == yy:
               price10 = int(applePrice.iloc[9, i+1].replace(',', '')) 
               price11 = int(applePrice.iloc[10, i+1].replace(',', ''))
               # price12 = int(applePrice.iloc[11, i+1].replace(',', '')) 
               break
  
   if price10 is None or price11 is None:
       print(f"{year}년 데이터를 찾을 수 없습니다.")
       return

   newData = pd.DataFrame({'year':[year], '10_price':[price10],'11_price':[price11]})

   if os.path.isfile(firstpath):
       rawData = pd.read_csv(firstpath)
       size = rawData.shape
  
   for week in range(0, int(size[0]/7)):
       avgC = 0
       highC = 0
       lowC =0
       rain = 0
       for day in range(0, 7):
           avgC += rawData.iloc[week*7+day][1]
           if day ==0:
               highC = rawData.iloc[week*7+day][2]
               lowC = rawData.iloc[week*7+day][3]
           else :
              if highC < rawData.iloc[week*7+day][2]:
                   highC = rawData.iloc[week*7+day][2]
               if lowC > rawData.iloc[week*7+day][3]:
                   lowC = rawData.iloc[week*7+day][3]
           rain += rawData.iloc[week*7+day][4]
       newData[str(week+1)+"_avg_temp"] = round(avgC/7, 1)
       newData[str(week+1)+"_high_temp"] = round(highC,1)
       newData[str(week+1)+"_low_temp"] = round(lowC,1)
       newData[str(week+1)+"_rainfall"] = round(rain/7,1)
   newData.to_csv("anz_"+location+"_"+yy+".csv", index=False)

for year in range(2013, 2024):
   getData('geochang', year)
   getData('chungju', year)
   getData('pohang', year)

def mergeData(year, location):
   yy = str(year)

   firstpath = "anz_"+location+"_"+yy+".csv"
   if os.path.isfile(firstpath):
       df = pd.read_csv(firstpath)
     
       for i in range(int(year)+1, 2024):
           filepath = "anz_"+location+"_"+str(i)+".csv"
           if os.path.isfile(filepath):
              df2 = pd.read_csv(filepath)
              df = pd.concat([df, df2])
       df.to_csv("Total_"+location+".csv", index=False)


for i in place:
   mergeData(2013, i)


import statsmodels.api as sm

df = pd.read_csv("Total_chungju.csv")

# 15 ~ 35주로 계산 할 경우
avg_temp_data = df.iloc[:, 59:143:4]
rainfall_data = df.iloc[:, 62:143:4]

# 전체 기간으로 계산할 경우
# avg_temp_data = df.iloc[:, 3::4]
# rainfall_data = df.iloc[:, 6::4]

# print(avg_temp_data)
# print(rainfall_data)

avg_temp_data = avg_temp_data.sum(axis=1)
rainfall_data = rainfall_data.sum(axis=1)

# 15~35주로 계산할 경우
rainfall_data = rainfall_data / 20
avg_temp_data = avg_temp_data / 20

# # 전체 기간으로 계산할 경우
# rainfall_data = rainfall_data / 52
# avg_temp_data = avg_temp_data / 52

df['average_price'] = (df['10_price'] + df['11_price']) / 2
df['avg_temp'] = avg_temp_data
df['avg_rainfall'] = rainfall_data

# print(avg_temp_data)
# print(rainfall_data)
# print(df['average_price'])

avg_price_by_year = df.groupby('year')['average_price'].mean()

X = df[['avg_temp', 'avg_rainfall']]
y = df['average_price']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())
