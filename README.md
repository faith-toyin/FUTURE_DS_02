## SALES FORECASTING

### CONTENT:
* [INTRODUCTION](#introduction)
* [DATA COLLECTION](#data-collection)
* [DATA PREPROCESSING](#data-preprocessing)
* [EXPLORATORY DATA ANALYSIS](#exploratory-data-analysis)
* [MODEL EVALUATION](#model-evaluation)
* [PREDICTIONS](#predictions)
* [DATA VISUALIZATION](#data-visualization)
* [SUMMARY](#summary)


#### INTRODUCTION
The purpose of this analysis is to leverage data analytics and machine learning techniques to predict future sales trends. Accurate sales forecasting is crucial for businesses to optimize inventory management, enhance customer satisfaction, and improve overall financial performance. In this project, we will explore various methodologies, including time series analysis and regression models, to analyze historical sales data and generate reliable forecasts. The insights gained from this project will empower stakeholders to make informed decisions and strategically plan for the future.
The sales forecasting data includes several key columns: 
1. Date: This column records the specific date for each sales entry, allowing us to analyze trends over time.
2. Sales: This represents the total number of units sold on that particular date, serving as the primary variable we aim to forecast.
3. OnPromotion: This column indicates whether the products were on promotion during the recorded date, which can significantly impact sales figures.
4. Family: This refers to the category or family of products, helping to segment sales data and analyze trends across different product types.


#### DATA COLLECTION
The dataset was downloaded from Kaggle and the specific dataset used is the "Store Sales-Time Series Forecasting" dataset, which contains dates, store and product information, whether that item was being promoted, as well as the sales numbers. The dataset is publicly available and can be accessed [Here](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)


#### DATA PREPROCESSING
Data preprocessing is a crucial step in sales forecasting as it ensures the quality and reliability of the data used for building predictive models. It involves:
* Identify and address missing values in the dataset.
  ```
    df.isnull().sum()
  ```
  ![image](https://github.com/user-attachments/assets/5698db49-f47f-4f50-a2c6-cfcca44fdeec)
* Ensuring there are no duplicate records in the dataset, as they can skew the analysis and predictions.
```
   df.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/6552cd86-7e32-4d1c-afc4-33b7cb2fbe88)
* Convert categorical variables (Date, family, onpromotion) into numerical format using techniques like one-hot encoding or label encoding.
```
categorical_features = ["date", "family", "onpromotion"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  categorical_features)],
                                  remainder="passthrough")
transformed_x = transformer.fit_transform(x)
transformed_x
```
![image](https://github.com/user-attachments/assets/5f8e9359-42a0-4683-88d7-403e099620bf)
* Split the dataset into training and testing sets to evaluate the model's performance.
```
np.random.seed(42)
x_train, x_test, y_train, y_test = train_test_split(transformed_x,
                                            y,
                                            test_size=0.2)

model.fit(x_train, y_train)
```


#### EXPLORATORY DATA ANALYSIS
It involves examining the dataset to uncover patterns, trends, and anomalies that can inform the forecasting model.
```
#descriptive statistics
df.describe()
```
![image](https://github.com/user-attachments/assets/c6b8c6e9-1b5d-4f53-8d47-20ea967acaf4)

Visualizing the distribution of sales data using histograms, linechart, and box plots to understand the spread and identify any outliers.
```
data = {
    'Date': pd.date_range(start='2013-01-01', periods=100, freq='D'),
    'Sales': [100 + i * 5 + (i % 7) * 10 for i in range(100)]
}
df = pd.DataFrame(data)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as index
df.set_index('Date', inplace=True)
```
```
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Sales'], label='Sales')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/3718aa59-6cb7-4ecc-a764-a258ae296843)
```
plt.figure(figsize=(12, 6))
sns.histplot(df['Sales'], kde=True)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/user-attachments/assets/49b70951-d758-4019-b5cd-ce025e13824a)
```
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Sales'])
plt.title('Sales Boxplot')
plt.xlabel('Sales')
```
![image](https://github.com/user-attachments/assets/dc2f005b-3811-4671-b0e9-44193d33ba59)

Examine seasonal patterns by aggregating sales data by weekday. This helps in identifying recurring patterns that can impact sales.
```
df['Weekday'] = df.index.weekday
plt.figure(figsize=(12, 6))
sns.boxplot(x='Weekday', y='Sales', data=df)
plt.title('Sales by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Sales')
plt.show()
```
![image](https://github.com/user-attachments/assets/42ec7bf7-e542-4cb2-be77-70ce870d3b18)

Creating a correlation matrix to identify relationships between sales, promotions, product families. This can help in understanding how different factors influence sales.
```
data = {
    'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Sales': [100 + i * 5 + (i % 7) * 10 for i in range(100)],
    'Promotions': [10 + (i % 5) * 2 for i in range(100)],
    'Product_Family': [1 if i % 2 == 0 else 2 for i in range(100)]
}
df = pd.DataFrame(data)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as index
df.set_index('Date', inplace=True)
```
```
corr_matrix = df.corr()

# Plot correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
```
![image](https://github.com/user-attachments/assets/53281d20-2325-4128-b296-af5624a75826)


#### MODEL EVALUATION
It involves assessing the performance of the forecasting model to ensure its accuracy and reliability. it includes:
* Mean Absolute Error (MAE): It Measures the average absolute difference between the actual and predicted values. Lower MAE indicates better model performance.
* Mean Squared Error (MSE):It Measures the average squared difference between the actual and predicted values. Lower MSE indicates better model performance.
* Root Mean Squared Error (RMSE): The square root of MSE, providing a measure of the average magnitude of the error. Lower RMSE indicates better model performance.
```
# Define the model
model = ARIMA(df['Sales'], order=(5, 1, 0))

# Fit the model
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())
```
![image](https://github.com/user-attachments/assets/67b3cabc-e6a2-4da1-83c7-625db9b852df)
```
# Calculate MAE
mae = mean_absolute_error(actual, forecast)
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate MSE
mse = mean_squared_error(actual, forecast)
print(f'Mean Squared Error (MSE): {mse}')

# Calculate RMSE
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')
```
![image](https://github.com/user-attachments/assets/c4463cb1-2f80-4b08-a3b3-7b2e63487eb8)


#### PREDICTIONS
```
forecast = model_fit.forecast(steps=10)

# Actual values for comparison (using the last 10 values of the dataset)
actual = df['Sales'][-10:]

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Sales'], label='Sales')
plt.plot(pd.date_range(start=df.index[-1], periods=11, freq='D')[1:], forecast, label='Forecast')
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/818cc53c-20b9-4bea-91ee-58a97a0c3854)


#### SUMMARY
The sales forecast analysis reveals a consistent upward trend in sales, indicating steady growth. Seasonal patterns and promotions significantly impact sales, highlighting the importance of optimizing stock levels and marketing strategies. Different product families and stores exhibit varying sales performance, providing opportunities to tailor strategies. The forecasted sales closely align with actual sales data, ensuring reliable predictions for informed decision-making and strategic planning.

