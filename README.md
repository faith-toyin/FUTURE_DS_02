## SALES FORECASTING

### CONTENT:
* [INTRODUCTION](#introduction)
* [DATA COLLECTION](#data-collection)
* [DATA PREPROCESSING](#data-preprocessing)
* [EXPLORATORY DATA ANALYSIS](#exploratory-data-analysis)
* [BUILDING MODEL](#building-model)
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
