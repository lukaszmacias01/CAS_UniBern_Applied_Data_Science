<img src="https://github.com/lukaszmacias01/CAS_UniBern_Applied_Data_Science/raw/master/UniBern/uni%20bern.png" alt="Uni Bern" width="200" style="float:right;" />

# Final Project Report — CAS Applied Data Science  
## “Forecasting S&P 500 Index Daily Movements with Machine Learning. A Comparative Study of Random Forest and XGBoost”

---

## Project Abstract

This project focuses on forecasting the next-day price movement of the S&P 500 Index using two machine learning algorithms — Random Forest and XGBoost — applied to a diverse set of low-correlated features.

The main objectives are:
1. To predict the magnitude of the S&P 500’s daily price changes  
2. To predict the direction of the movement (up or down)

Both models were trained on a dataset containing daily closing prices along with a variety of engineered financial and macroeconomic features.

Although XGBoost had a slightly higher Mean Absolute Error compared to Random Forest, it demonstrated better performance in capturing the variance of daily price change magnitudes.

In terms of predicting the direction of price movements:
- **XGBoost** correctly predicted the direction on **54%** of days  
- **Random Forest** achieved **53%**

These findings are significant because they show that both models consistently outperform random guessing — a notable achievement given that short-term stock market predictions are widely regarded as extremely difficult.

---

> *This project was submitted as part of the CAS in Applied Data Science at the University of Bern.*

