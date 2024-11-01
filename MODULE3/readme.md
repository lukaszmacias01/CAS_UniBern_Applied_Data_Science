# Predicting Tesla Stock Price Movements Using Selected Machine Learning Techniques, with a Focus on investors Sentiment expressed in comments section of Yahoo Finance
---

## Overview
This module explores the prediction of Tesla stock price movements using selected machine learning techniques, focusing on investors' sentiment as expressed in the comments section of Yahoo Finance. By analyzing sentiment scores alongside historical stock prices, we aim to identify patterns that may influence stock performance.

![Tesla Stock Price vs. Sentiment Score](https://github.com/lukaszmacias01/CAS_UniBern_Applied_Data_Science/raw/master/MODULE3/Visuals/line%20chart%20tsla%20x%20senti%20score%205%20days%20runnign%20avg.png)

## Objectives
- To develop a robust model for predicting Tesla's stock price movements.
- To incorporate sentiment analysis of investor comments from Yahoo Finance.
- To evaluate the performance of different machine learning techniques in forecasting stock prices.

## Data Sources
- **Tesla Stock Prices**: Historical stock price data sourced from Yahoo Finance.
- **Investor Sentiment**: Comments collected from the Yahoo Finance platform, processed to generate sentiment scores.

## Methodology
1. **Data Collection**:
   - Gather historical stock prices and comments related to Tesla from Yahoo Finance.
   - Preprocess the data to align sentiment scores with corresponding stock prices.

2. **Sentiment Analysis**:
   - Utilize natural language processing (NLP) techniques to analyze comments and derive sentiment scores.
   - Calculate daily average sentiment scores to match the stock price data.

3. **Machine Learning Techniques**:
   - Implement various machine learning algorithms, including:
     - Linear Regression
     - Random Forest
     - Long Short-Term Memory (LSTM) Networks
    
![LSTM Prediction - Full History](https://github.com/lukaszmacias01/CAS_UniBern_Applied_Data_Science/raw/master/MODULE3/Visuals/LSTM_prediction_full_history.png)

4. **Model Evaluation**:
   - Assess model performance using metrics such as Mean Squared Error (MSE) and R-squared.
   - Compare predictions against actual stock prices to determine accuracy.

5. **Trading Strategy Implementation**:
   - Utilize the LSTM model to develop a trading strategy that integrates sentiment analysis.
   - This strategy resulted in the outperformance of a benchmark, demonstrating its effectiveness in predicting stock price movements.

![LSTM Prediction - Trading Strategy]((https://github.com/lukaszmacias01/CAS_UniBern_Applied_Data_Science/blob/master/MODULE3/Visuals/trading_strategy_with_dots_gain_loss.png))


