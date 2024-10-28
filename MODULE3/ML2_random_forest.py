# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:46:35 2024

@author: PC
"""

import pandas as pd 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df_tsla = pd.read_excel('tsla_prep_for_ml.xlsx')

# --------------------------------------------------------------------------------
# features selection 

f1 = ['vader_running_avg', 'Close_running_avg']
f2 = ['vader_running_avg', 'Close_running_avg', 'Close']
f3 = ['vader_running_avg', 'Close_running_avg', 'Close', 'daily_volatility_running_avg', 'Volume_running_avg']
f4 = ['vader_running_avg', 
      'Close_running_avg', 
      'Volume_running_avg',
      'daily_movement_running_avg'
      ]


X = df_tsla[f4]

# Target 
y = df_tsla['Close_t_plus_1']

# --------------------------------------------------------------------------------
# split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)

# train the model 
logreg = RandomForestRegressor(n_estimators=100, random_state=333)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# --------------------------------------------------------------------------------
# evaluate using MSE

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# results 
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

results['Diff'] = results['Actual'] - results['Predicted']

# -------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Sort the 'Diff' column in descending order and reset the index
results_sorted = results.sort_values(by='Diff', ascending=False).reset_index(drop=True)

# Create a line chart for the sorted 'Diff' values
plt.figure(figsize=(10, 6))

# Plot the sorted 'Diff' values
plt.plot(results_sorted.index, results_sorted['Diff'], label='Diff', color='green', linestyle='-', marker='o')

# Adding labels and title
plt.xlabel('Index (sorted by Diff)')
plt.ylabel('Difference')
plt.title('Line Chart of Differences (Descending Order)')

# Show legend
plt.legend()

# Display the plot
plt.show()

# -------------------------------------------------------------------------------------




print("Predicting Tesla Stock Price Movements through Sentiment Analysis of Yahoo Finance Comments".upper())




