# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:10:13 2024

@author: PC
"""

import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('TSLA_price_x_senti_agg.xlsx')

df_tsla = df[['Date','vader_mean','vader_running_avg','Close_running_avg', 'Close', 'Volume','daily_movement', 'daily_volatility']]

# ---------------------------------------------------------------------------------

df_tsla.set_index('Date', inplace = True)

args = ['Volume', 'daily_movement', 'daily_volatility']

window = 5 

for i in args:
    df_tsla[i+'_running_avg'] = df_tsla[i].rolling(window=window).mean()

df_tsla.reset_index(inplace=True)

# --------------------------------------------------------------------------------

start_date = '2021-05-01'

df_tsla = df_tsla[(df_tsla['Date'] >= start_date)]

# -------------------------------------------------------------------------------

# create target - close price t+1 

df_tsla['Close_t_plus_1'] = df_tsla['Close'].shift(-1)

df_tsla.dropna(inplace=True)

df_tsla.to_excel('tsla_prep_for_ml.xlsx')

# ------------------------------------------------------------------------------
# binary classification TARGET - check if next day share price is higher = 1 or lower = 0

df_tsla['Price_Up'] = (df_tsla['Close_t_plus_1'] > df_tsla['Close']).astype(int)

# ------------------------------------------------------------------------------
# features selection 

features = ['vader_running_avg', 'Close_running_avg', 'daily_volatility', 'daily_movement_running_avg']

X = df_tsla[features]

# Target 
y = df_tsla['Price_Up']

# -------------------------------------------------------------------------------
# split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# train the model 
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# -------------------------------------------------------------------------------
# evaluate accuracy 

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

precision = precision_score(y_test, y_pred)
print(precision)

recall = recall_score(y_test, y_pred)
print(recall)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# -------------------------------------------------------------------------------

# Visualize confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['UP', 'DOWN'], yticklabels=['UP', 'DOWN'])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Confusion Matrix')
plt.show()








