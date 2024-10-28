# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:58:30 2024

@author: Lukasz Macias
"""

import pandas as pd 

df_f = pd.read_excel('FINAL sentiment and price per day.xlsx')
df_c = pd.read_excel('TSLA_sentiment.xlsx')

corr_vol_com = df_f['Volume'].corr(df_f['comments_count'])
corr_vol_com 

corr_vol_com = df_f['Close'].corr(df_f['comments_count'])
corr_vol_com 

corr_volat_com = df_f['daily_volatility'].corr(df_f['comments_count'])
corr_volat_com 

corr_vol_com = df_f['daily_movement'].corr(df_f['comments_count'])
corr_vol_com 

# Plotting Volume and comments_count over time with Volume on the secondary y-axis
fig, ax1 = plt.subplots(figsize=(10,6))

# Plotting comments_count on the primary y-axis
ax1.plot(df_f['Date'], df_f['comments_count'], label='Comments Count', color='green', marker='x')
ax1.set_xlabel('Date')
ax1.set_ylabel('Comments Count', color='green')
ax1.tick_params(axis='y', labelcolor='green')

# Creating a secondary y-axis for Volume
ax2 = ax1.twinx()
ax2.plot(df_f['Date'], df_f['Volume'], label='Volume', color='blue', marker='o')
ax2.set_ylabel('Volume', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Adding title and grid
plt.title('Volume and Comments Count Over Time (with Secondary Axis)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot
plt.show()


df_f['vader_mean'].describe()




import numpy as np

# Scatter plot with linear regression line

# Creating a scatter plot
plt.figure(figsize=(8,6))
plt.scatter(df_f['Volume'], df_f['comments_count'], color='orange', label='Data points')

# Fitting a linear regression line
m, b = np.polyfit(df_f['Volume'], df_f['comments_count'], 1)
plt.plot(df_f['Volume'], m*df_f['Volume'] + b, color='red', label='Linear Regression')

# Adding labels and title
plt.xlabel('Daily Volume')
plt.ylabel('Daily Comments Count')
plt.title('TSLA Daily Volume vs Daily Comments Count')
plt.legend()

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()




import numpy as np

# Scatter plot with linear regression line

# Creating a scatter plot
plt.figure(figsize=(8,6))
plt.scatter(df_f['daily_volatility'], df_f['comments_count'], color='orange', label='Data points')

# Fitting a linear regression line
m, b = np.polyfit(df_f['daily_volatility'], df_f['comments_count'], 1)
plt.plot(df_f['daily_volatility'], m*df_f['daily_volatility'] + b, color='red', label='Linear Regression')

# Adding labels and title
plt.xlabel('Daily Volatility (Daily MAX price - daily MIN price)')
plt.ylabel('Daily Comments Count')
plt.title('TSLA Daily Volatility vs Daily Comments Count')
plt.legend()

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()


df_cx = df_c[df_c['senti_tag'].isnull()]

plt.figure(figsize=(10, 6))

plt.hist(df_cx['vader_compound'], bins=30, alpha=0.5, label='VADER Score', color='orange', edgecolor='black')

plt.title('Histogram of VADER sentiment scores for comments without sentiment tag', fontsize=16)
plt.xlabel('VADER Sentiment Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(loc='upper right')
plt.show()
