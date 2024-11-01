# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:17:18 2024

@author: PC
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from keras.models import Sequential 
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns 

# Load the dataset
df = pd.read_excel('tsla_prep_for_ml.xlsx')

# Define features for the model
f4 = ['vader_running_avg', 
      'Close_running_avg', 
      'Volume_running_avg',
      'daily_movement_running_avg', 
      'Close'
      ]

# Select the relevant columns from the DataFrame
df_tsla = df[f4]

# Preprocess the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_prep = scaler.fit_transform(df_tsla)

# Create sequences for LSTM - for LSTM we need to reshape the data 
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i :i+n_steps])  # Use all features for the input sequence
        y.append(data[i+n_steps, df_tsla.columns.get_loc('Close')])  # Target is the next day's closing price
    return np.array(X), np.array(y)

# Specify the number of steps for the prediction
n_steps = 5  # Number of days for prediction

# Create sequences
X_lstm, y_lstm = create_sequences(df_prep, n_steps)

# Reshape X_lstm for LSTM input
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], len(f4)))  # Adjusted for number of features

# Train-test split 
split = int(0.8 * len(X_lstm))
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(n_steps, len(f4))))  # Adjusted input shape
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=2)

# Predictions
y_pred_lstm = model.predict(X_test_lstm)

# Prepare a full array for inverse transformation
full_pred = np.zeros((y_pred_lstm.shape[0], len(f4)))  # Create an array of zeros
full_pred[:, df_tsla.columns.get_loc('Close')] = y_pred_lstm.flatten()  # Set the predicted prices in the correct column

# Inverse transform to get actual price predictions
y_pred_lstm = scaler.inverse_transform(full_pred)[:, df_tsla.columns.get_loc('Close')]  # Get the actual predicted prices

# Prepare predictions DataFrame
df_prediction = pd.DataFrame(y_pred_lstm, columns=['predicted_price'])
df_prediction.index = df_prediction.index + split + n_steps
df_tsla_pred = df.join(df_prediction, how='left')

# Evaluate the model
y_test = df_tsla_pred.loc[df_tsla_pred['predicted_price'].notnull(), 'Close']
y_pred = df_tsla_pred.loc[df_tsla_pred['predicted_price'].notnull(), 'predicted_price']

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output metrics
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R2: {r2}')

# --------------------------------------------------------------------------------------------

# df_tsla_pred.reset_index(inplace=True)

plt.figure(figsize=(12, 6))

sns.lineplot(data=df_tsla_pred, x='Date',
             y='Close',
             label = 'actual close price', 
             color = 'orange'
             )

sns.lineplot(data=df_tsla_pred, x='Date',
             y='predicted_price',
             label = 'predicted close price', 
             color = 'green'
             )


plt.title('Prediction')
plt.xlabel('Date')
plt.ylabel('Close Share Price')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.title('TSLA Close price prediction')

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------

df_pred_2024 = df_tsla_pred[df_tsla_pred['Date'].dt.year == 2024]

plt.figure(figsize=(12, 6))

sns.lineplot(data=df_pred_2024, x='Date',
             y='Close',
             label = 'actual close price', 
             color = 'orange'
             )

sns.lineplot(data=df_pred_2024, x='Date',
             y='predicted_price',
             label = 'predicted close price', 
             color = 'green'
             )


plt.title('Prediction')
plt.xlabel('Date')
plt.ylabel('Close Share Price')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.title('TSLA Close price prediction')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

# Create the first axis
ax1 = plt.gca()

# Plot actual close price and predicted close price on the primary y-axis
sns.lineplot(data=df_pred_2024, x='Date', y='Close', label='Actual Close Price', color='orange', ax=ax1)
sns.lineplot(data=df_pred_2024, x='Date', y='predicted_price', label='Predicted Close Price', color='green', ax=ax1)

# Create a secondary y-axis
ax2 = ax1.twinx()

# Plot vader_running_avg on the secondary y-axis
sns.lineplot(data=df_pred_2024, x='Date', y='vader_running_avg', label='Vader Running Avg', color='red', linestyle='--', ax=ax2)

# Set titles and labels
ax1.set_title('TSLA Close Price Prediction')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Share Price')
ax2.set_ylabel('Sentiment Score')

# Show legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Additional formatting
plt.grid()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------------------------------

df_sim = df_pred_2024[['Date','Close', 'Close_t_plus_1', 'predicted_price', 'vader_running_avg']].dropna(subset=['predicted_price'])
df_sim = df_sim.reset_index(drop=True)

# Define the buy_signal function
def buy_signal(row):
    close = row['Close']
    pred = row['predicted_price']
    if pred > close:
        return 1
    return None # Optional: specify "None" or another label for no signal

# Apply the buy_signal function row-wise
df_sim['buy_signal'] = df_sim.apply(buy_signal, axis=1)

df_sim['price_delta'] = df_sim['Close_t_plus_1'] - df_sim['Close']

df_sim['price_delta_prc'] = df_sim['Close_t_plus_1'] / df_sim['Close']

df_sim['price_delta_prc_buy_sig'] = df_sim['Close_t_plus_1'] / df_sim['Close'] * df_sim['buy_signal'] 

df_sim['price_delta_prc_buy_sig'] = df_sim['price_delta_prc_buy_sig'].fillna(1)

def gain_loss(num):
    if num > 0:
        return 1
    elif num < 0:
        return -1
    else:
        return 0
    
df_sim['gain_loss'] = df_sim['price_delta'].apply(gain_loss)

df_sim['Running_investment'] = None
df_sim.loc[0, 'Running_investment'] = 10000

df_sim['Return'] = df_sim['Running_investment'] * df_sim['price_delta_prc_buy_sig']

for i in range(1, len(df_sim)):
    # Current running investment is the previous row's running investment multiplied by (1 + current price_delta)
    df_sim.loc[i, 'Running_investment'] = df_sim.loc[i-1, 'Running_investment'] * (df_sim.loc[i, 'price_delta_prc_buy_sig'])


plt.figure(figsize=(14, 8))

# Plot Running Investment on primary y-axis
plt.plot(df_sim['Date'], df_sim['Running_investment'], label='Running Investment', color='red')
plt.title('Trading Strategy')
plt.xlabel('Date')
plt.ylabel('Running Investment', color='red')
plt.grid()
plt.xticks(rotation=45)

# Create secondary y-axis for Close Price
ax2 = plt.gca().twinx()  # Create a second y-axis
ax2.plot(df_sim['Date'], df_sim['Close'], label='Close Price', color='orange')
ax2.set_ylabel('TSLA Close Price', color='orange')

# Combine legends from both axes
lines, labels = plt.gca().get_legend_handles_labels()  # Get labels from primary axis
lines2, labels2 = ax2.get_legend_handles_labels()  # Get labels from secondary axis
# ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))

# Plot Running Investment on primary y-axis
plt.plot(df_sim['Date'], df_sim['Running_investment'], label='Running Investment', color='red')
plt.title('Trading Strategy')
plt.xlabel('Date')
plt.ylabel('Running Investment', color='red')
plt.grid()
plt.xticks(rotation=45)

# Mark buy signals on the Running Investment plot
buy_signal_dates = df_sim[df_sim['buy_signal'] == 1]['Date']
buy_signal_values = df_sim[df_sim['buy_signal'] == 1]['Running_investment']
plt.scatter(buy_signal_dates, buy_signal_values, color='green', label='Buy Signal', marker='o', s=50)

# Create secondary y-axis for Close Price
ax2 = plt.gca().twinx()  # Create a second y-axis
ax2.plot(df_sim['Date'], df_sim['Close'], label='Close Price', color='orange')
ax2.set_ylabel('TSLA Close Price', color='orange')

# Combine legends from both axes
lines, labels = plt.gca().get_legend_handles_labels()  # Get labels from primary axis
lines2, labels2 = ax2.get_legend_handles_labels()  # Get labels from secondary axis
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))

# Plot Running Investment on primary y-axis
plt.plot(df_sim['Date'], df_sim['Running_investment'], label='Running Investment', color='red')
plt.title('Trading Strategy')
plt.xlabel('Date')
plt.ylabel('Running Investment', color='red')
plt.grid()
plt.xticks(rotation=45)

# Create secondary y-axis for Close Price
ax2 = plt.gca().twinx()  # Create a second y-axis
ax2.plot(df_sim['Date'], df_sim['Close'], label='Close Price', color='orange')
ax2.set_ylabel('TSLA Close Price', color='orange')

# Mark buy signals on the Close Price plot (secondary axis)
buy_signal_dates = df_sim[df_sim['buy_signal'] == 1]['Date']
buy_signal_values_close = df_sim[df_sim['buy_signal'] == 1]['Close']
ax2.scatter(buy_signal_dates, buy_signal_values_close, color='gray', label='Buy Signal', marker='o', s=50)

# Combine legends from both axes
lines, labels = plt.gca().get_legend_handles_labels()  # Get labels from primary axis
lines2, labels2 = ax2.get_legend_handles_labels()  # Get labels from secondary axis
ax2.legend(labels + labels2, loc='upper left')

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))

# Plot Running Investment on primary y-axis
plt.plot(df_sim['Date'], df_sim['Running_investment'], label='Running Investment', color='red')
plt.title('Trading Strategy')
plt.xlabel('Date')
plt.ylabel('Running Investment', color='red')
plt.grid()
plt.xticks(rotation=45)

# Create secondary y-axis for Close Price
ax2 = plt.gca().twinx()  # Create a second y-axis
ax2.plot(df_sim['Date'], df_sim['Close'], label='Close Price', color='orange')
ax2.set_ylabel('TSLA Close Price', color='orange')

# Separate buy signals into gains and losses
buy_signal_gains = df_sim[(df_sim['buy_signal'] == 1) & (df_sim['gain_loss'] == 1)]
buy_signal_losses = df_sim[(df_sim['buy_signal'] == 1) & (df_sim['gain_loss'] == -1)]

# Plot gain signals in green
ax2.scatter(buy_signal_gains['Date'], buy_signal_gains['Close'], color='green', label='Buy Signal (Gain)', marker='o', s=50)

# Plot loss signals in red
ax2.scatter(buy_signal_losses['Date'], buy_signal_losses['Close'], color='red', label='Buy Signal (Loss)', marker='o', s=50)

# Combine legends from both axes
lines, labels = plt.gca().get_legend_handles_labels()  # Get labels from primary axis
lines2, labels2 = ax2.get_legend_handles_labels()  # Get labels from secondary axis
ax2.legend(labels + labels2, loc='upper left')

plt.tight_layout()
plt.show()



# Filter rows where buy_signal is 1
buy_signal_rows = df_sim[df_sim['buy_signal'] == 1]

# Count rows with gain_loss = 1 and gain_loss = -1
gain_count = buy_signal_rows[buy_signal_rows['gain_loss'] == 1].shape[0]
loss_count = buy_signal_rows[buy_signal_rows['gain_loss'] == -1].shape[0]

print(f"Number of gains (gain_loss = 1): {gain_count}")
print(f"Number of losses (gain_loss = -1): {loss_count}")

# Calculate average price_delta_prc_buy_sig for gain_loss = 1 (gains) within buy_signal_rows
avg_price_delta_gain_buy_signal = buy_signal_rows[buy_signal_rows['gain_loss'] == 1]['price_delta_prc_buy_sig'].mean()

# Calculate average price_delta_prc_buy_sig for gain_loss = -1 (losses) within buy_signal_rows
avg_price_delta_loss_buy_signal = buy_signal_rows[buy_signal_rows['gain_loss'] == -1]['price_delta_prc_buy_sig'].mean()

print(f"Average price_delta_prc_buy_sig for gains (gain_loss = 1) with buy_signal = 1: {avg_price_delta_gain_buy_signal}")
print(f"Average price_delta_prc_buy_sig for losses (gain_loss = -1) with buy_signal = 1: {avg_price_delta_loss_buy_signal}")








