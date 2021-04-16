import os
import numpy as np
import pandas as pd
import math
import datetime
from dateutil.relativedelta import *
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from csv import writer

ticker_list = ['AAPL', 'BA', 'GOOG', 'MSFT', 'WMT']

def load_ts_data(ts_dir):
    for index, ticker in enumerate(ticker_list):
        print('Loading Ticker ' + str(index+1) + '/' + str(len(ticker_list)) + '...')

        df = pd.read_csv(ts_dir + '/' + ticker + '.csv', index_col=0, header=None)
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={1: 'Adj Close', 2: 'Close', 3: 'High', 4: 'Low', 5: 'Open', 6: 'Volume'})
        df = df.fillna(method='ffill') # Fill data with previous day's -> Avoid lookahead bias

        # Stochastic oscillator (%K)
        rolling_low = df['Low'].rolling(window=14).min()
        rolling_high = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - rolling_low) / (rolling_high - rolling_low))

        # Larry William indicator (%R)
        df['%R'] = -100 * (rolling_high - df['Close']) / (rolling_high - rolling_low)

        #  Relative Strength Index (RSI)
        delta = df['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        _gain = up.ewm(com = 13, min_periods = 14).mean()
        _loss = down.abs().ewm(com = 13, min_periods = 14).mean()
        RS = _gain / _loss
        df['RSI'] = 100 - (100 / (1 + RS))

        # Drop unnecessary columns
        df = df.drop(columns=['Adj Close'])
        df = df.fillna(0)

        if index == 0:
            # Set an empty numpy array
            ts_data = np.zeros([len(ticker_list), df.shape[0] - 1, df.shape[1]], dtype=np.float32)
            gt_data = np.zeros([len(ticker_list), df.shape[0] - 1, 2], dtype=np.float32)

        delta_data = df['Close'].pct_change()[1:] # Get a dataframe of day-to-day change
        delta_data[delta_data < 0] = 0 # Map loss to value 0
        delta_data[delta_data > 0] = 1 # Map gain to value 1
        ohe = OneHotEncoder(categories = "auto", sparse = False)
        delta_data = ohe.fit_transform(pd.DataFrame(delta_data)) # Do one hot encoding

        ts_data[index, :, :] = df[:-1] # Timeseries array: No. of tickers x No. of days x 8 features
        gt_data[index, :, :] = delta_data # Ground Truth array: No. of tickers x No. of days x 2 (Up or Down)

        # Data Normalization
        scaler = StandardScaler()
        seq_len = 5
        train_ratio = 0.6

        for i in range(ts_data.shape[2]):
            ts_data[:, :math.ceil((ts_data.shape[1] - seq_len) * train_ratio)-1, i] = scaler.fit_transform(ts_data[:, :math.ceil((ts_data.shape[1] - seq_len) * train_ratio)-1, i].T).T # Train Data
            ts_data[:, math.ceil((ts_data.shape[1] - seq_len) * train_ratio):, i] = scaler.transform(ts_data[:, math.ceil((ts_data.shape[1] - seq_len) * train_ratio):, i].T).T # Valid / Test Data

    return ts_data, gt_data

def load_vec_data(vec_dir):
    with open('../data/news_vectors/vec.npy', 'rb') as f:
        vec_data = np.load(f)

    return vec_data

def load_kg_data(kg_dir):
    pass
