import numpy as np
import pandas as pd
import math
import datetime
from dateutil.relativedelta import *
from sklearn.preprocessing import StandardScaler
from csv import writer

def load_data(data_dir):
    # Data Folder looks like this:
    # data_dir
    # -> /available_data.csv (List of tickers)
    # -> /historical (Folder containing all .csv files)
    available_tickers = np.genfromtxt(data_dir + '/available_data.csv', skip_header=True)

    for index, ticker in enumerate(available_tickers):
        print('Loading Ticker ' + str(index+1) + '/' + str(len(available_tickers)) + '...')

        df = pd.read_csv(data_dir + '/historical/' +  ticker + '.csv').set_index('Date')
        df.index = pd.to_datetime(df.index)
        df = df.fillna(method='ffill') # Fill data with previous day's -> Avoid lookahead bias

        if index == 0:
            ts_data = np.zeros([len(available_tickers), df.shape[0], df.shape[1]], dtype=np.float32) # Timeseries Shape: ??? days * 6 features
            gt_data = np.zeros([len(available_tickers), df.shape[0],  dtype=np.float32) # Ground Truth Shape: ??? days * 1 (close price)

        ts_data[index, :, :] = df[:-1]
        gt_data[index, :] = df[4, 1:]

    return ts_data, gt_data

def load_kg_embeddings(kg_dir):
    pass
