import os
import numpy as np
import pandas as pd
import math
import datetime
from dateutil.relativedelta import *
from sklearn.preprocessing import StandardScaler
from csv import writer

def load_data(data_dir):
    for (_, _, filename) in os.walk(data_dir):
        ticker_files = filename

    for index, ticker_file in enumerate(ticker_files):
        print('Loading Ticker ' + str(index+1) + '/' + str(len(ticker_files)) + '...')

        df = pd.read_csv(data_dir + '/' + ticker_file, index_col=0, header=None)
        df.index = pd.to_datetime(df.index)
        df = df.fillna(method='ffill') # Fill data with previous day's -> Avoid lookahead bias

        if index == 0:
            ts_data = np.zeros([len(ticker_files), df.shape[0] - 1, df.shape[1]], dtype=np.float32) # Timeseries Shape: ??? days * 6 features
            gt_data = np.zeros([len(ticker_files), df.shape[0] - 1],  dtype=np.float32) # Ground Truth Shape: ??? days * 1 (close price)

        ts_data[index, :, :] = df[:-1]
        gt_data[index, :] = df[1:][2]

    return ts_data, gt_data

def load_kg_embeddings(kg_dir):
    pass
