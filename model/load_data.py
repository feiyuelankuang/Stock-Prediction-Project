import os
import numpy as np
import pandas as pd
import math
import datetime
from dateutil.relativedelta import *
from sklearn.preprocessing import OneHotEncoder
from csv import writer

def load_ts_data(ts_dir):
    for (_, _, filename) in os.walk(ts_dir):
        ticker_files = filename

    for index, ticker_file in enumerate(ticker_files):
        print('Loading Ticker ' + str(index+1) + '/' + str(len(ticker_files)) + '...')

        df = pd.read_csv(data_dir + '/' + ticker_file, index_col=0, header=None)
        df.index = pd.to_datetime(df.index)
        df = df.fillna(method='ffill') # Fill data with previous day's -> Avoid lookahead bias

        if index == 0:
            ts_data = np.zeros([len(ticker_files), df.shape[0] - 1, df.shape[1]], dtype=np.float32)
            gt_data = np.zeros([len(ticker_files), df.shape[0] - 1, 2], dtype=np.float32)

        # Timeseries: No. of days x 6 features
        ts_data[index, :, :] = df[:-1]

        # Ground Truth: No. of days x 2 (Up or Down)
        delta_data = df[2].pct_change()[1:]
        delta_data[delta_data < 0] = 0 # Loss is 0
        delta_data[delta_data > 0] = 1 # Gain is 1
        ohe = OneHotEncoder(categories = "auto", sparse = False)
        delta_data = ohe.fit_transform(pd.DataFrame(delta_data))
        gt_data[index, :, :] = delta_data

    return ts_data, gt_data

def load_vec_data(vec_dir):
    


def load_kg_embeddings(kg_dir):
    pass
