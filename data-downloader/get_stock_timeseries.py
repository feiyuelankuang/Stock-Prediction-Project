import numpy as np
import pandas as pd
import datetime
import yfinance as yf

ts_folder = '../data/timeseries_data/'

start_date = datetime.datetime(2011, 9, 1)
end_date = datetime.datetime(2021, 4, 2)

tickers_str = 'AAPL MSFT SSNLF BA GOOG WMT' # Note: SSNLF not available

stock_data = yf.download(tickers_str, start=start_date, end=end_date, threads=False)

# Save data
APPL_data = stock_data.xs('AAPL', axis=1, level=1, drop_level=False)
APPL_data.to_csv(ts_folder + 'AAPL.csv', header=False)

BA_data = stock_data.xs('BA', axis=1, level=1, drop_level=False)
BA_data.to_csv(ts_folder + 'BA.csv', header=False)

GOOG_data = stock_data.xs('GOOG', axis=1, level=1, drop_level=False)
GOOG_data.to_csv(ts_folder + 'GOOG.csv', header=False)

MSFT_data = stock_data.xs('MSFT', axis=1, level=1, drop_level=False)
MSFT_data.to_csv(ts_folder + 'MSFT.csv', header=False)

WMT_data = stock_data.xs('WMT', axis=1, level=1, drop_level=False)
WMT_data.to_csv(ts_folder + 'WMT.csv', header=False)
