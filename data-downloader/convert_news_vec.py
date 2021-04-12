import pandas as pd
import numpy as np

# Loop through folder
for ticker in (['AAPL', 'BA', 'GOOG', 'MSFT', 'WMT']):
    ts = pd.read_csv('../data/timeseries_data/' + ticker + '.csv', header=None)

    if ticker == 'AAPL':
        filename = 'apple_reuters'
    elif ticker == 'BA':
        filename = 'Boeing_news_CNN'
    elif ticker == 'GOOG':
        filename = 'google_news_CNN'
    elif ticker == 'MSFT':
        filename = 'microsoft_reuters'
    elif ticker == 'WMT':
        filename = 'walmart_news_CNN'

    vec = pd.read_pickle('../data/news_vectors/' + filename + '.pkl')

    ts = ts.rename(columns={0:'time'})
    ts['time'] = pd.to_datetime(ts['time'])
    ts = ts.set_index(['time'])

    vec = vec.sort_values(by=['time'])
    vec['time'] = pd.to_datetime(vec['time'])
    vec['time'] = vec['time'].dt.date
    vec = vec.set_index(['time'])
    vec = vec.groupby('time').agg({'vec': 'sum'})

    new_vec = ts.join(vec)

    for row in new_vec.loc[new_vec.vec.isnull(), 'vec'].index:
        new_vec.at[row, 'vec'] = []

    new_vec.to_pickle('../data/news_vectors/' + ticker + '.pkl')
