import pandas as pd
import numpy as np

vec_arr = np.zeros([5, 2831, 2958, 300], dtype=np.float32)

# Loop through folder
for i, ticker in enumerate(['AAPL', 'BA', 'GOOG', 'MSFT', 'WMT']):
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

    combined_df = ts.join(vec)

    for row in combined_df.loc[combined_df.vec.isnull(), 'vec'].index:
        combined_df.at[row, 'vec'] = []

    new_vec = combined_df['vec'].reset_index(drop=True)

    for index, item in new_vec.iteritems():
        arr = np.array(item)
        if arr.shape[0] == 0:
            arr = np.zeros([2958, 300])
        arr = np.pad(arr, ((0, 2958 - arr.shape[0]), (0,0)), 'constant')
        vec_arr[i, index, :, :] = arr

# vec_arr.to_pickle('../data/news_vectors/vec.pkl')
with open('../data/news_vectors/vec.npy', 'wb') as f:
    np.save(f, vec_arr)
