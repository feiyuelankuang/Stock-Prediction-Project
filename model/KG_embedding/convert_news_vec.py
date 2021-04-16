import pandas as pd
import numpy as np

vec_arr = []
root = '../../data/'
max_len = 0
# Loop through folder
for i, ticker in enumerate(['AAPL', 'BA', 'GOOG', 'MSFT', 'WMT']):
    ts = pd.read_csv(root + 'timeseries_data/' + ticker + '.csv', header=None)

    if ticker == 'AAPL':
        filename = 'apple_reuters_result_processed'
    elif ticker == 'BA':
        filename = 'boeing_news_CNN_result_processed'
    elif ticker == 'GOOG':
        filename = 'google_news_CNN_result_processed'
    elif ticker == 'MSFT':
        filename = 'microsoft_reuters_result_processed'
    elif ticker == 'WMT':
        filename = 'walmart_news_CNN_result_processed'

    vec = pd.read_pickle(root + 'news_vectors/' + filename + '_combined_embedding.pkl')

    ts = ts.rename(columns={0:'time'})
    ts['time'] = pd.to_datetime(ts['time'])
    ts = ts.set_index(['time'])

    vec = vec.sort_values(by=['time'])
    vec['time'] = pd.to_datetime(vec['time'])
    vec['time'] = vec['time'].dt.date
    vec = vec.set_index(['time'])
    #vec = vec.groupby('time').mean({'vec': 'sum'})
    vec = vec.groupby('time')['vec'].apply(np.mean)

    combined_df = ts.join(vec)

    for row in combined_df.loc[combined_df.vec.isnull(), 'vec'].index:
        combined_df.at[row, 'vec'] = []

    new_vec = combined_df['vec'].reset_index(drop=True)

    arr_list = []
    
    for index, item in new_vec.iteritems():
        arr = np.array(item)
        #print(index,arr.shape)
        if arr.shape[0] == 0:
            arr = np.zeros(1412)
        arr_list.append(arr)
        #arr = np.pad(arr, ((0, 2958 - arr.shape[0]), (0,0)), 'constant')
        #vec_arr[i, index, :, :] = arr
    #if len(arr_list) > max_len:
    vec_arr.append(np.array(arr_list))
    print(len(arr_list))

# vec_arr.to_pickle('../data/news_vectors/vec.pkl')
with open(root+'/news_vectors/vec.npy', 'wb') as f:
    np.save(f, np.array(vec_arr))
