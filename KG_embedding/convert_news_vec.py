import pandas as pd
import numpy as np
from get_news_vec import *
from model import *
import torch

vec_arr = []
root = '../data/'
max_len = 0
vec_dict = {}
transE = TranE(device,model_path=root+'/GoogleNews-vectors-negative300.bin',d_norm=2, gamma=1).to(device)
checkpoint = torch.load('trnsE.t7')
transE.load_state_dict(checkpoint['state_dict'])
transD = TranD(device,model_path=root+'/GoogleNews-vectors-negative300.bin',d_norm=2, gamma=1).to(device)
checkpoint = torch.load('trnsD.t7')
transD.load_state_dict(checkpoint['state_dict'])
vec_dict_e,vec_dict_combine_e = get_news_vec(transE)
vec_dict_d,vec_dict_combine_d = get_news_vec(transD)
print('file loaded')

# Loop through folder
def get_array(dict,vec_name,size=1412):
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

        vec = dict[filename]

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
                arr = np.zeros(size)
            arr_list.append(arr)
        #arr = np.pad(arr, ((0, 2958 - arr.shape[0]), (0,0)), 'constant')
        #vec_arr[i, index, :, :] = arr
    #if len(arr_list) > max_len:
    vec_arr.append(np.array(arr_list))
    print(len(arr_list))

# vec_arr.to_pickle('../data/news_vectors/vec.pkl')
    with open(root+'/news_vectors/'+vec_name+'.npy', 'wb') as f:
        np.save(f, np.array(vec_arr))

print('begin alignment')
get_array(vec_dict_e,'transE_KG',900)
get_array(vec_dict_d,'transD_KG',900)
#get_array(vec_dict_combine_e,'transE_combine')
#get_array(vec_dict_combine_d,'transD_combine')
print('finish all')