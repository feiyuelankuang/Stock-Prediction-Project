import argparse
from model import LSTM

def main():
    ### Parse arguments here ###
    ts_dir = '../data/timeseries_data'
    vec_dir = '../data/news_vectors'
    kg_dir = '../saved_embeddings'
    parameters = {'hidden_units': 16, 'lr': 0.001, 'epochs': 10}
    ### Parse arguments here ###

    lstm = LSTM(
        ts_dir = ts_dir,
        vec_dir = vec_dir,
        kg_dir = kg_dir,
        parameters=parameters
    )
    lstm.train()

if __name__ == '__main__':
    main()
