import argparse
from lstm import LSTM

def main():
    ### Parse arguments here ###
    data_dir = '../data/timeseries_data'
    kg_dir = '../saved_embeddings'
    parameters = {'hidden_units': 16, 'lr': 0.001, 'epochs': 10}
    ### Parse arguments here ###

    lstm_model = LSTM(
        data_dir = data_dir,
        kg_dir = kg_dir,
        parameters=parameters
    )
    lstm_model.train()

if __name__ == '__main__':
    main()
