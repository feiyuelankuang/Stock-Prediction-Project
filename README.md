# Stock-Prediction-Project

This Project is for CS6216 course, the aim is to use knowledge graph extracted from news to help prediction of stock trend(up or down)

## Report
report can be edit here https://www.overleaf.com/6175999788qykjydxqwhkb

## Getting Started

### Dependencies

numpy

panda

nltk

tensorflow

tenforflow_hub

tensorflow_probability 

pytorch

sklearn

## Running the code

1. Before running the code, please download word2vec model from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit, unzip it and put it under data folder


2. To build the model, firstly run python save_news_vector.py under kg_embedding folders


3.Then under model folder, run python main.py --model '' --combine '' to train stock prediction model

