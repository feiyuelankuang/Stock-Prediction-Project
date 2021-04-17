import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from load_data import load_ts_data, load_vec_data, load_kg_data
import random
import math
import time
from tensorflow.python.ops.nn_ops import leaky_relu
from sklearn.metrics import accuracy_score, f1_score
import csv

class LSTM:
    def __init__(self, ts_dir, vec_dir, kg_dir, parameters,stock_index=0):
        self.parameters = parameters
        self.ts_data, self.gt_data = load_ts_data(ts_dir)
        self.filename = 'LSTM'
        company = ['AAPL', 'BA', 'GOOG', 'MSFT', 'WMT']
        if stock_index is not None:
            self.filename += ('_'+ company[stock_index])
        #self.vec_data = load_vec_data(vec_dir)

        # self.kg_data = load_kg_data(kg_dir)
        self.kg_data = np.zeros(self.ts_data.shape)
        #self.kg_data = self.kg_data[stock_index]
        self.ts_data = self.ts_data[stock_index]
        self.kg_data = self.kg_data[stock_index]
        self.gt_data = self.gt_data[stock_index]

        self.batch_size = 32 # Fixed batch size (can be changed)

        self.seq = self.parameters['seq_len']
        self.total = math.ceil((self.ts_data.shape[1] - self.seq)/32)# 60-20-20 Split
        self.valid_index = math.ceil(self.total*0.6)
        self.test_index = math.ceil(self.total*0.8)


    def get_batch(self, batch_index):
        ts_data=[]
        kg_data=[]
        gt_data=[]
        batch_index = self.seq + self.batch_size*batch_index
        for i in range(self.batch_size):
            ts_data.append(self.ts_data[batch_index - self.seq + i : batch_index + i,:])
            kg_data.append(self.kg_data[batch_index - self.seq + i : batch_index + i,:])
            gt_data.append(self.gt_data[batch_index + i])
        #print(np.stack(gt_data, axis=0).shape)
        return np.stack(ts_data, axis=0),np.stack(kg_data, axis=0),np.stack(gt_data, axis=0)

    def train(self):
        device_name = '/gpu:0'
        #device_name = 'cpu'
        print('device name:', device_name)

        with tf.device(device_name):
            tf.reset_default_graph()
            #print(self.ts_data.shape)

            ts_feature = tf.placeholder(tf.float32, [self.batch_size, self.seq, self.ts_data.shape[1]])
            #vec_feature = tf.placeholder(tf.float32, [self.batch_size, self.seq, self.vec_data.shape[2], self.vec_data.shape[3]])
            kg_feature = tf.placeholder(tf.float32, [self.batch_size, self.seq, self.kg_data.shape[1]])
            ground_truth = tf.placeholder(tf.float32, [self.batch_size, 2])

            # TS embeddings
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(16)
            initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, ts_feature, dtype=tf.float32, initial_state=initial_state)
            ts_embedding = outputs[:, -1, :]

            # Vec embeddings
            # expanded_vec = tf.expand_dims(vec_feature, -1)
            # sequence_length = vec_feature.shape[1].value
            # embedding_size = vec_feature.shape[2].value
            # filter_size = 3
            # num_filters = 128
            #
            # filter_shape = [filter_size, embedding_size, 1, num_filters]
            # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            # b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            # conv = tf.nn.conv2d(expanded_vec, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            # vec_embedding = tf.squeeze(tf.layers.dense(pooled, units=50, activation=tf.nn.leaky_relu))

            # # KG embeddings
            # kg_embedding = tf.layers.dense(kg_feature, units=16, activation=leaky_relu)

            # Combined embeddings + Prediction
            stock_embedding = ts_embedding #tf.concat([ts_embedding, vec_embedding], axis=1)
            binary_pred = tf.layers.dense(stock_embedding, units=2, activation=tf.nn.softmax)
            prediction = tf.cast(binary_pred, dtype=tf.float32)

            # Loss
            loss = tf.losses.softmax_cross_entropy(ground_truth, prediction)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters['lr']).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # To save and extract
        best_valid_loss = np.inf
        best_acc = 0
        best_f1 = 0
        best_pred = np.zeros([self.batch_size, self.batch_size], dtype=np.float32)

        for epoch in range(self.parameters['epochs']):
            print('-----Epoch #' + str(epoch) + '-----')
            t1 = time.time()

            ### TRAINING SET ###
            batch_index = np.arange(start=0, stop=self.valid_index, dtype=int)
            np.random.shuffle(batch_index) # Randomly shuffle training set each epoch

            train_loss = 0.0

            for i in range(self.valid_index):
                #ts_batch, vec_batch, kg_batch, gt_batch = self.get_batch(batch_index[i - self.seq])
                ts_batch, kg_batch, gt_batch = self.get_batch(batch_index[i])

                feed_dict = {
                    ts_feature: ts_batch,
                    #vec_feature: vec_batch,
                    kg_feature: kg_batch,
                    ground_truth: gt_batch
                }

                curr_loss, curr_pred, batch_out = sess.run((loss, prediction, optimizer), feed_dict)
                train_loss += curr_loss

            print('Train Loss:', train_loss / self.valid_index / self.batch_size)


            ### VALIDATION SET ###
            val_loss = 0.0
            val_acc = 0.0

            for i in range(self.valid_index, self.test_index):
                ts_batch, kg_batch, gt_batch = self.get_batch(i)

                feed_dict = {
                    ts_feature: ts_batch,
                    #vec_feature: vec_batch,
                    kg_feature: kg_batch,
                    ground_truth: gt_batch
                }

                curr_loss, curr_pred = sess.run((loss, prediction), feed_dict)
                val_loss += curr_loss

            print(self.test_index - self.valid_index,batch_size)

            print('Valid Loss:', val_loss / (self.test_index - self.valid_index) / self.batch_size)


            ### TEST SET ###
            test_loss = 0.0
            test_acc = 0.0
            test_f1 = 0.0
            test_pred = np.zeros([self.ts_data.shape[1] - self.test_index, self.batch_size, 2], dtype=np.float32)
            test_batch = 0

            for i in range(self.test_index, self.total):
                #ts_batch, vec_batch, kg_batch, gt_batch = self.get_batch(i)
                ts_batch, kg_batch, gt_batch = self.get_batch(i)

                feed_dict = {
                    ts_feature: ts_batch,
                    #vec_feature: vec_batch,
                    kg_feature: kg_batch,
                    ground_truth: gt_batch
                }

                curr_loss, curr_pred = sess.run((loss, prediction), feed_dict)
                test_loss += curr_loss
                test_pred[test_batch, :, :] = curr_pred
                test_batch += 1
                #print('Test Loss:', test_loss / (self.ts_data.shape[1] - self.test_index))
                test_acc += accuracy_score(np.argmax(gt_batch, 1), np.argmax(curr_pred, 1))
                test_f1 += f1_score(np.argmax(gt_batch, 1), np.argmax(curr_pred, 1))

            print()
            print('Test Loss:', test_loss / (self.total - self.test_index) / self.batch_size)
            print('Acc:', test_acc / (self.total - self.test_index))
            print('F1:', test_f1 / (self.total - self.test_index))
            print()
            print('Took {:.3f}s.'.format(time.time() - t1))
            print()


            ### For tracking best performance ###
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                best_acc = test_acc / (self.ts_data.shape[1] - self.test_index)
                best_f1 = test_f1 / (self.ts_data.shape[1] - self.test_index)
                best_pred = test_pred

        print('Best accuracy:', best_acc)
        with open ('log/'+self.filename+'.csv', 'a') as csvfile:
        #writer.writerow('news','date','confidence','agent','predicate','object')
            writer = csv.writer(csvfile)
            writer.writerow([best_acc,best_f1])

class LSTM_KG:
    def __init__(self, ts_dir, kg_dir, model, combine, parameters,stock_index=None):
        self.parameters = parameters
        self.ts_data, self.gt_data = load_ts_data(ts_dir)        
        self.kg_data = load_kg_data(kg_dir+self.filename+'.npy')
        self.ts_data = self.ts_data[stock_index]
        self.gt_data = self.gt_data[stock_index]
        self.kg_data = self.kg_data[stock_index]
        self.kg_size = self.parameters['kg_size']
        self.hidden_size = self.parameters['hidden_size']
        model_dict={('TransE',True):'transE_combine', ('TransD',True):'transD_combine', ('TransE',False):'transE_KG', ('TransD',False):'transD_KG'}
        company = ['AAPL', 'BA', 'GOOG', 'MSFT', 'WMT']
        #print(model,combine)
        self.filename = model_dict[(model,combine)]
        print(kg_dir+self.filename+'.npy')
        if stock_index is not None:
            self.filename += ('_'+ company[stock_index])

        #print(self.kg_data.shape)
        #self.kg_data = np.zeros(self.ts_data.shape)

        self.batch_size = self.ts_data.shape[0] # Fixed batch size (can be changed)

        self.seq = self.parameters['seq_len']
        self.total = math.ceil((self.ts_data.shape[1] - self.seq)/32)# 60-20-20 Split
        self.valid_index = math.ceil(self.total*0.6)
        self.test_index = math.ceil(self.total*0.8)

    def get_batch(self, batch_index):
        ts_data=[]
        kg_data=[]
        gt_data=[]
        batch_index = self.seq + self.batch_size*batch_index
        for i in range(self.batch_size):
            ts_data.append(self.ts_data[batch_index - self.seq + i : batch_index + i,:])
            kg_data.append(self.kg_data[batch_index - self.seq + i : batch_index + i,:])
            gt_data.append(self.gt_data[batch_index + i])
        print(np.stack(gt_data, axis=0).shape)
        return np.stack(ts_data, axis=0),np.stack(kg_data, axis=0),np.stack(gt_data, axis=0)

                #self.vec_data[:, batch_index - self.seq : batch_index, :, :], \

    def train(self):
        device_name = '/gpu:0'
        #device_name = 'cpu'
        print('device name:', device_name)

        with tf.device(device_name):
            tf.reset_default_graph()

            ts_feature = tf.placeholder(tf.float32, [self.batch_size, self.seq, self.ts_data.shape[1]])
            #vec_feature = tf.placeholder(tf.float32, [self.batch_size, self.seq, self.vec_data.shape[2], self.vec_data.shape[3]])
            kg_feature = tf.placeholder(tf.float32, [self.batch_size, self.seq, self.kg_data.shape[1]])
            ground_truth = tf.placeholder(tf.float32, [self.batch_size, 2])

            # # KG embeddings
            kg_embedding = tf.layers.dense(kg_feature, units=self.kg_size, activation=leaky_relu)

            # TS embeddings
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            concat_feature = tf.concat([ts_feature, kg_embedding], axis=2)
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, concat_feature, dtype=tf.float32, initial_state=initial_state)
            ts_embedding = outputs[:, -1, :]


            # Vec embeddings
            # expanded_vec = tf.expand_dims(vec_feature, -1)
            # sequence_length = vec_feature.shape[1].value
            # embedding_size = vec_feature.shape[2].value
            # filter_size = 3
            # num_filters = 128
            #
            # filter_shape = [filter_size, embedding_size, 1, num_filters]
            # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            # b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            # conv = tf.nn.conv2d(expanded_vec, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            # vec_embedding = tf.squeeze(tf.layers.dense(pooled, units=50, activation=tf.nn.leaky_relu))

            # Combined embeddings + Prediction
            stock_embedding = ts_embedding #tf.concat([ts_embedding, vec_embedding], axis=1)
            binary_pred = tf.layers.dense(stock_embedding, units=2, activation=tf.nn.softmax)
            prediction = tf.cast(binary_pred, dtype=tf.float32)

            # Loss
            loss = tf.losses.softmax_cross_entropy(ground_truth, prediction)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters['lr']).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # To save and extract
        best_valid_loss = np.inf
        best_acc = 0
        best_f1 = 0
        best_pred = np.zeros([self.batch_size, self.batch_size], dtype=np.float32)

        for epoch in range(self.parameters['epochs']):
            print('-----Epoch #' + str(epoch) + '-----')
            t1 = time.time()

            ### TRAINING SET ###
            batch_index = np.arange(start=0, stop=self.valid_index, dtype=int)
            np.random.shuffle(batch_index) # Randomly shuffle training set each epoch

            train_loss = 0.0

            for i in range(self.valid_index):
                #ts_batch, vec_batch, kg_batch, gt_batch = self.get_batch(batch_index[i - self.seq])
                ts_batch, kg_batch, gt_batch = self.get_batch(batch_index[self.seq])

                feed_dict = {
                    ts_feature: ts_batch,
                    #vec_feature: vec_batch,
                    kg_feature: kg_batch,
                    ground_truth: gt_batch
                }

                curr_loss, curr_pred, batch_out = sess.run((loss, prediction, optimizer), feed_dict)
                train_loss += curr_loss

            print('Train Loss:', train_loss / self.valid_index / self.batch_size)


            ### VALIDATION SET ###
            val_loss = 0.0
            val_acc = 0.0

            for i in range(self.valid_index, self.test_index):
                ts_batch, kg_batch, gt_batch = self.get_batch(i)

                feed_dict = {
                    ts_feature: ts_batch,
                    #vec_feature: vec_batch,
                    kg_feature: kg_batch,
                    ground_truth: gt_batch
                }

                curr_loss, curr_pred = sess.run((loss, prediction), feed_dict)
                val_loss += curr_loss

            print('Valid Loss:', val_loss / (self.test_index - self.valid_index) / self.batch_size)


            ### TEST SET ###
            test_loss = 0.0
            test_acc = 0.0
            test_f1 = 0.0
            test_pred = np.zeros([self.ts_data.shape[1] - self.test_index, self.batch_size, 2], dtype=np.float32)
            test_batch = 0

            for i in range(self.test_index, self.total):
                #ts_batch, vec_batch, kg_batch, gt_batch = self.get_batch(i)
                ts_batch, kg_batch, gt_batch = self.get_batch(i)

                feed_dict = {
                    ts_feature: ts_batch,
                    #vec_feature: vec_batch,
                    kg_feature: kg_batch,
                    ground_truth: gt_batch
                }

                curr_loss, curr_pred = sess.run((loss, prediction), feed_dict)
                test_loss += curr_loss
                test_pred[test_batch, :, :] = curr_pred
                test_batch += 1
                #print('Test Loss:', test_loss / (self.ts_data.shape[1] - self.test_index))
                test_acc += accuracy_score(np.argmax(gt_batch, 1), np.argmax(curr_pred, 1))
                test_f1 += f1_score(np.argmax(gt_batch, 1), np.argmax(curr_pred, 1))

            print()
            print('Test Loss:', test_loss / (self.total - self.test_index) / self.batch_size)
            print('Acc:', test_acc / (self.total - self.test_index))
            print('F1:', test_f1 / (self.total - self.test_index))
            print()
            print('Took {:.3f}s.'.format(time.time() - t1))
            print()


            ### For tracking best performance ###
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                best_acc = test_acc / (self.ts_data.shape[1] - self.test_index)
                best_f1 = test_f1 / (self.ts_data.shape[1] - self.test_index)
                best_pred = test_pred

        print('Best accuracy:', best_acc)
        with open ('log/'+self.filename+'.csv', 'a') as csvfile:
        #writer.writerow('news','date','confidence','agent','predicate','object')
            writer = csv.writer(csvfile)
            writer.writerow([best_acc,best_f1])
