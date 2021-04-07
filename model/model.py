import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from load_data import load_data, load_kg_embeddings
import random
import math
import time
from tensorflow.python.ops.nn_ops import leaky_relu

class model:
    def __init__(self, data_dir, kg_dir, parameters):
        self.parameters = parameters
        self.ts_data, self.gt_data = load_data(data_dir)
        # self.kg_data = load_kg_embeddings(kg_dir)
        self.kg_data = np.zeros(self.ts_data.shape)

        self.batch_size = self.ts_data.shape[0] # Fixed batch size (can be changed)
        self.valid_index = math.ceil(self.ts_data.shape[1] * 0.6) # 60-20-20 Split
        self.test_index = math.ceil(self.ts_data.shape[1] * 0.8)

    def get_batch(self, batch_index):
        return self.ts_data[:, batch_index, :], self.kg_data[:, batch_index, :], self.gt_data[:, batch_index]

    def train(self):
        # device_name = '/cpu:0'
        device_name = '/gpu:0'
        print('device name:', device_name)

        with tf.device(device_name):
            tf.reset_default_graph()

            ts_feature = tf.placeholder(tf.float32, [self.batch_size, self.ts_data.shape[2]])
            kg_feature = tf.placeholder(tf.float32, [self.batch_size, self.kg_data.shape[2]])
            ground_truth = tf.placeholder(tf.float32, [self.batch_size, 2])

            # TS embeddings
            ts_embedding = tf.layers.dense(ts_feature, units=2, activation=tf.nn.softmax)

            # # KG embeddings
            # kg_embedding = tf.layers.dense(kg_feature, units=16, activation=leaky_relu)

            # # TS + KG embeddings
            # kg_feature = tf.reshape(kg_feature, [self.batch_size, 16])
            # stock_embedding = tf.layers.dense(tf.concat([lstm_embedding, kg_embedding], axis=1), units=1, activation=leaky_relu)

            prediction = tf.cast(ts_embedding, dtype=tf.float32)

            # Loss
            loss = tf.losses.softmax_cross_entropy(ground_truth, prediction)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters['lr']).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # To save and extract
        best_valid_loss = np.inf
        best_acc = 0
        best_pred = np.zeros([self.batch_size, self.batch_size], dtype=np.float32)

        for epoch in range(self.parameters['epochs']):
            print('-----Epoch #' + str(epoch) + '-----')
            t1 = time.time()

            ### TRAINING SET ###
            batch_index = np.arange(start=0, stop=self.valid_index, dtype=int)
            np.random.shuffle(batch_index) # Randomly shuffle training set each epoch

            train_loss = 0.0

            for i in range(0, self.valid_index):
                ts_batch, kg_batch, gt_batch = self.get_batch(batch_index[i])

                feed_dict = {
                    ts_feature: ts_batch,
                    kg_feature: kg_batch,
                    ground_truth: gt_batch
                }

                curr_loss, curr_pred, batch_out = sess.run((loss, prediction, optimizer), feed_dict)
                train_loss += curr_loss

            print('Train Loss:', train_loss / self.valid_index)


            ### VALIDATION SET ###
            val_loss = 0.0
            val_acc = 0.0

            for i in range(self.valid_index, self.test_index):
                ts_batch, kg_batch, gt_batch = self.get_batch(i)

                feed_dict = {
                    ts_feature: ts_batch,
                    kg_feature: kg_batch,
                    ground_truth: gt_batch
                }

                curr_loss, curr_pred = sess.run((loss, prediction), feed_dict)
                val_loss += curr_loss

            print('Valid Loss:', val_loss / (self.test_index - self.valid_index))


            ### TEST SET ###
            test_loss = 0.0
            test_acc = 0.0
            test_pred = np.zeros([self.ts_data.shape[1] - self.test_index, self.batch_size, 2], dtype=np.float32)
            test_batch = 0

            for i in range(self.test_index, self.ts_data.shape[1]):
                ts_batch, kg_batch, gt_batch = self.get_batch(i)

                feed_dict = {
                    ts_feature: ts_batch,
                    kg_feature: kg_batch,
                    ground_truth: gt_batch
                }

                curr_loss, curr_pred = sess.run((loss, prediction), feed_dict)
                test_loss += curr_loss
                test_pred[test_batch, :, :] = curr_pred
                test_batch += 1

            print('Test Loss:', test_loss / (self.ts_data.shape[1] - self.test_index))
            print('Took {:.3f}s.'.format(time.time() - t1))
            print()


            ### For tracking best performance ###
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                best_acc = test_loss / (self.ts_data.shape[1] - self.test_index)
                best_pred = test_pred

        print('Best accuracy:', best_acc)
