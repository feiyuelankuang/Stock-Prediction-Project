import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from load_data import load_ts_data, load_vec_data, load_kg_data
import random
import math
import time
from tensorflow.python.ops.nn_ops import leaky_relu

class LSTM:
    def __init__(self, ts_dir, vec_dir, kg_dir, parameters):
        self.parameters = parameters
        self.ts_data, self.gt_data = load_ts_data(ts_dir)
        self.vec_data = load_vec_data(vec_dir)

        # self.kg_data = load_kg_data(kg_dir)
        self.kg_data = np.zeros(self.ts_data.shape)

        self.batch_size = self.ts_data.shape[0] # Fixed batch size (can be changed)
        self.valid_index = math.ceil(self.ts_data.shape[1] * 0.6) # 60-20-20 Split
        self.test_index = math.ceil(self.ts_data.shape[1] * 0.8)

    def get_batch(self, batch_index):
        return self.ts_data[:, batch_index, :], self.vec_data[:, batch_index, :, :], self.kg_data[:, batch_index, :], self.gt_data[:, batch_index]

    def train(self):
        # device_name = '/cpu:0'
        device_name = '/gpu:0'
        print('device name:', device_name)

        with tf.device(device_name):
            tf.reset_default_graph()

            ts_feature = tf.placeholder(tf.float32, [self.batch_size, self.ts_data.shape[2]])
            vec_feature = tf.placeholder(tf.float32, [self.batch_size, self.vec_data.shape[2], self.vec_data.shape[3]])
            kg_feature = tf.placeholder(tf.float32, [self.batch_size, self.kg_data.shape[2]])
            ground_truth = tf.placeholder(tf.float32, [self.batch_size, 2])

            # TS embeddings
            ts_embedding = ts_feature

            # Vec embeddings
            expanded_vec = tf.expand_dims(vec_feature, -1)
            sequence_length = vec_feature.shape[1].value
            embedding_size = vec_feature.shape[2].value
            filter_size = 3
            num_filters = 128

            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(expanded_vec, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            vec_embedding = tf.squeeze(tf.layers.dense(pooled, units=50, activation=tf.nn.leaky_relu))

            # # KG embeddings
            # kg_embedding = tf.layers.dense(kg_feature, units=16, activation=leaky_relu)

            # Combined embeddings + Prediction
            stock_embedding = tf.concat([ts_embedding, vec_embedding], axis=1)
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
        best_pred = np.zeros([self.batch_size, self.batch_size], dtype=np.float32)

        for epoch in range(self.parameters['epochs']):
            print('-----Epoch #' + str(epoch) + '-----')
            t1 = time.time()

            ### TRAINING SET ###
            batch_index = np.arange(start=0, stop=self.valid_index, dtype=int)
            np.random.shuffle(batch_index) # Randomly shuffle training set each epoch

            train_loss = 0.0

            for i in range(0, self.valid_index):
                ts_batch, vec_batch, kg_batch, gt_batch = self.get_batch(batch_index[i])

                feed_dict = {
                    ts_feature: ts_batch,
                    vec_feature: vec_batch,
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
                ts_batch, vec_batch, kg_batch, gt_batch = self.get_batch(i)

                feed_dict = {
                    ts_feature: ts_batch,
                    vec_feature: vec_batch,
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
                ts_batch, vec_batch, kg_batch, gt_batch = self.get_batch(i)

                feed_dict = {
                    ts_feature: ts_batch,
                    vec_feature: vec_batch,
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
