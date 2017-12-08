import tensorflow as tf
import numpy as np

from tensorflow.python.ops import rnn, rnn_cell
from data_preprocessing import get_audio_dataset_features_labels

DATASET_PATH = 'G:/DL/tf_speech_recognition'


dataset_train_features, dataset_train_labels = get_audio_dataset_features_labels(DATASET_PATH, type='train')
# dataset_test_features, dataset_test_labels = get_audio_dataset_features_labels(DATASET_PATH, type='test')

print('dataset_train_features.shape:', dataset_train_features.shape, 'dataset_train_labels.shape:', dataset_train_labels.shape)


CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
NUM_CLASSES = len(CLASSES)
CHUNK_SIZE = dataset_train_features.shape[2]	# 99 
NUM_CHUNKS = dataset_train_features.shape[1]	# 161
NUM_EPOCHS = 100
BATCH_SIZE = 128
RNN_SIZE = 128

x = tf.placeholder(tf.float32, shape=[-1, ])
y = tf.placeholder(tf.float32, shape=[-1, NUM_CLASSES])

def recurrent_neural_network(x):
	lstm_cell_1 = rnn_cell.BasicLSTMCell(128)
	lstm_cell_2 = rnn_cell.BasicLSTMCell(192)
	lstm_cell_3 = rnn_cell.BasicLSTMCell(256)
	weights_1 = tf.Variable(tf.random_normal([256, NUM_CLASSES]), dtype=tf.float32)
	biases_1 = tf.Variable(tf.random_normal([NUM_CLASSES]), dtype=tf.float32)

	lstm_layer_1, lstm_layer_1_states = rnn.rnn(lstm_cell_1, x, dtype=tf.float32)
	lstm_layer_2, lstm_layer_2_states = rnn.rnn(lstm_cell_2, lstm_layer_1, dtype=tf.float32)
	lstm_layer_3, lstm_layer_3_states = rnn.rnn(lstm_cell_3, lstm_layer_2, dtype=tf.float32)

	output = tf.matmul(lstm_layer_3[-1], weights_1) + biases_1



