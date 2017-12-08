import tensorflow as tf
import numpy as np

from tensorflow.python.ops import rnn, rnn_cell
from data_preprocessing import get_audio_dataset_features_labels

def get_batch(dataset, i, BATCH_SIZE):
	if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
		return dataset[i*BATCH_SIZE:, :]
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]


DATASET_PATH = 'G:/DL/tf_speech_recognition'

dataset_train_features, dataset_train_labels = get_audio_dataset_features_labels(DATASET_PATH, type='train')
# dataset_test_features, dataset_test_labels = get_audio_dataset_features_labels(DATASET_PATH, type='test')

print('dataset_train_features.shape:', dataset_train_features.shape, 'dataset_train_labels.shape:', dataset_train_labels.shape)


CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
NUM_CLASSES = len(CLASSES)
NUM_EXAMPLES = dataset_train_features.shape[0]
NUM_CHUNKS = dataset_train_features.shape[1]	# 161
CHUNK_SIZE = dataset_train_features.shape[2]	# 99 
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

	return output

logits = recurrent_neural_network(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer()
training = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())		# initialize all global variables, which includes weights and biases

	# training start
	for epoch in range(0, NUM_EPOCHS):
		total_cost = 0

		for i in range(0, int(NUM_EXAMPLES/BATCH_SIZE)):
			batch_x = get_batch(dataset_train_features, i, BATCH_SIZE)	# get batch of features of size BATCH_SIZE
			batch_y = get_batch(dataset_train_labels, i, BATCH_SIZE)	# get batch of labels of size BATCH_SIZE

			_, batch_cost = sess.run([training, loss], feed_dict={x: batch_x, y: batch_y})	# train on the given batch size of features and labels
			total_cost += batch_cost

		print("Epoch:", epoch, "\tCost:", total_cost)

		# predict validation accuracy after every epoch
		y_predicted = tf.nn.softmax(logits)
		correct = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y, 1))
		accuracy_function = tf.reduce_mean(tf.cast(correct, 'float'))
		accuracy_validation = accuracy_function.eval({x:dataset_validation_features, y:dataset_validation_labels})
		print("Validation Accuracy in Epoch ", epoch, ":", accuracy_validation)
		# training end



