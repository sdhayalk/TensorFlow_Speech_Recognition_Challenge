import tensorflow as tf
import numpy as np
import random

# from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn
from data_preprocessing import get_audio_dataset_features_labels, get_audio_test_dataset_filenames, get_audio_test_dataset_features_labels, normalize_training_dataset, normalize_test_dataset

def shuffle_randomize(dataset_features, dataset_labels):
	dataset_combined = list(zip(dataset_features, dataset_labels))
	random.shuffle(dataset_combined)
	dataset_features[:], dataset_labels[:] = zip(*dataset_combined)
	return dataset_features, dataset_labels

def get_batch(dataset, i, BATCH_SIZE):
	if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
		return dataset[i*BATCH_SIZE:, :]
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]


DATASET_PATH = 'G:/DL/tf_speech_recognition'
ALLOWED_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
ALLOWED_LABELS_MAP = {}
for i in range(0, len(ALLOWED_LABELS)):
	ALLOWED_LABELS_MAP[str(i)] = ALLOWED_LABELS[i]

dataset_train_features, dataset_train_labels, labels_one_hot_map = get_audio_dataset_features_labels(DATASET_PATH, ALLOWED_LABELS, type='train')
audio_filenames = get_audio_test_dataset_filenames(DATASET_PATH)

print('dataset_train_features.shape:', dataset_train_features.shape, 'dataset_train_labels.shape:', dataset_train_labels.shape)

# normalize training and testing features dataset
print('Normalizing datasets')
dataset_train_features, min_value, max_value = normalize_training_dataset(dataset_train_features)

# randomize shuffle
print('Shuffling training dataset')
dataset_train_features, dataset_train_labels = shuffle_randomize(dataset_train_features, dataset_train_labels)

# divide training set into training and validation
dataset_validation_features, dataset_validation_labels = dataset_train_features[57000:dataset_train_features.shape[0], :], dataset_train_labels[57000:dataset_train_labels.shape[0], :]
dataset_train_features, dataset_train_labels = dataset_train_features[0:57000, :], dataset_train_labels[0:57000, :]
print('dataset_validation_features.shape:', dataset_validation_features.shape, 'dataset_validation_labels.shape:', dataset_validation_labels.shape)

CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
NUM_CLASSES = len(CLASSES)
NUM_EXAMPLES = dataset_train_features.shape[0]
NUM_CHUNKS = dataset_train_features.shape[1]	# 161
CHUNK_SIZE = dataset_train_features.shape[2]	# 99 
NUM_EPOCHS = 100
BATCH_SIZE = 32

x = tf.placeholder(tf.float32, shape=[None, NUM_CHUNKS, CHUNK_SIZE])
y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

weights = {
	'w_conv1': get_variable('w_conv1', shape=[3,3,1,32], dtype=tf.float32)
	'w_conv2': get_variable('w_conv2', shape=[3,3,32,64], dtype=tf.float32)
	'w_conv3': get_variable('w_conv3', shape=[3,3,64,128], dtype=tf.float32)
}
biases = {
	'b_conv1': get_variable('b_conv1', shape=[32], dtype=tf.float32)
	'b_conv2': get_variable('b_conv2', shape=[64], dtype=tf.float32)
	'b_conv3': get_variable('b_conv3', shape=[128], dtype=tf.float32)
}

def leakyrelu(x):
  return tf.nn.relu(x) - 0.2*tf.nn.relu(-x)

def recurrent_neural_network(x):

	lstm_cell_1_1 = rnn.LSTMCell(128, state_is_tuple=True)
	lstm_layer_1, lstm_layer_1_states = tf.nn.dynamic_rnn(lstm_cell_1_1, x, dtype=tf.float32)

	conv1 = tf.nn.conv2d(input, weights['w_conv1'], strides=[1,1,1,1], padding='SAME') + biases['b_conv1']
	conv1 = leakyrelu(conv1)
	conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	conv1_split1, conv1_split2, conv1_split3, conv1_split4 = tf.split(conv1, num_or_size_splits=4, axis=1)	# refer docs for tf.split here: https://www.tensorflow.org/api_docs/python/tf/split
	conv1_split1 = tf.reshape(conv1_split1, [-1, tf.shape(conv1_split1)[-2], tf.shape(conv1_split1)[-1]])
	conv1_split2 = tf.reshape(conv1_split2, [-1, tf.shape(conv1_split2)[-2], tf.shape(conv1_split2)[-1]])
	conv1_split3 = tf.reshape(conv1_split3, [-1, tf.shape(conv1_split3)[-2], tf.shape(conv1_split3)[-1]])
	conv1_split4 = tf.reshape(conv1_split4, [-1, tf.shape(conv1_split4)[-2], tf.shape(conv1_split4)[-1]])

	lstm_cell_2_1 = rnn.LSTMCell(32, state_is_tuple=True)
	lstm_cell_2_2 = rnn.LSTMCell(32, state_is_tuple=True)
	lstm_cell_2_3 = rnn.LSTMCell(32, state_is_tuple=True)
	lstm_cell_2_4 = rnn.LSTMCell(32, state_is_tuple=True)
	lstm_layer_2_1, lstm_layer_2_1_states = tf.nn.dynamic_rnn(lstm_cell_2_1, conv1_split1, dtype=tf.float32)
	lstm_layer_2_2, lstm_layer_2_2_states = tf.nn.dynamic_rnn(lstm_cell_2_2, conv1_split2, dtype=tf.float32)
	lstm_layer_2_3, lstm_layer_2_3_states = tf.nn.dynamic_rnn(lstm_cell_2_3, conv1_split3, dtype=tf.float32)
	lstm_layer_2_4, lstm_layer_2_4_states = tf.nn.dynamic_rnn(lstm_cell_2_4, conv1_split4, dtype=tf.float32)


	conv2 = tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1,1,1,1], padding='SAME') + biases['b_conv2']
	conv2 = leakyrelu(conv2)
	conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	conv2_split1, conv2_split2, conv2_split3, conv2_split4 = tf.split(conv2, num_or_size_splits=8, axis=1)	# refer docs for tf.split here: https://www.tensorflow.org/api_docs/python/tf/split
	conv2_split1 = tf.reshape(conv2_split1, [-1, tf.shape(conv2_split1)[-2], tf.shape(conv2_split1)[-1]])
	conv2_split2 = tf.reshape(conv2_split2, [-1, tf.shape(conv2_split2)[-2], tf.shape(conv2_split2)[-1]])
	conv2_split3 = tf.reshape(conv2_split3, [-1, tf.shape(conv2_split3)[-2], tf.shape(conv2_split3)[-1]])
	conv2_split4 = tf.reshape(conv2_split4, [-1, tf.shape(conv2_split4)[-2], tf.shape(conv2_split4)[-1]])
	conv2_split5 = tf.reshape(conv2_split5, [-1, tf.shape(conv2_split5)[-2], tf.shape(conv2_split5)[-1]])
	conv2_split6 = tf.reshape(conv2_split6, [-1, tf.shape(conv2_split6)[-2], tf.shape(conv2_split6)[-1]])
	conv2_split7 = tf.reshape(conv2_split7, [-1, tf.shape(conv2_split7)[-2], tf.shape(conv2_split7)[-1]])
	conv2_split8 = tf.reshape(conv2_split8, [-1, tf.shape(conv2_split8)[-2], tf.shape(conv2_split8)[-1]])





	lstm_layer_1 = tf.reshape(lstm_layer_1, [-1, 161*384])

	weights_1 = tf.Variable(tf.random_normal([161*384, 128]), dtype=tf.float32)
	weights_2 = tf.Variable(tf.random_normal([128, NUM_CLASSES]), dtype=tf.float32)
	biases_1 = tf.Variable(tf.random_normal([128]), dtype=tf.float32)
	biases_2 = tf.Variable(tf.random_normal([NUM_CLASSES]), dtype=tf.float32)

	fully_connected_1 = tf.matmul(lstm_layer_1, weights_1) + biases_1
	fully_connected_2 = tf.matmul(fully_connected_1, weights_2) + biases_2

	return fully_connected_2

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

		# testing
		y_predicted_labels = []
		audio_files_list = []
		dataset_test_features = []

		for audio_file in audio_filenames:
			audio_files_list.append(audio_file)
			dataset_test_features.append(get_audio_test_dataset_features_labels(DATASET_PATH, audio_file))

			if len(audio_files_list) > 10000:
				audio_files_list = []
				dataset_test_features = np.array(dataset_test_features)
				dataset_test_features = normalize_test_dataset(dataset_test_features, min_value, max_value)
				y_predicted_labels.append(sess.run(tf.argmax(y_predicted, 1), feed_dict={x: dataset_test_features}))

		# testing end

		# writing predicted labels into a csv file
		y_predicted_labels = np.array(y_predicted_labels)
		with open('run'+str(epoch)+'.csv','w') as file:	
			file.write('fname,label')
			file.write('\n')

			for i in range(0, y_predicted_labels.shape[0]):
				file.write(str(audio_filenames[i]) + ',' + str(ALLOWED_LABELS_MAP[str(int(y_predicted_labels[i]))]))
				file.write('\n')

		


