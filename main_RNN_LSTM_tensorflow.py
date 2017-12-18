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
		return dataset[i*BATCH_SIZE:, :], dataset.shape[0] - i*BATCH_SIZE
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :], BATCH_SIZE


#DATASET_PATH = 'G:/DL/tf_speech_recognition'
DATASET_PATH = '/home/paperspace/tf_speech_recognition'
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
current_batch_size = tf.placeholder(tf.int32)


def recurrent_neural_network(x, current_batch_size):
	lstm_cell_1 = rnn.LSTMCell(128, state_is_tuple=True)
	lstm_cell_2 = rnn.LSTMCell(256, state_is_tuple=True)
	lstm_cell_3 = rnn.LSTMCell(384, state_is_tuple=True)
	multi_lstm_cells = rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2, lstm_cell_3], state_is_tuple=True)

	lstm_layer_1, lstm_layer_1_states = tf.nn.dynamic_rnn(multi_lstm_cells, x, dtype=tf.float32)
	lstm_layer_1 = tf.reshape(lstm_layer_1, [-1, 161*384])

	weights_1 = tf.Variable(tf.random_normal([161*384, 128]), dtype=tf.float32)
	weights_2 = tf.Variable(tf.random_normal([128, NUM_CLASSES]), dtype=tf.float32)
	biases_1 = tf.Variable(tf.random_normal([128]), dtype=tf.float32)
	biases_2 = tf.Variable(tf.random_normal([NUM_CLASSES]), dtype=tf.float32)

	fully_connected_1 = tf.matmul(lstm_layer_1, weights_1) + biases_1
	fully_connected_2 = tf.matmul(fully_connected_1, weights_2) + biases_2

	return fully_connected_2
	

logits = recurrent_neural_network(x, current_batch_size)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer()
training = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())		# initialize all global variables, which includes weights and biases

	# training start
	for epoch in range(0, NUM_EPOCHS):
		total_cost = 0

		for i in range(0, int(NUM_EXAMPLES/BATCH_SIZE)):
			batch_x, batch_current_batch_size = get_batch(dataset_train_features, i, BATCH_SIZE)	# get batch of features of size BATCH_SIZE
			batch_y, _ = get_batch(dataset_train_labels, i, BATCH_SIZE)	# get batch of labels of size BATCH_SIZE

			_, batch_cost = sess.run([training, loss], feed_dict={x: batch_x, y: batch_y, current_batch_size: batch_current_batch_size})	# train on the given batch size of features and labels
			total_cost += batch_cost

		print("Epoch:", epoch, "\tCost:", total_cost)

		# predict validation accuracy after every epoch
		sum_accuracy_validation = 0.0
		sum_i = 0
		for i in range(0, int(dataset_validation_features.shape[0]/BATCH_SIZE)):
			batch_x, batch_current_batch_size = get_batch(dataset_validation_features, i, BATCH_SIZE)
			batch_y, _ = get_batch(dataset_validation_labels, i, BATCH_SIZE)
			# print(batch_current_batch_size)

			y_predicted = tf.nn.softmax(logits)
			correct = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y, 1))
			accuracy_function = tf.reduce_mean(tf.cast(correct, 'float'))
			accuracy_validation = accuracy_function.eval({x:batch_x, y:batch_y, current_batch_size:batch_current_batch_size})

			sum_accuracy_validation += accuracy_validation
			sum_i += 1
			print("Validation Accuracy in Epoch ", epoch, ":", accuracy_validation, 'sum_i:', sum_i, 'sum_accuracy_validation:', sum_accuracy_validation)
		# training end

		# testing
		if epoch > 0 and epoch%2 == 0:
			y_predicted_labels = []
			audio_files_list = []
			dataset_test_features = []
			test_samples_picked = 0
			y_predicted = tf.nn.softmax(logits)

			for audio_file in audio_filenames:
				audio_files_list.append(audio_file)
				dataset_test_features.append(get_audio_test_dataset_features_labels(DATASET_PATH, audio_file))

				if len(audio_files_list) == 3200:
					dataset_test_features = np.array(dataset_test_features)
					dataset_test_features = normalize_test_dataset(dataset_test_features, min_value, max_value)

					for i in range(0, int(dataset_test_features.shape[0]/BATCH_SIZE)):
						batch_x, batch_current_batch_size = get_batch(dataset_test_features, i, BATCH_SIZE)
						temp = sess.run(tf.argmax(y_predicted, 1), feed_dict={x: batch_x, current_batch_size: batch_current_batch_size})
						for element in temp:
							y_predicted_labels.append(element) 

					test_samples_picked += 3200
					print('test_samples_picked:', test_samples_picked)

					# writing predicted labels into a csv file
					with open('run'+str(epoch)+'.csv','a') as file:	
						for i in range(0, len(y_predicted_labels)):
							file.write(str(audio_files_list[i]) + ',' + str(ALLOWED_LABELS_MAP[str(int(y_predicted_labels[i]))]))
							file.write('\n')

					y_predicted_labels = []
					dataset_test_features = []
					audio_files_list = []

			# last set
			dataset_test_features = np.array(dataset_test_features)
			dataset_test_features = normalize_test_dataset(dataset_test_features, min_value, max_value)

			for i in range(0, int(dataset_test_features.shape[0]/BATCH_SIZE)):
				batch_x, batch_current_batch_size = get_batch(dataset_test_features, i, BATCH_SIZE)
				temp = sess.run(tf.argmax(y_predicted, 1), feed_dict={x: batch_x, current_batch_size: batch_current_batch_size})
				for element in temp:
					y_predicted_labels.append(element) 

			test_samples_picked += 3200
			print('test_samples_picked:', test_samples_picked)

			# writing predicted labels into a csv file
			with open('run'+str(epoch)+'.csv','a') as file:	
				for i in range(0, len(y_predicted_labels)):
					file.write(str(audio_files_list[i]) + ',' + str(ALLOWED_LABELS_MAP[str(int(y_predicted_labels[i]))]))
					file.write('\n')
