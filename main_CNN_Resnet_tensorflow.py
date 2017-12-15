import tensorflow as tf
import numpy as np
import random

from data_preprocessing import get_audio_dataset_features_labels, get_audio_test_dataset_filenames, get_audio_test_dataset_features_labels, normalize_training_dataset, normalize_test_dataset
from data_augmentation import augment_data
# from keras.layers import merge

def shuffle_randomize(dataset_features, dataset_labels):
	dataset_combined = list(zip(dataset_features, dataset_labels))
	random.shuffle(dataset_combined)
	dataset_features[:], dataset_labels[:] = zip(*dataset_combined)
	return dataset_features, dataset_labels

def get_batch(dataset, i, BATCH_SIZE):
	if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
		return dataset[i*BATCH_SIZE:, :]
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]


#DATASET_PATH = 'G:/DL/tf_speech_recognition'
DATASET_PATH = '/home/paperspace/tf_speech_recognition'
ALLOWED_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
ALLOWED_LABELS_MAP = {}
for i in range(0, len(ALLOWED_LABELS)):
	ALLOWED_LABELS_MAP[str(i)] = ALLOWED_LABELS[i]

dataset_train_features, dataset_train_labels, labels_one_hot_map = get_audio_dataset_features_labels(DATASET_PATH, ALLOWED_LABELS, type='train')
audio_filenames = get_audio_test_dataset_filenames(DATASET_PATH)
dataset_train_features = dataset_train_features.reshape((dataset_train_features[0], dataset_train_features[1], dataset_train_features[2], 1))

print('dataset_train_features.shape:', dataset_train_features.shape, 'dataset_train_labels.shape:', dataset_train_labels.shape)

# normalize training and testing features dataset
print('Normalizing datasets')
dataset_train_features, min_value, max_value = normalize_training_dataset(dataset_train_features)

# randomize shuffle
print('Shuffling training dataset')
dataset_train_features, dataset_train_labels = shuffle_randomize(dataset_train_features, dataset_train_labels)

# divide training set into training and validation
# dataset_validation_features, dataset_validation_labels = dataset_train_features[57000:dataset_train_features.shape[0], :], dataset_train_labels[57000:dataset_train_labels.shape[0], :]
# dataset_train_features, dataset_train_labels = dataset_train_features[0:57000, :], dataset_train_labels[0:57000, :]
# print('dataset_validation_features.shape:', dataset_validation_features.shape, 'dataset_validation_labels.shape:', dataset_validation_labels.shape)

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
	'w_conv0': tf.get_variable('w_conv0', shape=[3,3,1,32], dtype=tf.float32),
	'w_conv1': tf.get_variable('w_conv1', shape=[3,3,32,64], dtype=tf.float32),
	'w_conv2': tf.get_variable('w_conv2', shape=[3,3,64,96], dtype=tf.float32),
	'w_conv3': tf.get_variable('w_conv3', shape=[3,3,96,128], dtype=tf.float32),
	'w_conv4': tf.get_variable('w_conv4', shape=[3,3,128,192], dtype=tf.float32),
	'w_conv5': tf.get_variable('w_conv5', shape=[3,3,192,256], dtype=tf.float32)
}
biases = {
	'b_conv0': tf.get_variable('b_conv0', shape=[32], dtype=tf.float32),
	'b_conv1': tf.get_variable('b_conv1', shape=[64], dtype=tf.float32),
	'b_conv2': tf.get_variable('b_conv2', shape=[96], dtype=tf.float32),
	'b_conv3': tf.get_variable('b_conv3', shape=[128], dtype=tf.float32),
	'b_conv4': tf.get_variable('b_conv4', shape=[192], dtype=tf.float32),
	'b_conv5': tf.get_variable('b_conv5', shape=[256], dtype=tf.float32)
}

def leakyrelu(x):
	return tf.nn.relu(x) - 0.2*tf.nn.relu(-x)


def recurrent_neural_network(x):

	# x = tf.reshape(x, [-1, tf.shape(x)[-2], tf.shape(x)[-1], 1])
	conv0 = tf.nn.conv2d(x, weights['w_conv0'], strides=[1,1,1,1], padding='SAME') + biases['b_conv0']
	conv0 = tf.contrib.layers.batch_norm(conv0)
	conv0 = leakyrelu(conv0)
	conv0 = tf.nn.max_pool(conv0, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	conv1 = tf.nn.conv2d(conv0, weights['w_conv1'], strides=[1,1,1,1], padding='SAME') + biases['b_conv1']
	conv1 = tf.contrib.layers.batch_norm(conv1)
	conv1 = leakyrelu(conv1)
	conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	conv2 = tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1,1,1,1], padding='SAME') + biases['b_conv2']
	conv2 = tf.contrib.layers.batch_norm(conv2)
	conv2 = leakyrelu(conv2)
	conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	conv3 = tf.nn.conv2d(conv2, weights['w_conv3'], strides=[1,1,1,1], padding='SAME') + biases['b_conv3']
	conv3 = tf.contrib.layers.batch_norm(conv3)
	conv3 = leakyrelu(conv3)
	conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	conv4 = tf.nn.conv2d(conv3, weights['w_conv4'], strides=[1,1,1,1], padding='SAME') + biases['b_conv4']
	conv4 = tf.contrib.layers.batch_norm(conv4)
	conv4 = leakyrelu(conv4)
	conv4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	conv5 = tf.nn.conv2d(conv4, weights['w_conv5'], strides=[1,1,1,1], padding='SAME') + biases['b_conv5']
	conv5 = tf.contrib.layers.batch_norm(conv5)
	conv5 = leakyrelu(conv5)
	conv5 = tf.nn.max_pool(conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	num_features = 1536
	flattened = tf.reshape(conv5, [BATCH_SIZE, num_features])

	# fully connected layers
	w_fc1 = tf.get_variable('w_fc1', shape=[num_features,512], dtype=tf.float32)
	w_fc2 = tf.get_variable('w_fc2', shape=[512, NUM_CLASSES], dtype=tf.float32)
	b_fc1 = tf.get_variable('b_fc1', shape=[512], dtype=tf.float32)
	b_fc2 = tf.get_variable('b_fc2', shape=[NUM_CLASSES], dtype=tf.float32)

	fully_connected_1 = tf.matmul(flattened, w_fc1) + b_fc1
	fully_connected_2 = tf.matmul(fully_connected_1, w_fc2) + b_fc2

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

			batch_x, batch_y = augment_data(batch_x, batch_y, augementation_factor=3)	# augment the data

			_, batch_cost = sess.run([training, loss], feed_dict={x: batch_x, y: batch_y})	# train on the given batch size of features and labels
			total_cost += batch_cost

		print("Epoch:", epoch, "\tCost:", total_cost)

		# # predict validation accuracy after every epoch
		# sum_accuracy_validation = 0.0
		# sum_i = 0
		# for i in range(0, int(dataset_validation_features.shape[0]/BATCH_SIZE)):
		# 	batch_x = get_batch(dataset_validation_features, i, BATCH_SIZE)
		# 	batch_y = get_batch(dataset_validation_labels, i, BATCH_SIZE)

		# 	y_predicted = tf.nn.softmax(logits)
		# 	correct = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y, 1))
		# 	accuracy_function = tf.reduce_mean(tf.cast(correct, 'float'))
		# 	accuracy_validation = accuracy_function.eval({x:batch_x, y:batch_y})

		# 	sum_accuracy_validation += accuracy_validation
		# 	sum_i += 1
		# 	print("Validation Accuracy in Epoch ", epoch, ":", accuracy_validation, 'sum_i:', sum_i, 'sum_accuracy_validation:', sum_accuracy_validation)
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
					dataset_test_features = dataset_test_features.reshape((dataset_test_features[0], dataset_test_features[1], dataset_test_features[2], 1))
					dataset_test_features = normalize_test_dataset(dataset_test_features, min_value, max_value)

					for i in range(0, int(dataset_test_features.shape[0]/BATCH_SIZE)):
						batch_x = get_batch(dataset_test_features, i, BATCH_SIZE)
						temp = sess.run(tf.argmax(y_predicted, 1), feed_dict={x: batch_x})
						for element in temp:
							y_predicted_labels.append(element) 

					test_samples_picked += 3200
					print('test_samples_picked:', test_samples_picked)

					# writing predicted labels into a csv file
					# y_predicted_labels = np.array(y_predicted_labels)
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
				batch_x = get_batch(dataset_test_features, i, BATCH_SIZE)
				temp = sess.run(tf.argmax(y_predicted, 1), feed_dict={x: batch_x})
				for element in temp:
					y_predicted_labels.append(element) 

			test_samples_picked += 3200
			print('test_samples_picked:', test_samples_picked)

			# writing predicted labels into a csv file
			# y_predicted_labels = np.array(y_predicted_labels)
			with open('run'+str(epoch)+'.csv','a') as file:	
				for i in range(0, len(y_predicted_labels)):
					file.write(str(audio_files_list[i]) + ',' + str(ALLOWED_LABELS_MAP[str(int(y_predicted_labels[i]))]))
					file.write('\n')


		
