import tensorflow as tf
import numpy as np
import random

from data_preprocessing import get_audio_dataset_features_labels, get_audio_test_dataset_filenames, get_audio_test_dataset_features_labels, normalize_training_dataset, normalize_test_dataset
from data_augmentation import augment_data
# from keras.layers import merge

# floyd run --gpu --env tensorflow --data sahilcrslab/datasets/tf_speech_recog/3 'python main_CNN_Resnet_tensorflow.py'

def shuffle_randomize(dataset_features, dataset_labels):
	dataset_combined = list(zip(dataset_features, dataset_labels))
	random.shuffle(dataset_combined)
	dataset_features[:], dataset_labels[:] = zip(*dataset_combined)
	return dataset_features, dataset_labels

def get_batch(dataset, i, BATCH_SIZE):
	if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
		return dataset[i*BATCH_SIZE:, :]
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]


#DATASET_PATH = 'G:/DL/tf_speech_recognition/data'
#DATASET_PATH = '/home/paperspace/tf_speech_recognition/data'
DATASET_PATH = '/input'
ALLOWED_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
ALLOWED_LABELS_MAP = {}
for i in range(0, len(ALLOWED_LABELS)):
	ALLOWED_LABELS_MAP[str(i)] = ALLOWED_LABELS[i]

dataset_train_features, dataset_train_labels, labels_one_hot_map = get_audio_dataset_features_labels(DATASET_PATH, ALLOWED_LABELS, type='train')
audio_filenames = get_audio_test_dataset_filenames(DATASET_PATH)
dataset_train_features = dataset_train_features.reshape((dataset_train_features.shape[0], dataset_train_features.shape[1], dataset_train_features.shape[2], 1))

print('dataset_train_features.shape:', dataset_train_features.shape, 'dataset_train_labels.shape:', dataset_train_labels.shape)

# normalize training and testing features dataset
print('Normalizing datasets')
dataset_train_features, min_value, max_value = normalize_training_dataset(dataset_train_features)

# randomize shuffle
print('Shuffling training dataset')
dataset_train_features, dataset_train_labels = shuffle_randomize(dataset_train_features, dataset_train_labels)

# divide training set into training and validation
partition_num = 57000
dataset_validation_features, dataset_validation_labels = dataset_train_features[partition_num:dataset_train_features.shape[0], :], dataset_train_labels[partition_num:dataset_train_labels.shape[0], :]
dataset_train_features, dataset_train_labels = dataset_train_features[0:partition_num, :], dataset_train_labels[0:partition_num, :]
print('dataset_validation_features.shape:', dataset_validation_features.shape, 'dataset_validation_labels.shape:', dataset_validation_labels.shape)

CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
NUM_CLASSES = len(CLASSES)
NUM_EXAMPLES = dataset_train_features.shape[0]
NUM_CHUNKS = dataset_train_features.shape[1]	# 161
CHUNK_SIZE = dataset_train_features.shape[2]	# 99 
NUM_EPOCHS = 100
BATCH_SIZE = 64

x = tf.placeholder(tf.float32, shape=[None, NUM_CHUNKS, CHUNK_SIZE, 1])
y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])


def residual_block(x, num_input_filters, num_output_filters, block_num):

	# defining the weights and biases for this residual block
	w_conv_1 = tf.get_variable('rs_block_'+str(block_num)+'_w_conv_1', shape=[3,3,num_input_filters, num_output_filters], dtype=tf.float32)
	w_conv_2 = tf.get_variable('rs_block_'+str(block_num)+'_w_conv_2', shape=[3,3,num_output_filters, num_output_filters], dtype=tf.float32)
	b_conv_1 = tf.get_variable('rs_block_'+str(block_num)+'_b_conv_1', shape=[num_output_filters], dtype=tf.float32)
	b_conv_2 = tf.get_variable('rs_block_'+str(block_num)+'_b_conv_2', shape=[num_output_filters], dtype=tf.float32)
	
	# implementing residual block logic
	input_1 = tf.contrib.layers.batch_norm(x)
	input_1 = tf.nn.relu(input_1)
	weight_layer_1 = tf.nn.conv2d(input_1, w_conv_1, strides=[1,1,1,1], padding='SAME') + b_conv_1
	intermediate = tf.contrib.layers.batch_norm(weight_layer_1)
	intermediate = tf.nn.relu(intermediate)
	weight_layer_2 = tf.nn.conv2d(intermediate, w_conv_2, strides=[1,1,1,1], padding='SAME') + b_conv_2

	# elementwise addition of x and weight_layer_2
	if num_input_filters != num_output_filters:
		w_conv_increase = tf.get_variable('rs_block_'+str(block_num)+'_w_conv_increase', shape=[1,1,num_input_filters, num_output_filters], dtype=tf.float32)
		b_conv_increase = tf.get_variable('rs_block_'+str(block_num)+'_b_conv_increase', shape=[num_output_filters], dtype=tf.float32)
		x = tf.nn.conv2d(x, w_conv_increase, strides=[1,1,1,1], padding='SAME') + b_conv_increase
	output = tf.add(x, weight_layer_2)

	return output


def recurrent_neural_network(x):

	rs_block_1 = residual_block(x, 1, 32, 1)
	rs_block_2 = residual_block(rs_block_1, 32, 32, 2)
	# rs_block_3 = residual_block(rs_block_2, 32, 32, 3)
	rs_block_3 = tf.nn.max_pool(rs_block_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	rs_block_4 = residual_block(rs_block_3, 32, 64, 4)
	rs_block_5 = residual_block(rs_block_4, 64, 64, 5)
	# rs_block_6 = residual_block(rs_block_5, 64, 64, 6)
	rs_block_6 = tf.nn.max_pool(rs_block_5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	rs_block_7 = residual_block(rs_block_6, 64, 128, 7)
	rs_block_8 = residual_block(rs_block_7, 128, 128, 8)
	# rs_block_9 = residual_block(rs_block_8, 128, 128, 9)
	rs_block_9 = tf.nn.max_pool(rs_block_8, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	rs_block_10 = residual_block(rs_block_9, 128, 256, 10)
	rs_block_11 = residual_block(rs_block_10, 256, 256, 11)
	# rs_block_12 = residual_block(rs_block_11, 256, 256, 12)
	rs_block_12 = tf.nn.max_pool(rs_block_11, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	num_features = 19712
	flattened = tf.reshape(rs_block_12, [-1, num_features])

	# fully connected layers
	w_fc1 = tf.get_variable('w_fc1', shape=[num_features,128], dtype=tf.float32)
	w_fc2 = tf.get_variable('w_fc2', shape=[128, NUM_CLASSES], dtype=tf.float32)
	b_fc1 = tf.get_variable('b_fc1', shape=[128], dtype=tf.float32)
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

			batch_x, batch_y = augment_data(batch_x, batch_y, augmentation_factor=1)	# augment the data

			_, batch_cost = sess.run([training, loss], feed_dict={x: batch_x, y: batch_y})	# train on the given batch size of features and labels
			total_cost += batch_cost
			if i % 25 == 0:
				print(i)

		print("Epoch:", epoch, "\tCost:", total_cost)

		# predict validation accuracy after every epoch
		sum_accuracy_validation = 0.0
		sum_i = 0
		for i in range(0, int(dataset_validation_features.shape[0]/BATCH_SIZE)):
			batch_x = get_batch(dataset_validation_features, i, BATCH_SIZE)
			batch_y = get_batch(dataset_validation_labels, i, BATCH_SIZE)

			y_predicted = tf.nn.softmax(logits)
			correct = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y, 1))
			accuracy_function = tf.reduce_mean(tf.cast(correct, 'float'))
			accuracy_validation = accuracy_function.eval({x:batch_x, y:batch_y})

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
					dataset_test_features = dataset_test_features.reshape((dataset_test_features.shape[0], dataset_test_features,shape[1], dataset_test_features.shape[2], 1))
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
			dataset_test_features = dataset_test_features.reshape((dataset_test_features.shape[0], dataset_test_features,shape[1], dataset_test_features.shape[2], 1))
			dataset_test_features = normalize_test_dataset(dataset_test_features, min_value, max_value)

			for i in range(0, int(dataset_test_features.shape[0]/BATCH_SIZE)):
				batch_x = get_batch(dataset_test_features, i, BATCH_SIZE)
				temp = sess.run(tf.argmax(y_predicted, 1), feed_dict={x: batch_x})
				for element in temp:
					y_predicted_labels.append(element) 

			test_samples_picked += 6400
			print('test_samples_picked:', test_samples_picked)

			# writing predicted labels into a csv file
			# y_predicted_labels = np.array(y_predicted_labels)
			with open('run'+str(epoch)+'.csv','a') as file:	
				for i in range(0, len(y_predicted_labels)):
					file.write(str(audio_files_list[i]) + ',' + str(ALLOWED_LABELS_MAP[str(int(y_predicted_labels[i]))]))
					file.write('\n')


		
