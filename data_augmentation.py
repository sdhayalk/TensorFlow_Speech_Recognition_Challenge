''' 
references from:
	https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_shift
	https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_zoom
'''

import tensorflow as tf
import numpy as np

def augment_data(dataset, dataset_labels, augmentation_factor=1, use_random_shift=True, use_random_zoom=True):
	augmented_image = []
	augmented_image_labels = []

	for num in range (0, dataset.shape[0]):

		for i in range(0, augementation_factor):
			# original image:
			augmented_image.append(dataset[num])
			augmented_image_labels.append(dataset_labels[num])

			if use_random_shift:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_zoom:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_zoom(dataset[num], [], row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])


	return np.array(augmented_image), np.array(augmented_image_labels)
