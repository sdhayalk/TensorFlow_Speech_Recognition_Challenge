import numpy as np
import os
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile
import statistics


def get_silence_average(path):
	path = path + os.sep + 'silence' + os.sep + '3e7124ba_nohash_0.wav'
	samplerate, test_sound  = wavfile.read(path)
	print(test_sound)


# referred from: https://www.kaggle.com/davids1992/data-visualization-and-investigation
def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)

    return freqs, np.log(spec.T.astype(np.float32) + eps)


def get_audio_dataset_features_labels(path, type='train'):
	'''TODO: for type='test' and 'both'
	
	Arguments:
		path {[type]} -- [description]
	
	Keyword Arguments:
		type {str} -- [description] (default: {'train'})
	
	Returns:
		[type] -- [description]
	'''

	TYPES = ['train', 'test', 'both']
	if type not in TYPES:
		print("Argument type should be one of 'train', 'test', 'both'")
		return 

	TRAIN_PATH = path + os.sep + 'train' + os.sep + 'audio'
	TEST_PATH = path + os.sep + 'test'
	ALLOWED_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
	SILENCE_AVERAGE = get_silence_average(TRAIN_PATH)
	dataset_features = []
	dataset_labels = []

	one_hot_map = {}
	label_index = 0
	for allowed_label in ALLOWED_LABELS:
		one_hot_map[allowed_label] = label_index
		label_index += 1


	if type == 'train':
		folders_list = os.listdir(TRAIN_PATH)
		folders_list.remove('_background_noise_')

		for folder in folders_list:
			print("In folder", folder)
			audio_files_list = os.listdir(TRAIN_PATH + os.sep + folder)

			for audio_file in audio_files_list:
				audio_file_path = TRAIN_PATH + os.sep + folder + os.sep + audio_file
				samplerate, test_sound  = wavfile.read(audio_file_path)
				
				if len(test_sound) < 16000:
					diff = 16000 - len(test_sound)
					while(diff > 0):
						test_sound = np.insert(test_sound, 1, 0)
						diff -= 1
				# print(len(test_sound))

				_, spectrogram = log_specgram(test_sound, samplerate)

				dataset_features.append(spectrogram.T)
				if folder in ALLOWED_LABELS:
					label_index = one_hot_map[folder]
					label = np.zeros(len(ALLOWED_LABELS))
					label[label_index] = 1
					dataset_labels.append(label)
				else:
					label_index = one_hot_map['unknown']
					label = np.zeros(len(ALLOWED_LABELS))
					label[label_index] = 1
					dataset_labels.append(label)

				break
	return np.array(dataset_features), np.array(dataset_labels)


# TRAIN_PATH = 'G:/DL/tf_speech_recognition'
# dataset_train_features, dataset_train_labels = get_audio_dataset_features_labels(TRAIN_PATH, type='train')
# print(dataset_train_features.shape, dataset_train_labels.shape)
