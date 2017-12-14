import numpy as np
from data_preprocessing import get_audio_dataset_features_labels, get_audio_test_dataset_filenames, get_audio_test_dataset_features_labels, normalize_training_dataset, normalize_test_dataset

DATASET_PATH = 'G:/DL/tf_speech_recognition'
audio_filenames = get_audio_test_dataset_filenames(DATASET_PATH)
audio_files_list = []

with open('test_files.csv','w') as file:	
	for audio_file in audio_filenames:
		file.write(audio_file)
		file.write('\n')

