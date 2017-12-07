import numpy as np
import os
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

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


def get_audio(path, type='train'):
	TYPES = ['train', 'test', 'both']
	ALLOWED_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
	if type not in TYPES:
		print("Argument type should be one of 'train', 'test', 'both'")
		return 

	TRAIN_PATH = path + os.sep + 'train' + os.sep + 'audio'
	TEST_PATH = path + os.sep + 'test'
	dataset_features = []
	dataset_labels = []

	if type == 'train':
		folders_list = os.listdir(TRAIN_PATH)
		folders_list.remove('_background_noise_')

		for folder in folders_list:
			audio_files_list = os.listdir(TRAIN_PATH + os.sep + folder)

			for audio_file in audio_files_list:
				audio_file_path = TRAIN_PATH + os.sep + folder + os.sep + audio_file
				samplerate, test_sound  = wavfile.read(audio_file_path)
				_, spectrogram = log_specgram(test_sound, samplerate)
				# print(spectrogram.shape)
				# plt.imshow(spectrogram, aspect='auto', origin='lower')
				# plt.show()






# samplerate, test_sound  = wavfile.read(filepath)
# _, spectrogram = log_specgram(test_sound, samplerate)
# plt.imshow(spectrogram.T, aspect='auto', origin='lower')

TRAIN_PATH = 'G:/DL/tf_speech_recognition'
get_audio(TRAIN_PATH, type='train')