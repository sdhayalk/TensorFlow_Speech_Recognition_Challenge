import numpy as np
import os
import matplotlib.pyplot as plt

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


def get_audio(type='train', path):
	TRAIN_PATH = path + os.sep + 'train'
	TEST_PATH = path + os.sep + 'test'
	dataset_features = []
	dataset_labels = []

	if type == 'train':
		print(os.listdir())




# samplerate, test_sound  = wavfile.read(filepath)
# _, spectrogram = log_specgram(test_sound, samplerate)
# plt.imshow(spectrogram.T, aspect='auto', origin='lower')