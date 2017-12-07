import tensorflow as tf
import numpy as np

from data_preprocessing import get_audio_dataset_features_labels

DATASET_PATH = 'G:/DL/tf_speech_recognition'


dataset_train_features, dataset_train_labels = get_audio_dataset_features_labels(DATASET_PATH, type='train')
# dataset_test_features, dataset_test_labels = get_audio_dataset_features_labels(DATASET_PATH, type='test')

print('dataset_train_features.shape:', dataset_train_features.shape, 'dataset_train_labels.shape:', dataset_train_labels.shape)
