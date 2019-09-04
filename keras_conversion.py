import tensorflow as tf
import numpy as np

### CONVERSION

keras_file = 'vae_test.h5'
dataset = np.random.rand(100, 90).astype(np.float32)

def representative_dataset_gen():
	for i in range(100):
		yield [dataset[i:i+1]]

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
converter.representative_dataset = representative_dataset_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_file_name = 'vae_keras.tflite'
tflite_model = converter.convert()
open(tflite_file_name, 'wb').write(tflite_model)