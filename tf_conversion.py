import tensorflow as tf
import numpy as np

### CONVERSION

graph_def_file = 'frozen_model.pb'
inputs = ['model/eval_inputs']
outputs = ['model/y']
dataset = np.random.rand(100, 90).astype(np.float32)

def representative_dataset_gen():
	for i in range(100):
		yield [dataset[i:i+1]]

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, inputs, outputs)
converter.representative_dataset = representative_dataset_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_file_name = 'vae_classic.tflite'
tflite_model = converter.convert()
open(tflite_file_name, 'wb').write(tflite_model)
