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
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_file_name = 'vae_keras.tflite'
tflite_model = converter.convert()
open(tflite_file_name, 'wb').write(tflite_model)

### Test tflite model

interpreter = tf.lite.Interpreter(model_path=tflite_file_name)
interpreter.allocate_tensors()

input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]
print('Input detail: ', input_detail)
print('Output detail: ', output_detail)

def quantize(real_value):
	std, mean = input_detail['quantization']
	return (real_value / std + mean).astype(np.uint8)

def dequantize(quant_value):
	std, mean = output_detail['quantization']
	return (std * (quant_value - mean)).astype(np.float32)

interpreter.set_tensor(input_detail['index'], quantize(np.random.rand(1, 90).astype(np.float32)))
interpreter.invoke()
pred_litemodel = interpreter.get_tensor(output_detail['index'])

deq_output = dequantize(pred_litemodel)
print(deq_output)