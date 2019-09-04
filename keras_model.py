import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras


print(tf.__version__)


training_data = np.random.rand(1000, 90)

train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)
keras.backend.set_session(train_sess)

with train_graph.as_default():

	keras.backend.set_learning_phase(1)

	train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_data))
	train_dataset = train_dataset.shuffle(1000).batch(1)

	### MODEL
	x = keras.layers.Input(shape=(90,))
	h = keras.layers.Dense(40, activation=tf.nn.relu)(x)
	z_mu = keras.layers.Dense(10)(h)
	z_sigma = keras.layers.Dense(10, activation=tf.nn.sigmoid)(h)

	eps = tf.random.normal(shape=(10,), mean=0, stddev=1, dtype=tf.float32)
	z = z_mu + eps * z_sigma

	h_decoded = keras.layers.Dense(40, activation=tf.nn.relu)(z)
	x_decoded = keras.layers.Dense(90)(h_decoded)

	model = keras.models.Model(x, x_decoded)

	### LOSS
	recon_err = tf.reduce_sum(tf.abs(x - x_decoded), axis=1)
	kl_div = -.5 * tf.reduce_sum(1 + 2 * tf.math.log(z_sigma) - tf.square(z_mu) - tf.square(z_sigma), axis=1)
	total_loss = tf.reduce_mean(recon_err + kl_div)
	model.add_loss(total_loss)

	### TRAINING
	model.compile(optimizer='adam')
	print(model.summary())
	model.fit(train_dataset, epochs=10)

	### SAVE
	keras_file = 'vae_test.h5'
	keras.models.save_model(model, keras_file)

