import numpy as np
import pandas as pd
import tensorflow as tf


print(tf.__version__)


### TRAINING

training_data = np.random.rand(1000, 90).astype(np.float32).reshape(-1, 90)
train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_data))
train_dataset = train_dataset.shuffle(1000)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_dataset_init_op = iterator.make_initializer(train_dataset)
inputs = iterator.get_next()

dense = tf.compat.v1.layers.dense

with tf.name_scope('model'):

	x = tf.identity(inputs, name='x')
	print(x)
	h = dense(x, 40, activation=tf.nn.relu)
	z_mu = dense(h, 10, activation=None)
	z_sigma = dense(h, 10, activation=tf.sigmoid)

	eps = tf.random.normal(shape=tf.shape(z_sigma), mean=0, stddev=1, dtype=tf.float32)
	z = z_mu + eps * z_sigma

	h_decoded = dense(z, 40, activation=tf.nn.relu)
	x_decoded = dense(h_decoded, 90, activation=None)
	y = tf.identity(x_decoded, name='y')

with tf.name_scope('loss'):
	recon_err = tf.reduce_sum(tf.abs(x - x_decoded), axis=1)
	kl_div = -.5 * tf.reduce_sum(1+2*tf.math.log(z_sigma)-tf.square(z_mu)-tf.square(z_sigma), axis=1)
	total_loss = tf.reduce_mean(recon_err + kl_div)

with tf.name_scope('optimizer'):
	train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(total_loss)

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	for epoch in range(10):

		sess.run(train_dataset_init_op)

		for _ in range(1000):
			_, loss = sess.run([train_op, total_loss])
		print(f'\n[Epoch {epoch}]\ntraining loss: {loss}')

	print('Training Done!')

	saver = tf.train.Saver()
	saver.save(sess, 'vae_test.ckpt')


### FREEZING THE GRAPH

eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

with eval_graph.as_default():

	dense = tf.compat.v1.layers.dense

	with tf.name_scope('model'):

		eval_inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 90], name='eval_inputs')
		
		x = tf.identity(eval_inputs, name='x')

		h = dense(x, 40, activation=tf.nn.relu)
		z_mu = dense(h, 10, activation=None)
		z_sigma = dense(h, 10, activation=tf.sigmoid)

		eps = tf.random.normal(shape=tf.shape(z_sigma), mean=0, stddev=1, dtype=tf.float32)
		z = z_mu + eps * z_sigma

		h_decoded = dense(z, 40, activation=tf.nn.relu)
		x_decoded = dense(h_decoded, 90, activation=None)

		y = tf.identity(x_decoded, name='y')

	eval_graph_def = eval_graph.as_graph_def()
	latest_checkpoint = tf.train.latest_checkpoint('.')
	saver = tf.compat.v1.train.Saver()
	saver.restore(eval_sess, latest_checkpoint)

	frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
				eval_sess,
				eval_graph_def,
				['model/y']
	)

	with open('frozen_model.pb', 'wb') as f:
		f.write(frozen_graph_def.SerializeToString())