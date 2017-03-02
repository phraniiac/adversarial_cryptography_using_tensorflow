from collections import namedtuple
import tensorflow as tf
import numpy as np

config = namedtuple('config', 'dropout num_epochs ')

class config:
	"""Config for outr Neural Network"""
	def __init__(self, **config_arg):
		self.dropout = 0.4
		self.lr = 0.01
		self.batch_size = 64
		self.plain_text_length = 128
		self.cipher_text_length = 128
		self.key_length = self.plain_text_length
		self.num_epochs = 10
		self.num_batches = 1000

		# update the manual provided args.
		self.__dict__.update(config_arg)


class layers_and_weights():
	def __init__(self):
		pass

	def FC_layer(self, x, W, b, op_type='relu'):
		dense = tf.reshape(x, [-1, W.get_shape().as_list()[0]])
		return tf.nn.relu(tf.add(tf.matmul(dense, W), b))

	def conv1d_layer(self, x, W, b, strides, padding='SAME'):
		conv_out = tf.nn.conv1d(x, W, strides, padding)
		return tf.add(conv_out, b)
	
	def get_weights(self, name, shape, initializer='xavier', constant=0.0):
		if initializer == 'xavier':
			return tf.get_variable(name=name, shape=shape, \
					initializer=tf.contrib.layers.xavier_initializer())
		if initializer == 'constant':
			return tf.get_variable(name=name, shape=shape, \
					initializer=tf.constant_initializer(constant))		


class NNModel():
	def __init__(self, config_object):
		self.model_name = 'crypto'
		self.conf = config_object
		
		self.lw = layers_and_weights()
		
		# add components to graphs.
		self.add_placeholders()
		self.add_weights()
		self.add_training_op()
		self.add_loss_op()
		self.add_optimizer_op()
		
		self.least_loss = float("inf")
		return
	
	def variable_summaries(self, var):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
				tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)


	# add placeholders
	def add_placeholders(self):
		self.input_message_placeholder = tf.placeholder(dtype=tf.float32, \
					shape=[self.conf.batch_size, self.conf.plain_text_length])
		self.input_key_placeholder = tf.placeholder(dtype=tf.float32, \
					shape=[self.conf.batch_size, self.conf.plain_text_length])

	# initialize variables
	def add_weights(self):
		# alice weights.
		with tf.name_scope('alice_fc_layer'):
			self.alice_weight_fc = self.lw.get_weights('alice_weight_fc', [self.conf.plain_text_length * 2,\
									 self.conf.plain_text_length * 2], 'xavier')
			self.alice_bias_fc = self.lw.get_weights('alice_bias_fc', [self.conf.plain_text_length * 2], \
									initializer='constant', constant=0.1)
			self.variable_summaries(self.alice_weight_fc)
			self.variable_summaries(self.alice_bias_fc)

		with tf.name_scope('alice_conv1_layer'):
			self.alice_weight_convl1 = self.lw.get_weights('alice_weight_convl1', [4, 1, 2], 'xavier')
			self.alice_bias_conv1 = self.lw.get_weights('alice_bias_conv1', [self.conf.plain_text_length * 2, 2], 'constant', 0.1)
			self.variable_summaries(self.alice_weight_convl1)
			self.variable_summaries(self.alice_bias_conv1)

		with tf.name_scope('alice_conv2_layer'):
			self.alice_weight_convl2 = self.lw.get_weights('alice_weight_convl2', [2, 2, 4], 'xavier')
			self.alice_bias_conv2 = self.lw.get_weights('alice_bias_conv2', [self.conf.plain_text_length, 4], 'constant', 0.1)
			self.variable_summaries(self.alice_weight_convl2)
			self.variable_summaries(self.alice_bias_conv2)

		with tf.name_scope('alice_conv3_layer'):
			self.alice_weight_convl3 = self.lw.get_weights('alice_weight_convl3', [1, 4, 4], 'xavier')
			self.alice_bias_conv3 = self.lw.get_weights('alice_bias_conv3', [self.conf.plain_text_length, 4], 'constant', 0.1)
			self.variable_summaries(self.alice_weight_convl3)
			self.variable_summaries(self.alice_bias_conv3)

		with tf.name_scope('alice_conv4_layer'):
			self.alice_weight_convl4 = self.lw.get_weights('alice_weight_convl4', [1, 4, 1], 'xavier')
			self.alice_bias_conv4 = self.lw.get_weights('alice_bias_conv4', [self.conf.plain_text_length, 1], 'constant', 0.1)
			self.variable_summaries(self.alice_weight_convl4)
			self.variable_summaries(self.alice_bias_conv4)


		with tf.name_scope('bob_fc_layer'):
			self.bob_weight_fc = self.lw.get_weights('bob_weight_fc', [self.conf.plain_text_length * 2,\
									 self.conf.plain_text_length * 2], 'xavier')
			self.bob_bias_fc = self.lw.get_weights('bob_bias_fc', [self.conf.plain_text_length * 2],\
									initializer='constant', constant=0.1)
			self.variable_summaries(self.bob_weight_fc)
			self.variable_summaries(self.bob_bias_fc)

		with tf.name_scope('bob_conv1_layer'):
			self.bob_weight_convl1 = self.lw.get_weights('bob_weight_convl1', [4, 1, 2], 'xavier')
			self.bob_bias_conv1 = self.lw.get_weights('bob_bias_conv1', [self.conf.plain_text_length * 2, 2], 'constant', 0.1)
			self.variable_summaries(self.bob_weight_convl1)
			self.variable_summaries(self.bob_bias_conv1)

		with tf.name_scope('bob_conv2_layer'):
			self.bob_weight_convl2 = self.lw.get_weights('bob_weight_convl2', [2, 2, 4], 'xavier')
			self.bob_bias_conv2 = self.lw.get_weights('bob_bias_conv2', [self.conf.plain_text_length, 4], 'constant', 0.1)
			self.variable_summaries(self.bob_weight_convl2)
			self.variable_summaries(self.bob_bias_conv2)

		with tf.name_scope('bob_conv3_layer'):
			self.bob_weight_convl3 = self.lw.get_weights('bob_weight_convl3', [1, 4, 4], 'xavier')
			self.bob_bias_conv3 = self.lw.get_weights('bob_bias_conv3', [self.conf.plain_text_length, 4], 'constant', 0.1)
			self.variable_summaries(self.bob_weight_convl3)
			self.variable_summaries(self.bob_bias_conv3)

		with tf.name_scope('bob_conv4_layer'):
			self.bob_weight_convl4 = self.lw.get_weights('bob_weight_convl4', [1, 4, 1], 'xavier')
			self.bob_bias_conv4 = self.lw.get_weights('bob_bias_conv4', [self.conf.plain_text_length, 1], 'constant', 0.1)
			self.variable_summaries(self.bob_weight_convl4)
			self.variable_summaries(self.bob_bias_conv4)

		with tf.name_scope('eve_fc_layer'):
			self.eve_weight_fc = self.lw.get_weights('eve_weight_fc', [self.conf.plain_text_length,\
									 self.conf.plain_text_length * 2], 'xavier')
			self.eve_bias_fc = self.lw.get_weights('eve_bias_fc', [self.conf.plain_text_length * 2],\
									initializer='constant', constant=0.1)
			self.variable_summaries(self.eve_weight_fc)
			self.variable_summaries(self.eve_weight_fc)

		with tf.name_scope('eve_conv1_layer'):
			self.eve_weight_convl1 = self.lw.get_weights('eve_weight_convl1', [4, 1, 2], 'xavier')
			self.eve_bias_conv1 = self.lw.get_weights('eve_bias_conv1', [self.conf.plain_text_length * 2, 2], 'constant', 0.1)
			self.variable_summaries(self.eve_weight_convl1)
			self.variable_summaries(self.eve_bias_conv1)

		with tf.name_scope('eve_conv2_layer'):
			self.eve_weight_convl2 = self.lw.get_weights('eve_weight_convl2', [2, 2, 4], 'xavier')
			self.eve_bias_conv2 = self.lw.get_weights('eve_bias_conv2', [self.conf.plain_text_length, 4], 'constant', 0.1)
			self.variable_summaries(self.eve_weight_convl2)
			self.variable_summaries(self.eve_bias_conv2)

		with tf.name_scope('eve_conv3_layer'):
			self.eve_weight_convl3 = self.lw.get_weights('eve_weight_convl3', [1, 4, 4], 'xavier')
			self.eve_bias_conv3 = self.lw.get_weights('eve_bias_conv3', [self.conf.plain_text_length, 4], 'constant', 0.1)
			self.variable_summaries(self.eve_weight_convl3)
			self.variable_summaries(self.eve_bias_conv3)

		with tf.name_scope('eve_conv4_layer'):
			self.eve_weight_convl4 = self.lw.get_weights('eve_weight_convl4', [1, 4, 1], 'xavier')
			self.eve_bias_conv4 = self.lw.get_weights('eve_bias_conv4', [self.conf.plain_text_length, 1], 'constant', 0.1)
			self.variable_summaries(self.eve_weight_convl4)
			self.variable_summaries(self.eve_bias_conv4)

		return

	# training_op
	def add_training_op(self):
		# lets start our layer trainings :)

		# training alice.
		with tf.name_scope('alice_training'):
			with tf.name_scope('alice_preprocess'):
				self.alice_input = tf.concat([self.input_message_placeholder, self.input_key_placeholder], axis=1)
			with tf.name_scope('alice_fc_layer1'):
				self.alice_fc_out = self.lw.FC_layer(self.alice_input, self.alice_weight_fc, self.alice_bias_fc)
				self.alice_fc_out = tf.nn.sigmoid(self.alice_fc_out)
				self.variable_summaries(self.alice_fc_out)
				self.alice_fc_out = tf.expand_dims(self.alice_fc_out, axis=2)
			with tf.name_scope('alice_conv_l1'):
				self.alice_c1_out = self.lw.conv1d_layer(self.alice_fc_out, self.alice_weight_convl1, self.alice_bias_conv1, strides=1)
				self.alice_c1_out = tf.nn.sigmoid(self.alice_c1_out)
				self.variable_summaries(self.alice_c1_out)
			with tf.name_scope('alice_conv_l2'):
				self.alice_c2_out = self.lw.conv1d_layer(self.alice_c1_out, self.alice_weight_convl2, self.alice_bias_conv2, strides=2)
				self.alice_c2_out = tf.nn.sigmoid(self.alice_c2_out)
				self.variable_summaries(self.alice_c2_out)
			with tf.name_scope('alice_conv_l3'):
				self.alice_c3_out = self.lw.conv1d_layer(self.alice_c2_out, self.alice_weight_convl3, self.alice_bias_conv3, strides=1)
				self.alice_c3_out = tf.nn.sigmoid(self.alice_c3_out)
				self.variable_summaries(self.alice_c3_out)
			with tf.name_scope('alice_conv_l4'):
				self.alice_c4_out = self.lw.conv1d_layer(self.alice_c3_out, self.alice_weight_convl4, self.alice_bias_conv4, strides=1)
				self.alice_c4_out = tf.nn.tanh(self.alice_c4_out)
				self.alice_out = tf.squeeze(self.alice_c4_out)
				self.variable_summaries(self.alice_out)

		with tf.name_scope('bob_training'):
			with tf.name_scope('bob_preprocess'):
				self.bob_input = tf.concat([self.alice_out, self.input_key_placeholder], axis=1)
			with tf.name_scope('bob_fc_layer1'):
				self.bob_fc_out = self.lw.FC_layer(self.bob_input, self.bob_weight_fc, self.bob_bias_fc)
				self.bob_fc_out = tf.nn.sigmoid(self.bob_fc_out)
				self.variable_summaries(self.bob_fc_out)
				self.bob_fc_out = tf.expand_dims(self.bob_fc_out, axis=2)
			with tf.name_scope('bob_conv_l1'):
				self.bob_c1_out = self.lw.conv1d_layer(self.bob_fc_out, self.bob_weight_convl1, self.bob_bias_conv1, strides=1)
				self.bob_c1_out = tf.nn.sigmoid(self.bob_c1_out)
				self.variable_summaries(self.bob_c1_out)
			with tf.name_scope('bob_conv_l2'):
				self.bob_c2_out = self.lw.conv1d_layer(self.bob_c1_out, self.bob_weight_convl2, self.bob_bias_conv2, strides=2)
				self.bob_c2_out = tf.nn.sigmoid(self.bob_c2_out)
				self.variable_summaries(self.bob_c2_out)
			with tf.name_scope('bob_conv_l3'):
				self.bob_c3_out = self.lw.conv1d_layer(self.bob_c2_out, self.bob_weight_convl3, self.bob_bias_conv3, strides=1)
				self.bob_c3_out = tf.nn.sigmoid(self.bob_c3_out)
				self.variable_summaries(self.bob_c3_out)
			with tf.name_scope('bob_conv_l4'):
				self.bob_c4_out = self.lw.conv1d_layer(self.bob_c3_out, self.bob_weight_convl4, self.bob_bias_conv4, strides=1)
				self.bob_c4_out = tf.nn.tanh(self.bob_c4_out)
				self.bob_out = tf.squeeze(self.bob_c4_out)
				self.variable_summaries(self.bob_out)

		with tf.name_scope('eve_training'):
			with tf.name_scope('eve_preprocess'):
				self.eve_input = self.alice_out
			with tf.name_scope('eve_fc_layer1'):
				self.eve_fc_out = self.lw.FC_layer(self.eve_input, self.eve_weight_fc, self.eve_bias_fc)
				self.eve_fc_out = tf.nn.sigmoid(self.eve_fc_out)
				self.variable_summaries(self.eve_fc_out)
				self.eve_fc_out = tf.expand_dims(self.eve_fc_out, axis=2)
			with tf.name_scope('eve_conv_l1'):
				self.eve_c1_out = self.lw.conv1d_layer(self.eve_fc_out, self.eve_weight_convl1, self.eve_bias_conv1, strides=1)
				self.eve_c1_out = tf.nn.sigmoid(self.eve_c1_out)
				self.variable_summaries(self.eve_c1_out)
			with tf.name_scope('eve_conv_l2'):
				self.eve_c2_out = self.lw.conv1d_layer(self.eve_c1_out, self.eve_weight_convl2, self.eve_bias_conv2, strides=2)
				self.eve_c2_out = tf.nn.sigmoid(self.eve_c2_out)
				self.variable_summaries(self.eve_c2_out)
			with tf.name_scope('eve_conv_l3'):
				self.eve_c3_out = self.lw.conv1d_layer(self.eve_c2_out, self.eve_weight_convl3, self.eve_bias_conv3, strides=1)
				self.eve_c3_out = tf.nn.sigmoid(self.eve_c3_out)
				self.variable_summaries(self.eve_c3_out)
			with tf.name_scope('eve_conv_l4'):
				self.eve_c4_out = self.lw.conv1d_layer(self.eve_c3_out, self.eve_weight_convl4, self.eve_bias_conv4, strides=1)
				self.eve_c4_out = tf.nn.tanh(self.eve_c4_out)
				self.eve_out = tf.squeeze(self.eve_c4_out)
				self.variable_summaries(self.eve_out)

	# loss_function
	def add_loss_op(self):
		self.loss_eve = tf.reduce_mean(tf.abs(tf.subtract(self.eve_out, self.input_message_placeholder)))
		with tf.name_scope('loss_eve'):
			self.variable_summaries(self.loss_eve)
		
		self.loss_bob = tf.reduce_mean(tf.abs(tf.subtract(self.bob_out, self.input_message_placeholder)))
		with tf.name_scope('loss_bob'):
			self.variable_summaries(self.loss_bob)

		self.loss_total = tf.subtract(self.loss_bob, self.loss_eve)
		with tf.name_scope('loss_total'):
			self.variable_summaries(self.loss_total)
		return

	# optimizer	
	def add_optimizer_op(self):
		self.optimizer = tf.train.AdamOptimizer(self.conf.lr).minimize(self.loss_total)

	def run_batch(self, session, batch_input_message, batch_input_key, is_summary=False):
		feed_dict = {
			self.input_message_placeholder: batch_input_message,
			self.input_key_placeholder: batch_input_key	
		}

		if not is_summary:
			loss, _ = session.run([self.loss_total, self.optimizer], feed_dict=feed_dict)
		else:
			merge = tf.summary.merge_all()
			return session.run(merge, feed_dict=feed_dict)


def get_batch_sized_data(batch_size, message_length, key_length):
	message_batch = np.random.uniform(-1.0, 1.0, [batch_size, message_length])
	key_batch = np.random.uniform(-1.0, 1.0, [batch_size, key_length])
	return message_batch, key_batch

def test_run():
	conf = config()
	with tf.Graph().as_default():
		model = NNModel(conf)
		init = tf.global_variables_initializer()
		# summary writer.
		writer = tf.summary.FileWriter('./tf_summary/crypto/1')
		with tf.Session() as sess:
			sess.run(init)
			writer.add_graph(sess.graph)
			for epoch in range(conf.num_epochs):
				for batches in range(conf.num_batches):
					batch_message, batch_key = get_batch_sized_data(conf.batch_size, conf.plain_text_length, conf.plain_text_length)
					if batches % 50 == 0:
						# write summaries.
						s = model.run_batch(sess, batch_message, batch_key, True)
						print("Summarizing - " + str((conf.num_batches * epoch) + batches))
						writer.add_summary(s, (conf.num_batches * epoch) + batches)
					else:
						res = model.run_batch(sess, batch_message, batch_key)
	print(res)
	print("Hello")
	return

if __name__ == "__main__":
  test_run()