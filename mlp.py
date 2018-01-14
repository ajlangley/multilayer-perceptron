import tensorflow as tf
from tensorflow.python import debug as tf_debug
import os


class PerceptronLayer:
    def __init__(self, input_size, output_size, layer_id, activation=None):
        self.W = tf.get_variable(name='W_' + str(layer_id),
                                 shape=[output_size, input_size],
                                 dtype=tf.float32,
                                 trainable=True,
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable(name='b_' + str(layer_id),
                                 dtype=tf.float32,
                                 trainable=True,
                                 initializer=tf.zeros([output_size, 1]))
        self.layer_id = layer_id
        self.activation = activation

    def forward_pass(self, x):
        z = tf.matmul(self.W, x, name='dot_' + str(self.layer_id)) + self.b
        if self.activation:
            h = self.activation(z)
            return h
        else:
            return z


class MLP:
    def __init__(self, n_features, n_classes, layer_sizes):
        self.input_sizes = [n_features] + layer_sizes
        self.output_sizes = layer_sizes + [n_classes]

    def __build_training_graph__(self, learning_rate, batch_size, debug):
        self.training_graph = tf.Graph()
        self.training_session = tf.Session(graph=self.training_graph)
        if debug:
            self.training_session = tf_debug.LocalCLIDebugWrapperSession(self.training_session)
            self.training_session.add_tensor_filter('has_inf_or_nan', tf.debug.has_inf_or_nan)

        with self.training_graph.as_default():
            self.x = tf.placeholder(dtype=tf.float32,
                                    shape=[batch_size, self.input_sizes[0]],
                                    name='x')
            self.y = tf.placeholder(dtype=tf.int32,
                                    shape=batch_size,
                                    name='y')
            true_dist = tf.one_hot(indices=self.y,
                                   depth=self.output_sizes[-1])

            hidden_layers = [PerceptronLayer(input_size, output_size, layer_id, activation=tf.nn.sigmoid) for
                             input_size, output_size, layer_id
                             in zip(self.input_sizes, self.output_sizes, range(len(self.output_sizes) - 1))] \
                            + [PerceptronLayer(self.input_sizes[-1], self.output_sizes[-1], len(self.output_sizes))]

            h = tf.transpose(self.x)
            for layer in hidden_layers:
                h = layer.forward_pass(h)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_dist,
                                                                               logits=tf.transpose(h),
                                                                               name='loss'))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)

            init_op = tf.global_variables_initializer()
            self.training_session.run(init_op)

            self.saver = tf.train.Saver()

        self.training_graph.finalize()

    def training_step(self, x, y):
        feed_dict = {self.x: x,
                     self.y: y}
        _, loss = self.training_session.run([self.optimizer, self.loss], feed_dict=feed_dict)

        return loss

    def validation_step(self, x, y):
        feed_dict = {self.x: x,
                     self.y: y}
        loss = self.training_session.run(self.loss, feed_dict=feed_dict)

        return loss

    def save(self, log_dir, step):
        filepath = os.path.join('training-logs', log_dir)
        if not os.path.exists(filepath):
            os.mkdir('training-logs')

        self.saver.save(self.training_session, os.path.join(filepath, 'model.ckpt'), step)
