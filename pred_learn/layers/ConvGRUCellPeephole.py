import tensorflow as tf
from pred_learn.layers.TensorLayerNorm import tensor_layer_norm
import math

class ConvGRUCell():
    def __init__(self, layer_name, filter_size, num_hidden, seq_shape, tln=True, initializer=None):
        """Initialize the basic Conv LSTM cell.
        Args:
            layer_name: layer names for different convlstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden: number of units in output tensor.
            tln: whether to apply tensor layer normalization.
        """
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden = num_hidden
        self.layer_norm = tln
        self.batch = seq_shape[0]
        self.height = seq_shape[2]
        self.width = seq_shape[3]
        self._forget_bias = 1.0

        def w_initializer(dim_in, dim_out):
            random_range = math.sqrt(6.0 / (dim_in + dim_out))
            return tf.random_uniform_initializer(-random_range, random_range)
        if initializer is None or initializer == -1:
            self.initializer = w_initializer
        else:
            self.initializer = tf.random_uniform_initializer(-initializer, initializer)

    def init_state(self, x):
        shape = [x.get_shape().as_list()[0], x.get_shape().as_list()[1], x.get_shape().as_list()[2], self.num_hidden]
        # shape = [self.batch, self.height, self.width, self.num_hidden]
        return tf.zeros(shape, dtype=tf.float32)

    def __call__(self, x, c_t):
        if c_t is None:
            c_t = self.init_state(x)
        with tf.variable_scope(self.layer_name):
            x1 = tf.concat([x, c_t], axis=-1)
            x_t_in = x1.get_shape().as_list()[-1]
            g_concat = tf.layers.conv2d(x1, self.num_hidden * 2,
                                        self.filter_size, 1, padding='same',
                                        kernel_initializer=self.initializer(x_t_in, self.num_hidden*2),
                                        name='g12')
            if self.layer_norm:
                g_concat = tensor_layer_norm(g_concat, 'g12')
            g1, g2 = tf.split(tf.sigmoid(g_concat), 2, 3)

            x2 = tf.concat([x, c_t * g1], axis=-1)
            x_t_in = x2.get_shape().as_list()[-1]
            g3 = tf.layers.conv2d(x2, self.num_hidden,
                                  self.filter_size, 1, padding='same',
                                  kernel_initializer=self.initializer(x_t_in, self.num_hidden),
                                  name='g3')
            if self.layer_norm:
                g3 = tensor_layer_norm(g3, 'g3')
            g3 = tf.nn.tanh(g3)

            c_new = g2 * g3 + (1 - g2) * c_t
            return c_new
