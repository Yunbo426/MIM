import tensorflow as tf
from pred_learn.layers.TensorLayerNorm import tensor_layer_norm

class ConvLSTMCell():
    def __init__(self, layer_name, filter_size, num_hidden, seq_shape, tln=True, initializer=0.001):
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
        if initializer == -1:
            self.initializer = None
        else:
            self.initializer = tf.random_uniform_initializer(-initializer,initializer)

    def init_state(self):
        shape = [self.batch, self.height, self.width, self.num_hidden]
        return tf.zeros(shape, dtype=tf.float32)

    def __call__(self, x, h_t, c_t):
        if h_t is None:
            h_t = self.init_state()
        if c_t is None:
            c_t = self.init_state()
        with tf.variable_scope(self.layer_name):
            h_concat = tf.layers.conv2d(h_t, self.num_hidden * 4,
                                        self.filter_size, 1, padding='same',
                                        kernel_initializer=self.initializer,
                                        name='state_to_state')
            if self.layer_norm:
                h_concat = tensor_layer_norm(h_concat, 'state_to_state')
            i_h, g_h, f_h, o_h = tf.split(h_concat, 4, 3)

            ct_weight = tf.get_variable(
                'c_t_weight', [self.height,self.width,self.num_hidden*2])
            ct_activation = tf.multiply(tf.tile(c_t, [1,1,1,2]), ct_weight)
            i_c, f_c = tf.split(ct_activation, 2, 3)

            i_ = i_h + i_c
            f_ = f_h + f_c
            g_ = g_h
            o_ = o_h

            if x != None:
                x_concat = tf.layers.conv2d(x, self.num_hidden * 4,
                                            self.filter_size, 1,
                                            padding='same',
                                            kernel_initializer=self.initializer,
                                            name='input_to_state')
                if self.layer_norm:
                    x_concat = tensor_layer_norm(x_concat, 'input_to_state')
                i_x, g_x, f_x, o_x = tf.split(x_concat, 4, 3)

                i_ += i_x
                f_ += f_x
                g_ += g_x
                o_ += o_x

            i_ = tf.nn.sigmoid(i_)
            f_ = tf.nn.sigmoid(f_ + self._forget_bias)
            c_new = f_ * c_t + i_ * tf.nn.tanh(g_)

            oc_weight = tf.get_variable(
                'oc_weight', [self.height,self.width,self.num_hidden])
            o_c = tf.multiply(c_new, oc_weight)

            h_new = tf.nn.sigmoid(o_ + o_c) * tf.nn.tanh(c_new)

            return h_new, c_new

