import tensorflow as tf
from pred_learn.layers.TensorLayerNorm import tensor_layer_norm
import math


class DiffSpatioTemporalLSTMCell():
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
                 seq_shape, tln=False, initializer=None):
        """Initialize the basic Conv LSTM cell.
        Args:
            layer_name: layer names for different convlstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden: number of units in output tensor.
            forget_bias: float, The bias added to forget gates (see above).
            tln: whether to apply tensor layer normalization
        """
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden
        self.convlstm_c = None
        self.batch = seq_shape[0]
        self.height = seq_shape[2]
        self.width = seq_shape[3]
        self.layer_norm = tln
        self._forget_bias = 1.0

        def w_initializer(dim_in, dim_out):
            random_range = math.sqrt(6.0 / (dim_in + dim_out))
            return tf.random_uniform_initializer(-random_range, random_range)
        if initializer is None or initializer == -1:
            self.initializer = w_initializer
        else:
            self.initializer = tf.random_uniform_initializer(-initializer, initializer)

    def init_state(self):
        return tf.zeros([self.batch, self.height, self.width, self.num_hidden],
                        dtype=tf.float32)

    def convlstm(self, x, h_t, c_t):
        if h_t is None:
            h_t = self.init_state()
        if c_t is None:
            c_t = self.init_state()
        with tf.variable_scope(self.layer_name):
            h_concat = tf.layers.conv2d(h_t, self.num_hidden * 4,
                                        self.filter_size, 1, padding='same',
                                        kernel_initializer=self.initializer(self.num_hidden, self.num_hidden * 4),
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
                                            kernel_initializer=self.initializer(self.num_hidden, self.num_hidden * 4),
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

    def __call__(self, x, diff_h, h, c, m):
        if h is None:
            h = self.init_state()
        if c is None:
            c = self.init_state()
        if m is None:
            m = self.init_state()
        if diff_h is None:
            diff_h = tf.zeros_like(h)

        with tf.variable_scope(self.layer_name):
            t_cc = tf.layers.conv2d(
                h, self.num_hidden * 3,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(self.num_hidden, self.num_hidden * 3),
                name='time_state_to_state')
            s_cc = tf.layers.conv2d(
                m, self.num_hidden * 4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(self.num_hidden, self.num_hidden * 4),
                name='spatio_state_to_state')
            x_shape_in = x.get_shape().as_list()[-1]
            x_cc = tf.layers.conv2d(
                x, self.num_hidden * 4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(x_shape_in, self.num_hidden * 4),
                name='input_to_state')
            if self.layer_norm:
                t_cc = tensor_layer_norm(t_cc, 'time_state_to_state')
                s_cc = tensor_layer_norm(s_cc, 'spatio_state_to_state')
                x_cc = tensor_layer_norm(x_cc, 'input_to_state')

            i_s, g_s, f_s, o_s = tf.split(s_cc, 4, 3)
            i_t, g_t, o_t = tf.split(t_cc, 3, 3)
            i_x, g_x, f_x, o_x = tf.split(x_cc, 4, 3)

            i = tf.nn.sigmoid(i_x + i_t)
            i_ = tf.nn.sigmoid(i_x + i_s)
            g = tf.nn.tanh(g_x + g_t)
            g_ = tf.nn.tanh(g_x + g_s)
            f_ = tf.nn.sigmoid(f_x + f_s + self._forget_bias)
            o = tf.nn.sigmoid(o_x + o_t + o_s)
            new_m = f_ * m + i_ * g_
            c, self.convlstm_c = self.convlstm(diff_h, c, self.convlstm_c)
            new_c = c + i * g
            cell = tf.concat([new_c, new_m], 3)
            cell = tf.layers.conv2d(cell, self.num_hidden, 1, 1,
                                    padding='same', name='cell_reduce')
            new_h = o * tf.nn.tanh(cell)

            return new_h, new_c, new_m
