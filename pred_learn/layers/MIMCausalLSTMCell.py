__author__ = 'yunbo'

import tensorflow as tf
from pred_learn.layers.TensorLayerNorm import tensor_layer_norm
import math


class MIMCausalLSTMCell:
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden_out,
                 seq_shape, forget_bias=1.0, tln=False, initializer=None):
        """Initialize the Causal LSTM cell.
        Args:
            layer_name: layer names for different lstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden_in: number of units for input tensor.
            num_hidden_out: number of units for output tensor.
            seq_shape: shape of a sequence.
            forget_bias: float, The bias added to forget gates.
            tln: whether to apply tensor layer normalization
        """
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden_out
        self.convlstm_c = None
        self.batch = seq_shape[0]
        self.height = seq_shape[2]
        self.width = seq_shape[3]
        self.layer_norm = tln
        self._forget_bias = forget_bias

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
            h = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden],
                         dtype=tf.float32)
        if c is None:
            c = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden],
                         dtype=tf.float32)
        if m is None:
            m = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden_in],
                         dtype=tf.float32)
        if diff_h is None:
            diff_h = tf.zeros_like(h)

        with tf.variable_scope(self.layer_name):
            h_cc = tf.layers.conv2d(
                h, self.num_hidden*3,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(self.num_hidden_in, self.num_hidden*3),
                name='time_state_to_state')
            m_cc = tf.layers.conv2d(
                m, self.num_hidden*3,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(self.num_hidden_in, self.num_hidden*3),
                name='spatio_state_to_state')
            if self.layer_norm:
                h_cc = tensor_layer_norm(h_cc, 'h2c')
                m_cc = tensor_layer_norm(m_cc, 'm2m')

            i_h, g_h, o_h = tf.split(h_cc, 3, 3)
            i_m, f_m, m_m = tf.split(m_cc, 3, 3)

            if x is None:
                i = tf.sigmoid(i_h)
                g = tf.tanh(g_h)
            else:
                x_shape_in = x.get_shape().as_list()[-1]
                x_cc = tf.layers.conv2d(
                    x, self.num_hidden*6,
                    self.filter_size, 1, padding='same',
                    kernel_initializer=self.initializer(x_shape_in, self.num_hidden*6),
                    name='input_to_state')
                if self.layer_norm:
                    x_cc = tensor_layer_norm(x_cc, 'x2c')

                i_x, g_x, o_x, i_x_, g_x_, f_x_ = tf.split(x_cc, 6, 3)

                i = tf.sigmoid(i_x + i_h)
                g = tf.tanh(g_x + g_h)

            c, self.convlstm_c = self.convlstm(diff_h, c, self.convlstm_c)
            c_new = c + i * g

            c_cc = tf.layers.conv2d(
                c_new, self.num_hidden*4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(self.num_hidden, self.num_hidden*4),
                name='c2m')
            if self.layer_norm:
                c_cc = tensor_layer_norm(c_cc, 'c2m')

            i_c, g_c, f_c, o_c = tf.split(c_cc, 4, 3)

            if x is None:
                ii = tf.sigmoid(i_c + i_m)
                ff = tf.sigmoid(f_c + f_m + self._forget_bias)
                gg = tf.tanh(g_c)
            else:
                ii = tf.sigmoid(i_c + i_x_ + i_m)
                ff = tf.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
                gg = tf.tanh(g_c + g_x_)

            m_new = ff * tf.tanh(m_m) + ii * gg

            o_m = tf.layers.conv2d(
                m_new, self.num_hidden,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(self.num_hidden, self.num_hidden),
                name='m_to_o')
            if self.layer_norm:
                o_m = tensor_layer_norm(o_m, 'm_to_o')

            if x is None:
                o = tf.tanh(o_h + o_c + o_m)
            else:
                o = tf.tanh(o_x + o_h + o_c + o_m)

            cell = tf.concat([c_new, m_new],-1)
            cell = tf.layers.conv2d(cell, self.num_hidden, 1, 1, padding='same',
                                    kernel_initializer=self.initializer(self.num_hidden*2, self.num_hidden),
                                    name='cell_reduce')

            h_new = o * tf.tanh(cell)

            return h_new, c_new, m_new


