__author__ = 'jianjin'

import tensorflow as tf
from src.layers.SpatioTemporalLSTMCellv2 import SpatioTemporalLSTMCell as stlstm
from src.layers.MIMBlock import MIMBlock as mimblock
from src.layers.MIMN import ConvLSTMCell as lstm
import math


def w_initializer(dim_in, dim_out):
    random_range = math.sqrt(6.0 / (dim_in + dim_out))
    return tf.random_uniform_initializer(-random_range, random_range)


def mim(images, params, schedual_sampling_bool, num_layers, num_hidden, filter_size,
        stride=1, total_length=20, input_length=10, tln=True):
    gen_images = []
    stlstm_layer = []
    stlstm_layer_diff = []
    cell_state = []
    hidden_state = []
    cell_state_diff = []
    hidden_state_diff = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1]

    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers - 1]
        else:
            num_hidden_in = num_hidden[i - 1]
        if i < 1:
            new_stlstm_layer = stlstm('stlstm_' + str(i + 1),
                                      filter_size,
                                      num_hidden_in,
                                      num_hidden[i],
                                      shape,
                                      tln=tln)
        else:
            new_stlstm_layer = mimblock('stlstm_' + str(i + 1),
                                        filter_size,
                                        num_hidden_in,
                                        num_hidden[i],
                                        shape,
                                        tln=tln)
        stlstm_layer.append(new_stlstm_layer)
        cell_state.append(None)
        hidden_state.append(None)

    for i in range(num_layers - 1):
        new_stlstm_layer = lstm('stlstm_diff' + str(i + 1),
                                filter_size,
                                num_hidden[i + 1],
                                shape,
                                tln=tln)
        stlstm_layer_diff.append(new_stlstm_layer)
        cell_state_diff.append(None)
        hidden_state_diff.append(None)

    st_memory = None

    for time_step in range(total_length - 1):
        reuse = bool(gen_images)
        with tf.variable_scope('predrnn', reuse=reuse):
            if time_step < input_length:
                x_gen = images[:,time_step]
            else:
                x_gen = schedual_sampling_bool[:,time_step-input_length]*images[:,time_step] + \
                        (1-schedual_sampling_bool[:,time_step-input_length])*x_gen
            preh = hidden_state[0]
            hidden_state[0], cell_state[0], st_memory = stlstm_layer[0](
                x_gen, hidden_state[0], cell_state[0], st_memory)
            for i in range(1, num_layers):
                if time_step > 0:
                    if i == 1:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = stlstm_layer_diff[i - 1](
                            hidden_state[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                    else:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = stlstm_layer_diff[i - 1](
                            hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
                else:
                    stlstm_layer_diff[i - 1](tf.zeros_like(hidden_state[i - 1]), None, None)
                preh = hidden_state[i]
                hidden_state[i], cell_state[i], st_memory = stlstm_layer[i](
                    hidden_state[i - 1], hidden_state_diff[i - 1], hidden_state[i], cell_state[i], st_memory)
            x_gen = tf.layers.conv2d(hidden_state[num_layers - 1],
                                     filters=output_channels,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     kernel_initializer=w_initializer(num_hidden[num_layers - 1], output_channels),
                                     name="back_to_pixel")
            gen_images.append(x_gen)

    gen_images = tf.stack(gen_images, axis=1)
    loss = tf.nn.l2_loss(gen_images - images[:, 1:])

    return [gen_images, loss]
