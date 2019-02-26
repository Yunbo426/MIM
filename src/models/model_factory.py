import os

import tensorflow as tf

from src.utils import optimizer
from src.models import mim


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        # inputs
        if configs.img_height > 0:
            height = configs.img_height
        else:
            height = configs.img_width
        self.x = [tf.placeholder(tf.float32,
                                 [self.configs.batch_size,
                                  self.configs.total_length,
                                  self.configs.img_width // self.configs.patch_size,
                                  height // self.configs.patch_size,
                                  self.configs.patch_size * self.configs.patch_size * self.configs.img_channel])
                  for i in range(self.configs.n_gpu)]

        self.real_input_flag = tf.placeholder(tf.float32,
                                        [self.configs.batch_size,
                                         self.configs.total_length - self.configs.input_length - 1,
                                         self.configs.img_width // self.configs.patch_size,
                                         height // self.configs.patch_size,
                                         self.configs.patch_size * self.configs.patch_size * self.configs.img_channel])

        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        self.params = dict()
        self.params.update(self.configs.__dict__['__flags'])
        num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        num_layers = len(num_hidden)
        for i in range(self.configs.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(tf.get_variable_scope(),
                                       reuse=True if i > 0 else None):
                    # define a model
                    output_list = self.construct_model(
                        self.configs.model_name,
                        self.x[i],
                        self.params,
                        self.real_input_flag,
                        num_layers,
                        num_hidden,
                        self.configs.filter_size,
                        self.configs.stride,
                        self.configs.total_length,
                        self.configs.input_length,
                        self.configs.layer_norm)

                    gen_ims = output_list[0]
                    loss = output_list[1]
                    if len(output_list) > 2:
                        self.debug = output_list[2]
                    else:
                        self.debug = []
                    pred_ims = gen_ims[:, self.configs.input_length - self.configs.total_length:]
                    loss_train.append(loss / self.configs.batch_size)
                    # gradients
                    all_params = tf.trainable_variables()
                    grads.append(tf.gradients(loss, all_params))
                    self.pred_seq.append(pred_ims)

        # add losses and gradients together and get training updates
        with tf.device('/gpu:0'):
            for i in range(1, self.configs.n_gpu):
                loss_train[0] += loss_train[i]
                for j in range(len(grads[0])):
                    grads[0][j] += grads[i][j]
        # keep track of moving average
        ema = tf.train.ExponentialMovingAverage(decay=0.9995)
        maintain_averages_op = tf.group(ema.apply(all_params))
        self.train_op = tf.group(optimizer.adam_updates(
            all_params, grads[0], lr=self.tf_lr, mom1=0.95, mom2=0.9995),
            maintain_averages_op)

        self.loss_train = loss_train[0] / self.configs.n_gpu

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = configs.allow_gpu_growth
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config=configProt)
        self.sess.run(init)
        if self.configs.pretrained_model:
            self.saver.restore(self.sess, self.configs.pretrained_model)

    def train(self, inputs, lr, real_input_flag):
        feed_dict = {self.x[i]: inputs[i] for i in range(self.configs.n_gpu)}
        feed_dict.update({self.tf_lr: lr})
        feed_dict.update({self.real_input_flag: real_input_flag})
        loss, _, debug = self.sess.run((self.loss_train, self.train_op, self.debug), feed_dict)
        return loss

    def test(self, inputs, real_input_flag):
        feed_dict = {self.x[i]: inputs[i] for i in range(self.configs.n_gpu)}
        feed_dict.update({self.real_input_flag: real_input_flag})
        gen_ims, debug = self.sess.run((self.pred_seq, self.debug), feed_dict)
        return gen_ims, debug

    def save(self, itr):
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + self.configs.save_dir)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        self.saver.restore(self.sess, checkpoint_path)

    def construct_model(self, name, images, model_params, real_input_flag, num_layers, num_hidden,
                        filter_size, stride, total_length, input_length, tln):
        '''Returns a sequence of generated frames
        Args:
            name: [predrnn_pp]
            params: dict for extra parameters of some models
            real_input_flag: for schedualed sampling.
            num_hidden: number of units in a lstm layer.
            filter_size: for convolutions inside lstm.
            stride: for convolutions inside lstm.
            total_length: including ins and outs.
            input_length: for inputs.
            tln: whether to apply tensor layer normalization.
        Returns:
            gen_images: a seq of frames.
            loss: [l2 / l1+l2].
        Raises:
            ValueError: If network `name` is not recognized.
        '''

        networks_map = {
            'mim': mim.mim,
        }

        params = dict(mask=real_input_flag, num_layers=num_layers, num_hidden=num_hidden, filter_size=filter_size,
                      stride=stride, total_length=total_length, input_length=input_length, is_training=True)
        params.update(model_params)
        if name in networks_map:
            func = networks_map[name]
            return func(images, params, real_input_flag, num_layers, num_hidden, filter_size,
                        stride, total_length, input_length, tln)
        else:
            raise ValueError('Name of network unknown %s' % name)
