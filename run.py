__author__ = 'yunbo'

import os

import tensorflow as tf
import numpy as np
from time import time

from src.data_provider import datasets_factory
from src.models.model_factory import Model
from src.utils import preprocess
import src.trainer as trainer

# -----------------------------------------------------------------------------
FLAGS = tf.app.flags.FLAGS

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# mode
tf.app.flags.DEFINE_boolean('is_training', True, 'training or testing')

# data I/O
tf.app.flags.DEFINE_string('dataset_name', 'mnist',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           'data/moving-mnist-example/moving-mnist-train.npz',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           'data/moving-mnist-example/moving-mnist-valid.npz',
                           'validation data paths.')
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/mnist_predrnn_pp',
                           'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_frm_dir', 'results/mnist_predrnn_pp',
                           'dir to store result.')
tf.app.flags.DEFINE_integer('input_length', 10,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('total_length', 20,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('img_width', 64,
                            'input image width.')
tf.app.flags.DEFINE_integer('img_channel', 1,
                            'number of image channel.')
# model[convlstm, predcnn, predrnn, predrnn_pp]
tf.app.flags.DEFINE_string('model_name', 'convlstm_net',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', '',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_string('num_hidden', '64,64,64,64',
                           'COMMA separated number of units in a convlstm layer.')
tf.app.flags.DEFINE_integer('filter_size', 5,
                            'filter of a convlstm layer.')
tf.app.flags.DEFINE_integer('stride', 1,
                            'stride of a convlstm layer.')
tf.app.flags.DEFINE_integer('patch_size', 1,
                            'patch size on one dimension.')
tf.app.flags.DEFINE_boolean('layer_norm', True,
                            'whether to apply tensor layer norm.')
# scheduled sampling
tf.app.flags.DEFINE_boolean('scheduled_sampling', True, 'for scheduled sampling')
tf.app.flags.DEFINE_integer('sampling_stop_iter', 50000, 'for scheduled sampling.')
tf.app.flags.DEFINE_float('sampling_start_value', 1.0, 'for scheduled sampling.')
tf.app.flags.DEFINE_float('sampling_changing_rate', 0.00002, 'for scheduled sampling.')
# optimization
tf.app.flags.DEFINE_float('lr', 0.001,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')
tf.app.flags.DEFINE_boolean('reverse_img', False,
                            'whether to reverse the input images while training.')
tf.app.flags.DEFINE_integer('batch_size', 8,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 80000,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 1000,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 1000,
                            'number of iters saving models.')
tf.app.flags.DEFINE_integer('num_save_samples', 10,
                            'number of sequences to be saved.')
tf.app.flags.DEFINE_integer('n_gpu', 1,
                            'how many GPUs to distribute the training across.')
# gpu 
tf.app.flags.DEFINE_boolean('allow_gpu_growth', False,
                            'allow gpu growth')

tf.app.flags.DEFINE_integer('img_height', 0,
                            'input image height.')


def main(argv=None):
    if tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.DeleteRecursively(FLAGS.save_dir)
    tf.gfile.MakeDirs(FLAGS.save_dir)
    if tf.gfile.Exists(FLAGS.gen_frm_dir):
        tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
    tf.gfile.MakeDirs(FLAGS.gen_frm_dir)

    gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(',') ,dtype=np.int32)
    FLAGS.n_gpu = len(gpu_list)
    print('Initializing models')

    model = Model(FLAGS)

    if FLAGS.is_training:
        train_wrapper(model)
    else:
        start = time()
        test_wrapper(model)
        stop = time()
        print("Time used: " + str(stop - start) + "s")


def schedule_sampling(eta, itr):
    if FLAGS.img_height > 0:
        height = FLAGS.img_height
    else:
        height = FLAGS.img_width
    zeros = np.zeros((FLAGS.batch_size,
                      FLAGS.total_length - FLAGS.input_length - 1,
                      FLAGS.img_width // FLAGS.patch_size,
                      height // FLAGS.patch_size,
                      FLAGS.patch_size ** 2 * FLAGS.img_channel))
    if not FLAGS.scheduled_sampling:
        return 0.0, zeros

    if itr < FLAGS.sampling_stop_iter:
        eta -= FLAGS.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (FLAGS.batch_size, FLAGS.total_length - FLAGS.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((FLAGS.img_width // FLAGS.patch_size,
                    height // FLAGS.patch_size,
                    FLAGS.patch_size ** 2 * FLAGS.img_channel))
    zeros = np.zeros((FLAGS.img_width // FLAGS.patch_size,
                      height // FLAGS.patch_size,
                      FLAGS.patch_size ** 2 * FLAGS.img_channel))
    real_input_flag = []
    for i in range(FLAGS.batch_size):
        for j in range(FLAGS.total_length - FLAGS.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (FLAGS.batch_size,
                            FLAGS.total_length - FLAGS.input_length - 1,
                            FLAGS.img_width // FLAGS.patch_size,
                            height // FLAGS.patch_size,
                            FLAGS.patch_size ** 2 * FLAGS.img_channel))
    return eta, real_input_flag


def train_wrapper(model):
    if FLAGS.pretrained_model:
        model.load(FLAGS.pretrained_model)
    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        FLAGS.dataset_name, FLAGS.train_data_paths, FLAGS.valid_data_paths,
        FLAGS.batch_size * FLAGS.n_gpu, FLAGS.img_width, seq_length=FLAGS.total_length, is_training=True)

    eta = FLAGS.sampling_start_value

    for itr in range(1, FLAGS.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()
        ims_reverse = None
        if FLAGS.reverse_img:
            ims_reverse = ims[:, :, :, ::-1]
            ims_reverse = preprocess.reshape_patch(ims_reverse, FLAGS.patch_size)
        ims = preprocess.reshape_patch(ims, FLAGS.patch_size)

        eta, real_input_flag = schedule_sampling(eta, itr)

        trainer.train(model, ims, real_input_flag, FLAGS, itr, ims_reverse)

        if itr % FLAGS.snapshot_interval == 0:
            model.save(itr)

        if itr % FLAGS.test_interval == 0:
            trainer.test(model, test_input_handle, FLAGS, itr)

        train_input_handle.next()


def test_wrapper(model):
    model.load(FLAGS.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        FLAGS.dataset_name, FLAGS.train_data_paths, FLAGS.valid_data_paths,
        FLAGS.batch_size * FLAGS.n_gpu, FLAGS.img_width, seq_length=FLAGS.total_length, is_training=False)
    trainer.test(model, test_input_handle, FLAGS, 'test_result')


if __name__ == '__main__':
    tf.app.run()

