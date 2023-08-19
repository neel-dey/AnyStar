#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taken directly from SynthSeg's public repo:
https://github.com/BBillot/SynthSeg/blob/master/ext/lab2im/utils.py
"""


import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


def draw_value_from_distribution(hyperparameter,
                                 size=1,
                                 distribution='uniform',
                                 centre=0.,
                                 default_range=10.0,
                                 positive_only=False,
                                 return_as_tensor=False,
                                 batchsize=None):
    """Sample values from a uniform, or normal distribution of given hyper-parameters.
    These hyper-parameters are to the number of 2 in both uniform and normal cases.
    :param hyperparameter: values of the hyper-parameters. Can either be:
    1) None, in each case the two hyper-parameters are given by [center-default_range, center+default_range],
    2) a number, where the two hyper-parameters are given by [centre-hyperparameter, centre+hyperparameter],
    3) a sequence of length 2, directly defining the two hyper-parameters: [min, max] if the distribution is uniform,
    [mean, std] if the distribution is normal.
    4) a numpy array, with size (2, m). In this case, the function returns a 1d array of size m, where each value has
    been sampled independently with the specified hyper-parameters. If the distribution is uniform, rows correspond to
    its lower and upper bounds, and if the distribution is normal, rows correspond to its mean and std deviation.
    5) a numpy array of size (2*n, m). Same as 4) but we first randomly select a block of two rows among the
    n possibilities.
    6) the path to a numpy array corresponding to case 4 or 5.
    7) False, in which case this function returns None.
    :param size: (optional) number of values to sample. All values are sampled independently.
    Used only if hyperparameter is not a numpy array.
    :param distribution: (optional) the distribution type. Can be 'uniform' or 'normal'. Default is 'uniform'.
    :param centre: (optional) default centre to use if hyperparameter is None or a number.
    :param default_range: (optional) default range to use if hyperparameter is None.
    :param positive_only: (optional) wheter to reset all negative values to zero.
    :param return_as_tensor: (optional) whether to return the result as a tensorflow tensor
    :param batchsize: (optional) if return_as_tensor is true, then you can sample a tensor of a given batchsize. Give
    this batchsize as a tensorflow tensor here.
    :return: a float, or a numpy 1d array if size > 1, or hyperparameter is itself a numpy array.
    Returns None if hyperparmeter is False.
    """

    # return False is hyperparameter is False
    if hyperparameter is False:
        return None

    # reformat parameter_range
    hyperparameter = load_array_if_path(hyperparameter, load_as_numpy=True)
    if not isinstance(hyperparameter, np.ndarray):
        if hyperparameter is None:
            hyperparameter = np.array([[centre - default_range] * size, [centre + default_range] * size])
        elif isinstance(hyperparameter, (int, float)):
            hyperparameter = np.array([[centre - hyperparameter] * size, [centre + hyperparameter] * size])
        elif isinstance(hyperparameter, (list, tuple)):
            assert len(hyperparameter) == 2, 'if list, parameter_range should be of length 2.'
            hyperparameter = np.transpose(np.tile(np.array(hyperparameter), (size, 1)))
        else:
            raise ValueError('parameter_range should either be None, a nummber, a sequence, or a numpy array.')
    elif isinstance(hyperparameter, np.ndarray):
        assert hyperparameter.shape[0] % 2 == 0, 'number of rows of parameter_range should be divisible by 2'
        n_modalities = int(hyperparameter.shape[0] / 2)
        modality_idx = 2 * np.random.randint(n_modalities)
        hyperparameter = hyperparameter[modality_idx: modality_idx + 2, :]

    # draw values as tensor
    if return_as_tensor:
        shape = KL.Lambda(lambda x: tf.convert_to_tensor(hyperparameter.shape[1], 'int32'))([])
        if batchsize is not None:
            shape = KL.Lambda(lambda x: tf.concat([x[0], tf.expand_dims(x[1], axis=0)], axis=0))([batchsize, shape])
        if distribution == 'uniform':
            parameter_value = KL.Lambda(lambda x: tf.random.uniform(shape=x,
                                                                    minval=hyperparameter[0, :],
                                                                    maxval=hyperparameter[1, :]))(shape)
        elif distribution == 'normal':
            parameter_value = KL.Lambda(lambda x: tf.random.normal(shape=x,
                                                                   mean=hyperparameter[0, :],
                                                                   stddev=hyperparameter[1, :]))(shape)
        else:
            raise ValueError("Distribution not supported, should be 'uniform' or 'normal'.")

        if positive_only:
            parameter_value = KL.Lambda(lambda x: K.clip(x, 0, None))(parameter_value)

    # draw values as numpy array
    else:
        if distribution == 'uniform':
            parameter_value = np.random.uniform(low=hyperparameter[0, :], high=hyperparameter[1, :])
        elif distribution == 'normal':
            parameter_value = np.random.normal(loc=hyperparameter[0, :], scale=hyperparameter[1, :])
        else:
            raise ValueError("Distribution not supported, should be 'uniform' or 'normal'.")

        if positive_only:
            parameter_value[parameter_value < 0] = 0

    return parameter_value


def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var


def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this funtion returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformated list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int32, np.int64, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var


def add_axis(x, axis=0):
    """Add axis to a numpy array.
    :param x: input array
    :param axis: index of the new axis to add. Can also be a list of indices to add several axes at the same time."""
    axis = reformat_to_list(axis)
    for ax in axis:
        x = np.expand_dims(x, axis=ax)
    return x


class SampleConditionalGMM(Layer):
    """This layer generates an image by sampling a Gaussian Mixture Model conditioned on a label map given as input.
    The parameters of the GMM are given as two additional inputs to the layer (means and standard deviations):
    image = SampleConditionalGMM(generation_labels)([label_map, means, stds])
    :param generation_labels: list of all possible label values contained in the input label maps.
    Must be a list or a 1D numpy array of size N, where N is the total number of possible label values.
    Layer inputs:
    label_map: input label map of shape [batchsize, shape_dim1, ..., shape_dimn, n_channel].
    All the values of label_map must be contained in generation_labels, but the input label_map doesn't necesseraly have
    to contain all the values in generation_labels.
    means: tensor containing the mean values of all Gaussian distributions of the GMM.
           It must be of shape [batchsize, N, n_channel], and in the same order as generation label,
           i.e. the ith value of generation_labels will be associated to the ith value of means.
    stds: same as means but for the standard deviations of the GMM.
    """

    def __init__(self, generation_labels, **kwargs):
        self.generation_labels = generation_labels
        self.n_labels = None
        self.n_channels = None
        self.max_label = None
        self.indices = None
        self.shape = None
        super(SampleConditionalGMM, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["generation_labels"] = self.generation_labels
        return config

    def build(self, input_shape):

        # check n_labels and n_channels
        assert len(input_shape) == 3, 'should have three inputs: labels, means, std devs (in that order).'
        self.n_channels = input_shape[1][-1]
        self.n_labels = len(self.generation_labels)
        assert self.n_labels == input_shape[1][1], 'means should have the same number of values as generation_labels'
        assert self.n_labels == input_shape[2][1], 'stds should have the same number of values as generation_labels'

        # scatter parameters (to build mean/std lut)
        self.max_label = np.max(self.generation_labels) + 1
        indices = np.concatenate([self.generation_labels + self.max_label * i for i in range(self.n_channels)], axis=-1)
        #self.shape = tf.convert_to_tensor([np.max(indices) + 1], dtype=tf.int32)
        self.shape = tf.convert_to_tensor([(np.max(indices) + 1).astype(np.int32)], dtype=tf.int32)
        self.indices = tf.convert_to_tensor(add_axis(indices, axis=[0, -1]), dtype=tf.int32)

        self.built = True
        super(SampleConditionalGMM, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # reformat labels and scatter indices
        batch = tf.split(tf.shape(inputs[0]), [1, -1])[0]
        tmp_indices = tf.tile(self.indices, tf.concat([batch, tf.convert_to_tensor([1, 1], dtype='int32')], axis=0))
        labels = tf.concat([tf.cast(inputs[0], dtype='int32') + self.max_label * i for i in range(self.n_channels)], -1)

        # build mean map
        means = tf.concat([inputs[1][..., i] for i in range(self.n_channels)], 1)
        tile_shape = tf.concat([batch, tf.convert_to_tensor([1, ], dtype='int32')], axis=0)
        means = tf.tile(tf.expand_dims(tf.scatter_nd(tmp_indices, means, self.shape), 0), tile_shape)
        means_map = tf.map_fn(lambda x: tf.gather(x[0], x[1]), [means, labels], dtype=tf.float32)

        # same for stds
        stds = tf.concat([inputs[2][..., i] for i in range(self.n_channels)], 1)
        stds = tf.tile(tf.expand_dims(tf.scatter_nd(tmp_indices, stds, self.shape), 0), tile_shape)
        stds_map = tf.map_fn(lambda x: tf.gather(x[0], x[1]), [stds, labels], dtype=tf.float32)

        return stds_map * tf.random.normal(tf.shape(labels)) + means_map

    def compute_output_shape(self, input_shape):
        return input_shape[0] if (self.n_channels == 1) else tuple(list(input_shape[0][:-1]) + [self.n_channels])
