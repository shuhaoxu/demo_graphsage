# template run on tensorflow framework: pytorch environment is building ... 
# docker image : e0d1; docker container : 00ca

from collections import namedtuple

import tensorflow as tf
import math

from prediction import BipartiteEdgePredLayer
from aggregators import MeanAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name','logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging      = kwargs.get('logging', False)
        self.logging = logging

        self.vars         = {}
        self.placeholders = {}

        self.layers      = []
        self.activations = []

        self.inputs  = None
        self.outputs = None

        self.loss      = 0
        self.accuracy  = 0
        self.optimizer = None
        self.opt_op    = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name:var for var in variables}

        # build metrics 
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver     = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print ("Model restored from file: %s" % save_path)

class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional, sequential layers.
    Subclasses must set self.outputs in _build method.

    (Removes the layers idiom from build method of the Model class)
    """
    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

"""
SAGEInfo is a namedtuple that specifies the parameters
of the recursive GraphSAGE layers.
"""
SAGEInfo = namedtuple("SAGEInfo",
                      ['layer_name',    # name of the layer (to get feature embedding etc.)
                       'neigh_sampler', # callable neigh_sampler constructor
                       'num_samples',
                       'output_dim'     # the output (i.e., hidden) dimension
                       ])
