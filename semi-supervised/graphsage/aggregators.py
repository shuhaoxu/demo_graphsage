#!/usr/bin/python

from __future__ import division, absolute_import, print_function

import tensorflow as tf

from layers import Layer, Dense
from inits  import glorot, zeros

class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias    = bias
        self.act     = act
        self.concat  = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights']  = glorot([input_dim, output_dim],
                                                name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim  = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs  = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs   = tf.nn.dropout(self_vecs,  1-self.dropout)
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])
        from_self   = tf.matmul(self_vecs,   self.vars['self_weights'])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias    = bias
        self.act     = act
        self.concat  = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                          name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim  = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs  = tf.nn.dropout(self_vecs,  1-self.dropout)
        means      = tf.reduce_mean(tf.concat([neigh_vecs, tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)

        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
