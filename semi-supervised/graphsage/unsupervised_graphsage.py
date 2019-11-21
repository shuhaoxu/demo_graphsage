#!/usr/bin/python

from __future__ import division, absolute_import, print_function

import math
import tensorflow as tf

from models import GeneralizedModel
from prediction import BipartiteEdgePredLayer
from aggregators import MeanAggregator, GCNAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

class SampleAndAggregate(GeneralizedModel):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, placeholders, features, adj, degrees,
                 layer_infos, concat=True, aggregator_type="mean",
                 model_size="small", identity_dim=0,
                 **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
                        NOTE: Pass a None object to train in featureless mode (identity features for nodes)!
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all the recursive
                           layers. See SAGEInfo definition in 'model.py'.
            - concat: whether to concatenate during recursive iterations.
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - identity_dim: Set to positive int to use identity features (slow and cannot generalized,
                            but better accuracy).
        '''
        super(SampleAndAggregate, self).__init__(**kwargs)
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1      = placeholders['batch1']
        self.inputs2      = placeholders['batch2']
        self.batch_size   = placeholders['batch_size']
        self.placeholders = placeholders
        self.model_size   = model_size
        self.layer_infos  = layer_infos
        self.adj_info     = adj

        if identity_dim > 0:
            self.embeds = tf.get_variable('node_embeddings', [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if"
                                 "no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat  = concat

        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def sample(self, inputs, layer_infos, batch_size=None):
        """
        Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            - inputs: batch_inputs
            - batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size  = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler      = layer_infos[t].neigh_sampler
            node         = sampler((samples[k], layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * batch_size, ]))
            support_sizes.append(support_size)
        return samples, support_sizes 

    def aggregate(self, samples, input_features, dims, num_samples, support_sizes,
                  batch_size=None, aggregators=None, name=None, concat=False,
                  model_size="small"):
        """
        At each layer, aggregate hidden representations of neighbors to compute the hidden representations
        at next layer.

        Args:
            - samples: A list of samples of variable hops away for convolving at each layer of the 
                       network. Length is the number of layers+1. Each is a vector of node indices.
            - input_features: The input features for each sample of various hops away.
            - dims: A list of dimensions of the hidden representations from the input layer to the 
                    final layer. Length is the number of layers+1.
            - num_samples: List of number of samples for each layer.
            - support_sizes: The number of nodes to gather information from for each layer.
            - batch_size: the number of inputs (different for batch inputs and negative samples).

        Returns:
            The hidden representation at the final layer for all nodes in batch.
        """

        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        hidden  = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples)-1:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                                                     act=lambda x: x,
                                                     dropout=self.placeholders['dropout'],
                                                     name=name,
                                                     concat=concat,
                                                     model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                                                     dropout=self.placeholders['dropout'],
                                                     name=name,
                                                     concat=concat,
                                                     model_size=model_size)
                aggregators.append(aggregator)
                #self.layers.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decrease
            for hop in range(len(num_samples) - layer):
                dim_mult   = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples)-hop-1],
                              dim_mult*dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop+1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators

    def _build(self):
        labels = tf.reshape(tf.cast(self.placeholders['batch2'], dtype=tf.int64),
                            [self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(true_classes=labels,
                                                                        num_true=1,
                                                                        num_sampled=FLAGS.neg_sample_size,
                                                                        unique=False,
                                                                        range_max=len(self.degrees),
                                                                        distortion=0.75,
                                                                        unigrams=self.degrees.tolist()))

        # perform "convolution"
        ## samples from nodes share same edges
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, 
                                                         [self.features], 
                                                         self.dims, 
                                                         num_samples,
                                                         support_sizes1, 
                                                         concat=self.concat,
                                                         model_size=self.model_size)
        self.outputs2, _                = self.aggregate(samples2,
                                                         [self.features],
                                                         self.dims,
                                                         num_samples,
                                                         support_sizes2,
                                                         aggregators=self.aggregators,
                                                         concat=self.concat,
                                                         model_size=self.model_size)
        ## negative samples
        neg_samples, neg_support_sizes  = self.sample(self.neg_samples, self.layer_infos, FLAGS.neg_sample_size)
        self.neg_outputs, _             = self.aggregate(neg_samples,
                                                         [self.features],
                                                         self.dims,
                                                         num_samples,
                                                         neg_support_sizes,
                                                         batch_size=FLAGS.neg_sample_size,
                                                         aggregators=self.aggregators,
                                                         concat=self.concat,
                                                         model_size=self.model_size)

        # compute outputs
        dim_mult = 2 if self.concat else 1
        self.link_pred_layer = BipartiteEdgePredLayer(dim_mult*self.dims[-1],
                                                      dim_mult*self.dims[-1],
                                                      self.placeholders,
                                                      act=tf.nn.sigmoid,
                                                      loss_fn='xent',
                                                      name='edge_predict')
        #self.layers.append(self.link_pred_layer)
        self.outputs1    = tf.nn.l2_normalize(self.outputs1, 1)
        self.outputs2    = tf.nn.l2_normalize(self.outputs2, 1)
        self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)
        #self.layers.append(self.outputs1)

    def build(self):
        self._build()

        # TF graph management
        self._loss()
        self._accuracy()
        self.loss              = self.loss / tf.cast(self.batch_size, tf.float32)
        grads_and_vars         = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                   for grad,var in grads_and_vars]
        self.grad, _           = clipped_grads_and_vars[0]
        self.opt_op            = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs)
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        # shape: [batch_size]
        aff          = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
        _aff         = tf.expand_dims(aff, axis=1)
        # shape: [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        # 
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size         = tf.shape(self.aff_all)[1]
        #
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks       = tf.nn.top_k(-indices_of_ranks, k=size)
        # 
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:,-1]+1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)
