#!/usr/bin/python

from __future__ import division, absolute_import, print_function

import os
import sys
import time
import json
import numpy as np
import networkx as nx
import tensorflow as tf
from networkx.readwrite import json_graph
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append('/demo_graphsage/semi-supervised/graphsage')
from utils import load_cora, split_data
from utils import log_dir, evaluate, incremental_evaluate, save_val_embeddings, construct_placeholders
from models import SAGEInfo
from minibatch import EdgeMinibatchIterator
from neigh_samplers import UniformNeighborSampler
from unsupervised_graphsage import SampleAndAggregate

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
## choice of log device.
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
## core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names, selected from graphsage_mean or gcn.')
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate.')
flags.DEFINE_string('model_size', 'small', "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '/demo_graphsage/cora/', 
                    'name of the object file that stores the training data.')
## left to default values in main experiments
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1-keep_probability)')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('dim_1', 128, 'size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 32, 'minibatch size')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features' 
                                        'of that dimension. Default 0.')
## logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 10, 'how often to run a validation minibatch')
flags.DEFINE_integer('validate_batch_size', 256, 'how many nodes per validation sample')
flags.DEFINE_integer('gpu', 0, 'which gpu to use')
flags.DEFINE_integer('print_every', 50, 'how often to print training info.')
flags.DEFINE_integer('max_total_steps', 100**10, 'maximum total number of iterations')

# GPU Setting
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
GPU_MEM_FRACTION = 0.8

def run_cora():
    # Load Data
    json_file = FLAGS.train_prefix + 'cora-G.json'
    G_data    = json.load(open(json_file))
    G         = json_graph.node_link_graph(G_data)
    num_nodes = 2708
    feat_data, labels, adj_lists, id_map = load_cora()
    ## pad with dummy zero vector
    if not feat_data is None:
        feat_data = np.vstack([feat_data, np.zeros((feat_data.shape[1],))])

    # Construct Iterator
    placeholders = construct_placeholders()
    minibatch    = EdgeMinibatchIterator(G=G, 
                                         id2idx=id_map,
                                         placeholders=placeholders,
                                         batch_size=FLAGS.batch_size,
                                         max_degree=FLAGS.max_degree,
                                         num_neg_sample=FLAGS.neg_sample_size)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info    = tf.Variable(adj_info_ph, trainable=False, name='adj_info')

    # Create Model
    if FLAGS.model == 'graphsage_mean':
        sampler     = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        model       = SampleAndAggregate(placeholders,
                                         feat_data,
                                         adj_info,
                                         minibatch.deg,
                                         layer_infos=layer_infos,
                                         model_size=FLAGS.model_size,
                                         identity_dim=FLAGS.identity_dim,
                                         logging=True)
    elif FLAGS.model == 'gcn':
        sampler     = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]
        model       = SampleAndAggregate(placeholders,
                                         feat_data, 
                                         adj_info, 
                                         minibatch.deg,
                                         layer_infos=layer_infos,
                                         aggregator_type="gcn",
                                         model_size=FLAGS.model_size,
                                         identity_dim=FLAGS.identity_dim,
                                         concat=False,
                                         logging=True)
    else:
        raise Exception('Error: model name unrecognized.')

    # Configuration
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement     = True

    # Initialize Session
    sess           = tf.Session(config=config)
    merged         = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)

    # Initialize Variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

    # Train Model
    ## records
    train_shadow_mrr = None
    shadow_mrr       = None
    total_steps      = 0
    avg_time         = 0.
    epoch_val_costs  = []
    ## adjacency list
    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info   = tf.assign(adj_info, minibatch.test_adj)
    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()
        iter = 0
        print ('Epoch: %04d' % (epoch+1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            # construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # training step
            outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all,
                             model.mrr, model.outputs1],
                            feed_dict=feed_dict)
            train_cost = outs[2]
            train_mrr  = outs[5]
            if train_shadow_mrr is None:
                train_shadow_mrr = train_mrr
            else:
                train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)

            if iter % FLAGS.validate_iter == 0:
                # validation
                sess.run(val_adj_info.op)
                val_cost, ranks, val_mrr, duration = evaluate(sess, 
                                                              model, 
                                                              minibatch, 
                                                              size=FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost
            if shadow_mrr is None:
                shadow_mrr = val_mrr
            else:
                shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)

            if total_steps % FLAGS.print_every == 0:
                print ("Iter: %04d" % iter,
                       "train_loss: {:.5f}".format(train_cost),
                       "train_mrr: {:.5f}".format(train_mrr),
                       "train_mrr_ema: {:.5f}".format(train_shadow_mrr),    # exponential moving average
                       "val_loss: {:.5f}".format(val_cost),
                       "val_mrr: {:.5f}".format(val_mrr),
                       "val_mrr_ema: {:.5f}".format(shadow_mrr),            # exponential moving average
                       "time: {:.5f}".format(avg_time))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break
    print ("Optimization Finished!")

    if FLAGS.save_embeddings:
        sess.run(val_adj_info.op)
        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir())
        print ("Successful Saved Embedding!")

def main(argv=None):
    run_cora()

if __name__ == '__main__':
    tf.app.run()
