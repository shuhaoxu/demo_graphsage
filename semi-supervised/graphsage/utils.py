import os
import time
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict

flags = tf.app.flags
FLAGS = flags.FLAGS

DATA_PATH = '/demo_graphsage/cora/'

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(DATA_PATH+"cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = map(float, info[1:-1])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(DATA_PATH+"cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    node_map = {int(k):int(v) for k,v in node_map.items()}
    return feat_data, labels, adj_lists, node_map

def split_data(labels):
    idxs      = np.arange(2708)
    labels    = labels.ravel()
    label_map = {}
    for i,l in enumerate(labels):
        if l not in label_map:
            label_map[l] = []
        label_map[l].append(i)
    train = []
    test  = []
    val   = []
    for _,v in label_map.items():
        train += v[:20]
    count = 0
    random.shuffle(idxs)
    for idx in idxs:
        if idx not in train:
            if count < 1000:
                test.append(idx)
                count += 1
            else:
                val.append(idx)
    return train, test, val

def log_dir():
    log_dir = FLAGS.base_log_dir + "/unsup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}".format(model=FLAGS.model,
                                                            model_size=FLAGS.model_size,
                                                            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def evaluate(sess, model, minibatch_iter, size=None):
    t_test        = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val      = sess.run([model.loss, model.ranks, model.mrr],
                             feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time()-t_test)

def incremental_evaluate(sess, model, minibatch_iter, size):
    t_test     = t.time()
    finished   = False
    val_losses = []
    val_mrrs   = []
    iter_num   = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([sess.loss, sess.ranks, sess.mrr],
                            feed_dict=feed_dict_val)
        val_losses.append(outs_val[0])
        val_mrrs.append(outs_val[2])
    return np.mean(val_losses), np.mean(val_mrrs), (time.time()-t_test)

def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    val_embeddings = []
    finished       = False
    seen           = set([])
    nodes          = []
    iter_num       = 0
    name           = "val"
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1], 
                            feed_dict=feed_dict_val)
        # only save for embeds1 beacause of planetoid
        for i,edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i,:])
                nodes.append(edge[0])
                seen.add(edge[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)
    np.save(out_dir + name + mod + ".npy", val_embeddings)
    with open(out_dir+name+mod+'.txt', 'w') as fp:
        fp.write("\n".join(map(str,nodes)))

def construct_placeholders():
    '''
    Define placeholders

    Return: placeholders
        - batch1: target training node of each batch
        - batch2: node share edges with target node of each batch
        - neg_samples: negative sampled nodes of target node of each batch
        - dropout: dropout rate of network layer
        - batch_size: number of nodes of each batch
    '''
    placeholders = {'batch1'     : tf.placeholder(tf.int32, shape=(None), name='batch1'),
                    'batch2'     : tf.placeholder(tf.int32, shape=(None), name='batch2'),
                    'neg_samples': tf.placeholder(tf.int32, shape=(None), name='neg_sample_size'),
                    'dropout'    : tf.placeholder_with_default(0., shape=(), name='dropout'),
                    'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
                   }
    return placeholders
