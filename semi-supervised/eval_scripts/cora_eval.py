#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np

from networkx.readwrite import json_graph
from argparse import ArgumentParser
from hparams import Hparams

def classifier_def(clf_type):
    if clf_type == "SGD":
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss='log', n_jobs=10)
    elif clf_type == "OVR":
        from sklearn.svm import SVC
        from sklearn.multiclass import OneVsRestClassifier
        clf = OneVsRestClassifier(esimator=SVC())
    elif clf_type == "LR":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
    else:
        raise "Wrong classifier type."
    return clf

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    # Random Baseline
    dummy = DummyClassifier()
    dummy.fit(train_embeds, train_labels.ravel())
    # Classifier
    log = classifier_def(clf_type="SGD")
    log.fit(train_embeds, train_labels.ravel())
    # Micro-F1
    print ("F1 score: ", f1_score(test_labels.ravel(), log.predict(test_embeds), average='micro'))
    print ("Random baseline f1 score: ", f1_score(test_labels.ravel(), dummy.predict(test_embeds), average='micro'))

    # Accuracy
    y_pred = log.predict(test_embeds)
    return y_pred

def result_analysis(y, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_true  = y.ravel()
    cnf_mat = confusion_matrix(y_true, y_pred)
    acc     = accuracy_score(y_true, y_pred)
    print ("Confussion Matrix:\n", cnf_mat)
    print ("accuracy: ", acc)

if __name__ == "__main__":
    # Get Params.
    parser = Hparams()
    args   = parser.parse_args()
    dataset_dir = args.dataset_dir
    data_dir    = args.embed_dir
    data_type   = args.embed_type
    setting     = args.setting
    model_type  = args.model_type

    # Load Data
    print ("Loading data...")
    ## graph
    G = json_graph.node_link_graph(json.load(open(dataset_dir + "cora-G.json")))
    ## indexes
    train_ids = [n for n in G.nodes() if G.node[n]['train']]
    test_ids  = [n for n in G.nodes() if G.node[n][setting]]
    ## label classes
    label_map = json.load(open(dataset_dir+'cora-label_map.json'))
    label_map = {int(k):v for k,v in label_map.items()}
    train_labels = [label_map[idx] for idx in train_ids]
    test_labels  = [label_map[idx] for idx in test_ids]
    c = len(np.unique(train_labels))
    train_one_hot = np.eye(c, dtype=int)[train_labels]
    test_one_hot  = np.eye(c, dtype=int)[test_labels]
    train_labels  = np.asarray(train_labels)
    test_labels   = np.asarray(test_labels)

    # Load embeddings
    if data_type == "feat":
        print ("Using only features..")
        feats = np.load(dataset_dir + "cora-feats.npy")
        feat_id_map = json.load(open(dataset_dir + "cora-id_map.json"))
        feat_id_map = {int(id):val for id,val in feat_id_map.iteritems()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]]
        test_feats  = feats[[feat_id_map[id] for id in test_ids]]
        print ("Running regression..")
        run_regression(train_feats, train_labels, test_feats, test_labels)
    elif data_type == "embed":
        embeds = np.load(data_dir + model_type + '_small_0.000010val.npy')
        id_map = {}
        with open(data_dir+model_type+"_small_0.000010val.txt") as fp:
            for i, line in enumerate(fp):
                id_map[int(line.strip())] = i
        train_embeds = embeds[[id_map[id] for id in train_ids]]
        test_embeds  = embeds[[id_map[id] for id in test_ids]]
        print ("Running regression..")
        test_pred = run_regression(train_embeds, train_labels, test_embeds, test_labels)
        result_analysis(test_labels, test_pred)
