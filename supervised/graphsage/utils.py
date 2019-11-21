import random
import numpy as np
from collections import defaultdict

DATA_PATH='/demo_graphsage/cora/'

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
    return feat_data, labels, adj_lists

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
