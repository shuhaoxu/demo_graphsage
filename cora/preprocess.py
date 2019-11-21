#!/usr/bin/python

import os
import sys
import json
import numpy as np
import networkx as nx
from collections import defaultdict
from networkx.readwrite import json_graph

os.chdir('/demo_graphsage/')
sys.path.append('/demo_graphsage/semi-supervised/graphsage')
from utils import load_cora, split_data

if __name__ == '__main__':
    # Load Data
    feat_data, labels, adj_lists, id2idx = load_cora()
    train_idx, test_idx, val_idx = split_data(labels)
    G = nx.read_edgelist('cora/cora.cites', nodetype=int)

    # Get True Node ID
    idx2id   = {v:int(k) for k,v in id2idx.items()}
    train_id = [idx2id[i] for i in train_idx]
    test_id  = [idx2id[i] for i in test_idx]
    val_id   = [idx2id[i] for i in val_idx]

    # Add Attribute to G: node attribute + edge attribute
    ## node attribute: train / val / test
    for n in train_id:
        G.node[n]['train'] = True
        G.node[n]['test']  = False
        G.node[n]['val']   = False
    for n in test_id:
        G.node[n]['train'] = False
        G.node[n]['test']  = True
        G.node[n]['val']   = False
    for n in val_id:
        G.node[n]['train'] = False
        G.node[n]['test']  = False
        G.node[n]['val']   = True
    ## edge attribute: train_removed
    ## make sure the graph has edge train_removed annotations
    for e in G.edges():
        if (G.node[e[0]]['val'] or G.node[e[0]]['test'] or 
            G.node[e[1]]['val'] or G.node[e[1]]['test']):
            G[e[0]][e[1]]['train_removed'] = True
        else:
            G[e[0]][e[1]]['train_removed'] = False

    # Label Map
    labels    = labels.ravel()
    label_map = defaultdict()
    for id in train_id+test_id+val_id:
        idx = id2idx[id]
        l   = labels[idx]
        label_map[id] = l

    # Write to File
    ### G
    #data = json_graph.node_link_data(G)
    #G_out_file = 'cora/cora-G.json'
    #with open(G_out_file, 'w') as f:
    #    f.write(json.dumps(data))

    ### label map
    #label_map_out = 'cora/cora-label_map.json'
    #with open(label_map_out, 'w') as f:
    #    f.write(json.dumps(label_map))
