#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from argparse import ArgumentParser

PATH = '/demo_graphsage/'

def Hparams():
    parser = ArgumentParser("Run evaluation on cora data.")

    parser.add_argument("--dataset_dir", default=PATH+"/cora/",
                        help="Path to directory containing the dataset.")
    parser.add_argument("--embed_dir", default=PATH+"/semi-supervised/unsup-cora/",
                        help="Path to directory containing the learned node embeddings.")
    parser.add_argument("--embed_type", default="embed",
                        help="embedding type of data.")
    #parser.add_argument("--setting", default="val",
    parser.add_argument("--setting", default="test",
                        help="Either val or test.")

    parser.add_argument("--model_type", default="gcn",
    #parser.add_argument("--model_type", default="graphsage_mean",
                        help="Model type for GraphSAGE, either graphsage_mean or gcn")

    return parser
