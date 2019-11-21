import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

import time
import random
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append('/demo_graphsage/supervised/graphsage')
from utils import load_cora, split_data
from encoders import Encoder
from aggregators import MeanAggregator
from supervised_graphsage import SupervisedGraphSage

def run_cora():
    np.random.seed(1)
    random.seed(1)

    # load data
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    train, test, val = split_data(labels)

    # construct model
    ## layer1 : Embedding layer
    features  = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    #features.cuda()
    ## layer2 : Sample and Aggregate 1433->128
    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    ## layer3 : Sample and Aggregate 128->128
    agg2 = MeanAggregator(lambda nodes:enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes:enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    ## layer4 : Classification layer
    enc1.num_samples = 5
    enc2.num_samples = 5
    graphsage = SupervisedGraphSage(7, enc2)

    # optimizer
    optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, graphsage.parameters()),
                                lr=0.7)

    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                              Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        if batch % 10 == 0:
            print batch, float(loss.data)
    print 'Finished training.'
    print '******************************'

    test_output = graphsage.forward(test)
    test_onehot = test_output.data.numpy()
    test_labels = labels[test]
    test_preds  = np.argmax(test_onehot, axis=1)
    print 'Test Accuracy:', accuracy_score(test_labels, test_preds)
    print 'Average batch time: ', np.mean(times)

if __name__ == '__main__':
    run_cora()
