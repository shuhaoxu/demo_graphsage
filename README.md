# Demo GraphSAGE
***Demo*** graphsage model running on **cora** dataset. Include both `supervised` && `semi-supervised` scripts.

## Environment
We build a *mirror* for demo run, the sample is as follows. Feel free to set `mount_path`, we fix it to `/demo-graphsage`.    
 
	step 1: docker pull shuhaoxu/gnn_mirrors:GraphSAGE_gpu
	step 2: nvidia-docker run -dit -v host_path:mount_path image_id
	step 3: docker exec -it container_id bash

## Dataset
We use **cora** dataset for demo run. Cora is a paper citation data, which includes 2708 nodes and 5278 edges. And its nodes are categorized
into seven classes.  
 `Case Based`
 `Genetic Algorithms`
 `Neural Networks`
 `Probabilistic Methods`
 `Reinforcement Learning`
 `Rule Learning`
 `Theory`. 
 
For model training, we split dataset into `train`,`val`,`test` three parts.
> `train` select 20 nodes from each node type  
> `test` randomly select 1000 nodes from rest data  
> `val` are rested nodes

## Model
We provide two types of GraphSAGE demo run, both scripts take [williamleif](https://github.com/williamleif) for reference. 
> [Supervised](https://github.com/williamleif/graphsage-simple) build GNN on [PyTorch](https://pytorch.org/)  
> [Semi-supervised](https://github.com/williamleif/GraphSAGE) build GNN on [TensorFlow](https://www.tensorflow.org/)

### Supervised GraphSAGE
**Supervised** GraphSAGE model demo, train a end-to-end graph neural network with 2-layer aggregator for feature encoding 
and 1-layer FC for classification.  

	python supervised_train.py
	
### Semi-supervised GraphSAGE
**Semi-supervised** GraphSAGE demo run, train a 2-layer aggregator with loss, namely semi-supervised
 computed according to func1 in [paper](https://arxiv.org/pdf/1706.02216.pdf) and get a node vector matrix. 
 
	python semi_supervised.py --model gcn --epochs 100
	python eval_scripts/cora_eval.py