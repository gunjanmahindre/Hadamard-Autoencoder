# Hadamard-Autoencoder

This research project was documented for Local Computer Networks (LCN) 2020.


## Problem statement:
Predicting missing network measurements in ultra-sparsely sampled social networks.


## Approach:
* The Hadamard Autoencoder (HA) is pretrained on synthetic Powerlaw networks. This pretrained neural network is then used to predict missing distance measurements in Facebook network.
* We deal with dop-distance matrices for each network.
* The model is tested for a small fraction (0.8, 0.6, 0.2, 0.1, 0.01, 0.005, 0.001, and 0.00001) of entries are sampled from the training networks to check the performance of the HA model with sparse training data.
* This pretrained model is tested on the same fraction of sampled entries from Facebook test network. 
* Note that the HA model has not been trained on any sample from the Facebook test network, i.e., we do not fine tune the network.


## Code:
The folder "autoencoder codes" provides all the python files used to run the HA model.

## Data used:
We simulate the Powerlaw networks using following functions in python's networkx module (https://networkx.github.io/):
* G1 = nx.barabasi_albert_graph(744, 2, seed=None)
* G2 = nx.barabasi_albert_graph(744, 10, seed=None)
* G3 = nx.barabasi_albert_graph(744, 50, seed=None)
* G4 = nx.powerlaw_cluster_graph(744, 2, 0.1, seed=None)
* G5 = nx.powerlaw_cluster_graph(744, 50, 0.5, seed=None)
* G6 = nx.powerlaw_cluster_graph(744, 200, 0.9, seed=None)

We use a 744 node Facebook network that is shared in this repository titled "Facebook.txt".
It has been downloaded from SNAP: http://snap.stanford.edu/data/index.html
