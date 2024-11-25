import random
import time
import numpy as np
import argparse
import torch
import networkx as nx
import pickle
from args import *
import pandas as pd

random.seed(args.seed)
np.random.seed(args.seed)

simulates = np.zeros((args.num_samples*args.length, args.nodes))

# get the neighbors of each node
def get_neighbors(adj):
    neighbors = {}
    for i in range(adj.shape[0]):
        neighbor = []
        for j in range(adj.shape[0]):
            if adj[i][j] == 1:
                neighbor.append(j)
        neighbors[i] = neighbor
    return neighbors

# init node data randomly
def init_node(dg):
    for i in range(dg.number_of_nodes()):
        dg.nodes[i]['state'] = 0 if random.random() < 0.95 else 1  # 5% chance to start as infected

# let the net spread by probability
def spread_prob(dg, neighbors, beta=0.15, gamma=0.08, step=100):
    node_num = dg.number_of_nodes()
    data = []
    origin_val = []
    for i in range(node_num):
        state = dg.nodes[i]['state']
        origin_val.append([1, 0, 0] if state == 0 else [0, 1, 0])
    data.append(origin_val)

    run = 0
    while run < step:
        run += 1
        next_val = []
        for i in range(node_num):
            if dg.nodes[i]['state'] == 1:  # Infected
                next_val.append([0, 1, 0])
                if random.random() < gamma:  # Recovery probability
                    dg.nodes[i]['state'] = 2
            elif dg.nodes[i]['state'] == 2:  # Recovered
                next_val.append([0, 0, 1])
            else:  # Susceptible
                infected_neighbors = sum(dg.nodes[neighbor]['state'] == 1 for neighbor in neighbors[i])
                if random.random() < 1 - (1 - beta) ** infected_neighbors:
                    next_val.append([0, 1, 0])
                    dg.nodes[i]['state'] = 1
                else:
                    next_val.append([1, 0, 0])
        data.append(next_val)

    return np.array(data)


import numpy as np


def convert_data_to_1d(data):
    # 初始化一个与 data 前两维相同的数组，但最后一维是 1
    converted_data = np.zeros((data.shape[0], data.shape[1]), dtype=int)

    # 检查 data 的第三维的第二个元素是否为 1，这表示状态是 (0, 1, 0)
    infected_indices = (data[:, :, 1] == 1)

    # 将感染状态的位置标记为 1
    converted_data[infected_indices] = 1

    return converted_data


def generate_network():
    if args.network == 'ER':
        print('ER')
        G = nx.erdos_renyi_graph(args.nodes, 0.01)
    elif args.network == 'WS':
        print('WS')
        G = nx.watts_strogatz_graph(args.nodes, 2, 0.03)
    elif args.network == 'BA':
        print('BA')
        G = nx.barabasi_albert_graph(args.nodes, 1)
    elif args.network == 'Karate':
        df = pd.read_csv('../dataset/Karate.csv')
        G = nx.from_pandas_edgelist(df, 'source', 'target')
    elif args.network == 'US':
        df = pd.read_csv('../dataset/USAir97.csv')
        G = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.Graph())
        # 重新标记节点
        mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    elif args.network == 'Dolphins':
        df = pd.read_csv('../dataset/Dolphins.csv')
        G = nx.from_pandas_edgelist(df, 'source', 'target')
        mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    else:
        print('error')
        exit(0)
    return G
# Generate data for multiple experiments at once
for exp_id in range(1, 2):
    dg = generate_network()
    edges = nx.adjacency_matrix(dg).toarray()
    neighbors = get_neighbors(edges)

    print('Simulating time series...')
    for i in range(args.num_samples):
        init_node(dg)
        data = spread_prob(dg, neighbors, step=args.length-1)
        converted_data = convert_data_to_1d(data)
        simulates[i*args.length:(i+1)*args.length, :] = converted_data
    print('Simulation finished!')

    print(simulates.shape)
    print(str(np.sum(edges)))
    all_data = simulates
    print(all_data.shape)

    # save the data
    series_path = f'../data/SIR_{args.network}_{args.parameter}_{args.nodes}_id{exp_id}_data.pickle'
    adj_path = f'../data/SIR_{args.network}_{args.parameter}_{args.nodes}_id{exp_id}_adj.pickle'
    with open(series_path, 'wb') as f:
        pickle.dump(all_data, f)

    with open(adj_path, 'wb') as f:
        pickle.dump(edges, f)
