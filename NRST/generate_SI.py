import random
import time
import numpy as np
import argparse
import torch
import networkx as nx
import pickle
from args import *
import pandas as pd
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
def spread_prob(dg, neighbors, beta=0.3, step=100):
    node_num = dg.number_of_nodes()
    # data to be returned
    data = []
    # add initial value to data
    origin_val = []
    for i in range(node_num):
        origin_val.append(dg.nodes[i]['state'])
    data.append(origin_val)

    run = 0
    while run < step:
        run += 1
        next_val = []

        for i in range(node_num):
            if dg.nodes[i]['state'] == 1:
                next_val.append(1)
            else:
                infected_neighbors = sum(dg.nodes[neighbor]['state'] for neighbor in neighbors[i])
                if random.random() < 1 - (1 - beta) ** infected_neighbors:
                    next_val.append(1)
                else:
                    next_val.append(0)

        for i in range(node_num):
            dg.nodes[i]['state'] = next_val[i]

        data.append(next_val)
    return np.array(data)

def generate_network():
    if args.network == 'ER':
        print('ER')
        G = nx.erdos_renyi_graph(args.nodes, 0.05)
    elif args.network == 'WS':
        print('WS')
        G = nx.watts_strogatz_graph(args.nodes, 2, 0.1)
    elif args.network == 'BA':
        print('BA')
        G = nx.barabasi_albert_graph(args.nodes, 2)
    elif args.network == 'Karate':
        df = pd.read_csv('../dataset/Karate.csv')
        G = nx.from_pandas_edgelist(df, 'source', 'target')
    elif args.network == 'Dolphins':
        df = pd.read_csv('../dataset/Dolphins.csv')
        G = nx.from_pandas_edgelist(df, 'source', 'target')
        mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    elif args.network == 'Jazz':
        df = pd.read_csv('../dataset/Jazz.csv')
        G = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.Graph())
        # 重新标记节点
        mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    elif args.network == 'Dorm':
        print('DORM')
        G = nx.DiGraph()
        node_num = 217
        for i in range(node_num):
            G.add_node(i, value=random.randint(0, 1))
            path = '../dataset/out.moreno_oz_oz'
            # 读取文件
            f = open(path)
            flag = 0
            for line in f:
                if flag < 2:
                    flag = flag + 1
                    continue
                first = int(line.split(' ')[0]) - 1
                second = int(line.split(' ')[1]) - 1
                G.add_edge(first, second)
            # print(len(G.edges()))
    elif args.network == 'Email':
        print('EMAIL')
        G=nx.Graph()
        node_num = 1133
        for i in range(node_num):
            G.add_node(i, value=random.randint(0, 1))
        path = '../dataset/out.arenas-email'
        # 读取文件
        f = open(path)
        flag = 0
        for line in f:
            if flag < 2:
                flag = flag + 1
                continue
            first = int(line.split(' ')[0]) - 1
            second = int(line.split(' ')[1]) - 1
            G.add_edge(first, second)
    elif args.network == 'ROAD':
        print('ROAD')
        G = nx.Graph()
        node_num = 1174
        for i in range(node_num):
            G.add_node(i, value=random.randint(0, 1))
        path = '../dataset/out.subelj_euroroad_euroroad'
        # 读取文件
        f = open(path)
        flag = 0
        for line in f:
            if flag < 2:
                flag = flag + 1
                continue
            first = int(line.split(' ')[0]) - 1
            second = int(line.split(' ')[1]) - 1
            G.add_edge(first, second)
        # print(len(G.edges()))
    elif args.network == 'ROAD2642':
        df = pd.read_csv('../dataset/road-minnesota.csv')
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
        simulates[i*args.length:(i+1)*args.length, :] = data
    print('Simulation finished!')

    print(simulates.shape)
    print(str(np.sum(edges)))
    all_data = simulates[:, :, np.newaxis]
    print(all_data.shape)

    # save the data
    series_path = f'../data/SI_{args.network}_{args.parameter}_{args.nodes}_id{exp_id}_data.pickle'
    adj_path = f'../data/SI_{args.network}_{args.parameter}_{args.nodes}_id{exp_id}_adj.pickle'

    with open(series_path, 'wb') as f:
        pickle.dump(all_data, f)

    with open(adj_path, 'wb') as f:
        pickle.dump(edges, f)
