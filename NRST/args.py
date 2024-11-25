import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='voter',
                    help='What simulation to generate.')
parser.add_argument('--num-samples', type=int, default=200,
                    help='Number of training simulations to generate.')

parser.add_argument('--length', type=int, default=50,
                    help='Length of trajectory.')
parser.add_argument('--parameter', type=str, default='WS',
                    help='Length of trajectory.')
parser.add_argument('--nodes', type=int, default=1000,
                    help='Number of balls in the simulation.')
parser.add_argument('--network', type=str, default='WS' ,help='network of the simulation')
parser.add_argument('--model', type=str, default='SIR')
parser.add_argument('--seed', type=int, default=25,
                    help='Random seed.')
parser.add_argument('--sys', type=str, default='voter', help='simulated system to model,spring or cmn')
parser.add_argument('--dim', type=int, default=2, help='# information dimension of each node spring:4 cmn:1 ')
parser.add_argument('--exp_id', type=int, default=1, help='experiment_id, default=1')
parser.add_argument('--device_id', type=int, default=0, help='Gpu_id, default=5')

datasets = ['jazz_SIR_198', 'jazz_SI_198', 'cora_ml_SIR_2810', 'cora_ml_SI_2810', 'power_grid_SIR', 'power_grid_SI',
            'karate_SIR_34', 'karate_SI_34', 'netscience_SIR_1565', 'netscience_SI_1565']
parser.add_argument("-d", "--dataset", default="cora_ml_SI", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
