import gym
import numpy as np
import argparse
import utils
from tqdm import tqdm
import os

"""
This sample code is used to run a random agent in the OfficeWorldDoorsTask1
environment then save all trajectories and traces which have the information of
labels.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=100, type=int,
                        help='Seed of numpy random functions. Default=100' )
parser.add_argument("--num_episodes", default=1000, type=int,
                        help='Number of Episodes. Default=1000')
parser.add_argument("--episode_len", default=1000, type=int,
                        help='Maximum Steps each episode can last. Default=1000')
parser.add_argument("--filename", default='figureout', type=str,
                        help='filename and/or directory of the file '
                        +'if is figureout'
                        +'selected filename is created according'
                        +' to the other parameters. Default=figureout')
parser.add_argument("--environment", default='Task1', type=str,
                        help='Select environment, viable options: Task1, Task2')
parser.add_argument("--N_pos", default=10, type=int,
                        help='Number of positive examples to sample. Default=10')
parser.add_argument("--N_neg", default=10, type=int,
                        help='Number of negative examples to sample. Default=10')

args = parser.parse_args()
SEED = args.seed
NUM_EPISODES = args.num_episodes
NUM_STEPS = args.episode_len
filename = args.filename
ENV_NAME = args.environment
N_pos = args.N_pos
N_neg = args.N_neg

np.random.seed(SEED)

env = gym.make("gym_LTL_RL:OfficeWorldDoors%s-v0" %(ENV_NAME))
env.action_space.seed(SEED)

pos_traces = []
neg_traces = []
predList = ['a', 'b', 'c', 'd']

# Run environment by randomly picking actions and save traces:
for i in tqdm(range(NUM_EPISODES)):
    env.reset()
    current_trace = []
    current_trace.append(env.get_observations())
    done = False
    reward = 0
    for j in range(NUM_STEPS):
        action = env.action_space.sample()
        next_state, reward, done, observations = env.step(action)
        current_trace.append(observations)
        if done or j==NUM_STEPS-1:
            if done:
                pos_traces.append(current_trace)
            else:
                neg_traces.append(current_trace)
            break

# Compress traces:
comp_neg_traces = utils.compress_list_traces(neg_traces)
comp_pos_traces = utils.compress_list_traces(pos_traces)

print('Number of Positive Traces : ', end='')
print(len(pos_traces))
print('Number of Negative Traces : ', end='')
print(len(neg_traces))

# Turn Compressed traces into trace file which can be read by flie:
# Random Pick N_pos number of positive traces and N_neg number of negative traces:

total_trace = utils.sample_traces(pos_traces=comp_pos_traces,
    neg_traces=comp_neg_traces, N_pos=N_pos, N_neg=N_neg)

if filename == 'figureout':
    filename = 'traces/'
    filename += '%s' %(ENV_NAME)
    filename += '_step%d' %(NUM_STEPS)
    filename += '_episode%d' %(NUM_EPISODES)
    filename += '_seed%d' %(SEED)
    filename += '_Npos%d' %(N_pos)
    filename += '_Nneg%d' %(N_neg)

utils.comptrace2flie(total_trace, predList, filename=filename)
