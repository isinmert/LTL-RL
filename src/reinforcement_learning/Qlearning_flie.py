import gym
import numpy as np
from utils import ltl2dot, dot2DFA
from utils import comptrace2flie, compress_list_traces, compress_trace_spaced
from utils import use_flie, ltl2dot
from utils import flie2genLTL, dot2DFA
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=100, type=int,
    help="Seed random number generator of environment and policy. Default=100")
parser.add_argument("--num_episodes", default=1000, type=int,
    help="Number of Episodes. Default=1000")
parser.add_argument("--episode_len", default=300, type=int,
    help="Maximum steps an episode can have. Default=300")
parser.add_argument("--learning_rate", default=0.99, type=float,
    help="Learning Rate of Q-learning algorithm. Default=0.99")
parser.add_argument("--epsilon", default=0.1, type=float,
    help="epsilon parameter of Q-learning. Default=0.1")
parser.add_argument("--gamma", default=0.99, type=float,
    help="Discount factor of RL task. Default=0.99")
parser.add_argument("--pos_thresh", default=10, type=int,
    help="maximum number of pos_traces stored. Default=10")
parser.add_argument("-print_traces", action="store_true")
parser.add_argument("-show_reward", action="store_true")
parser.add_argument("-tqdm_off", action="store_true")
parser.add_argument("--formula_init", default="Fa", type=str,
    help="Initial formula for learning. Default=Fa")
parser.add_argument("--result_dir", default="results", type=str,
    help="directory of results file. default=results")
parser.add_argument("-all_map", action="store_true")
parser.add_argument("--environment", default="OfficeWorldDoorsTask2", type=str,
    help="Environment, default=OfficeWorldDoorsTask2,"+
    " other options: OfficeWorldDoorsTask1, OfficeWorldBigTask1")

parser.set_defaults(print_traces=False, show_reward=False, tqdm_off=False,
    all_map=False)
args = parser.parse_args()

SEED = args.seed
NUM_EPISODES = args.num_episodes
MAX_EPISODE_LEN = args.episode_len
ALPHA = args.learning_rate      # Learning Rate
EPS = args.epsilon              # Epsilon Greedy Parameter
GAMMA = args.gamma              # Discount Factor
RESULTS_DIR = args.result_dir   # directory of results file
ALL_MAP = args.all_map          # enable initial state sampled from all map
ENV = args.environment          # Name of the environment and task
if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)


env = gym.make("gym_LTL_RL:"+ ENV +"-v0")
env.set_seed(SEED)
env.action_space.np_random.seed(SEED)
np.random.seed(SEED)

MAP_HEIGHT = env.height
MAP_WIDTH  = env.width
S_card = env._get_num_states()
NUM_ACTIONS = env.action_space.n

pos_traces_threshold = args.pos_thresh
print_all_traces = args.print_traces
show_episode_reward = args.show_reward

task_spec = args.formula_init
# task_spec = '((Fa)U(b))&(G(~c)) ' # This is the true spec


time_init = time.time()
automata_path, predicate_list = ltl2dot(task_spec)
predicate_list = ['a', 'b', 'c', 'd']
automaton0 = dot2DFA(automata_path, predicate_list)
if os.path.isfile(automata_path):
    os.remove(automata_path)

pos_traces = []
neg_traces = []
pos_traces_imp = []
neg_traces_imp = []
all_traces = []
all_traces.append(pos_traces)
all_traces.append(neg_traces)
one_pos_trace = [set(), set('b'),set(), set('a'), set()]
# one_neg_trace = [set(), set('b'),set()]
pos_traces_imp.append(one_pos_trace)
# neg_traces_imp.append(one_neg_trace)
print('Initial Strategy Formula: ', end='')
print(task_spec)

fileformula = open("%s/inferred_formulas.txt"%(RESULTS_DIR), "w+")
fileformula.write(task_spec+"\n")
filerewards = open("%s/rewards_eplen.txt"%(RESULTS_DIR), "w+")

automaton_changed = False
Q_values = {}
automatonk = automaton0
for states_automaton in list(automatonk.get_states()):
    Q_values[states_automaton] = np.zeros([S_card, NUM_ACTIONS])

print('Automaton States: ', end='')
print(automatonk.get_states())
print('Initial State: ', end='')
print(automatonk.get_init_state())
print('Accepting State: ', end='')
print(automatonk.get_acc_state())
edges_automatonk = automatonk.edges
print('Transitions : ')
for key, value in sorted(edges_automatonk.items()):
    print(key + ' : ', end='')
    print(value)

sum_reward_list = []
count_update = 0
for i in tqdm(range(NUM_EPISODES), disable=args.tqdm_off):
    env.reset(all_map=ALL_MAP)
    is_done_env = False

    if automaton_changed:
        automaton_changed = False
        Q_values = {}
        for states_automaton in list(automatonk.get_states()):
            Q_values[states_automaton] = np.zeros([S_card, NUM_ACTIONS])

        print('---FORMULA CHANGED---')
        print('New Strategy Formula: ', end='')
        print(new_formula_gen)
        print('------')
        print('New Automaton')
        print('Automaton States: ', end='')
        print(automatonk.get_states())
        print('Initial State: ', end='')
        print(automatonk.get_init_state())
        print('Accepting State: ', end='')
        print(automatonk.get_acc_state())
        edges_automatonk = automatonk.edges
        print('Transitions : ')
        for key, value in sorted(edges_automatonk.items()):
            print(key + ' : ', end='')
            print(value)

        print('List of Labels : ',end='')
        print(automatonk.labels)


    state_automaton = automatonk.reset()
    is_done_automata = False
    counter_example_found = False
    state_agent = env._get_state()
    trace_run = []
    sum_rewards = 0
    for j in range(MAX_EPISODE_LEN):
        # Pick action according to epsilon-greedy rule
        max_Q = np.max(Q_values[state_automaton][state_agent])
        list_possible_actions = []
        for actions_iter in range(NUM_ACTIONS):
            if Q_values[state_automaton][state_agent][actions_iter] >= max_Q:
                list_possible_actions.append(actions_iter)
        if np.random.random() > EPS:
            action = np.random.choice(list_possible_actions)
        else:
            action = env.action_space.sample()

        # Execute selected Action and update automaton based on the
        # observed label
        state_agent_prev = state_agent
        state_automaton_prev = state_automaton
        state_agent, reward, is_done_env, labels = env.step(action)
        state_automaton, is_done_automata = automatonk.step_automaton(list(labels))
        sum_rewards += reward

        # Update Q-values:
        Qk = Q_values[state_automaton_prev][state_agent_prev][action]
        Q_next_max = np.max(Q_values[state_automaton][state_agent])
        Qkp1 = (1 - ALPHA)*(Qk) + ALPHA * (reward + GAMMA * Q_next_max)
        Q_values[state_automaton_prev][state_agent_prev][action] = Qkp1

        # Append set of observations to trace
        trace_run.append(labels)

        # Check if DFA and Environment match with each other
        if is_done_automata and is_done_env:
            # Empty Set is added whenever a trace is finalized
            trace_run.append(set())
            pos_traces.append(trace_run)
            break
        elif not is_done_automata and is_done_env:
            trace_run.append(set())
            pos_traces.append(trace_run)
            pos_traces_imp.append(trace_run)
            counter_example_found = True
            break
        elif is_done_automata and not is_done_env:
            trace_run.append(set())
            neg_traces.append(trace_run)
            neg_traces_imp.append(trace_run)
            counter_example_found = True
            break
        else:
            pass

    if j == MAX_EPISODE_LEN-1:
        trace_run.append(set())
        neg_traces.append(trace_run)

    sum_reward_list.append(sum_rewards)
    filerewards.write("reward : %.2f  ; episode length:%d\n"%(sum_rewards, j+1))

    if show_episode_reward:
        if is_done_env:
            print('SUCCESS', end='    ')
        else:
            print('FAILURE', end='    ')
        print('Episode %d is done'%(i+1), end='     ')
        print('Reward is:%.2f'%(sum_rewards), end='    ')
        print('Episode Length:%d'%(j+1))
        print(compress_trace_spaced(trace_run))
        print('')
        print('')

    # Update Specification and Automaton if NECESSARY
    if (len(pos_traces_imp) > 0 and len(neg_traces_imp) > 0
            and counter_example_found):
        counter_example_found = False
        automaton_changed= True
        count_update += 1

        print('')
        print('')
        print('Founded counter-example trace for previous formula : ')
        print(compress_trace_spaced(trace_run))

        pos_traces_passed = compress_list_traces(pos_traces_imp, spaced=True)
        neg_traces_passed = compress_list_traces(neg_traces_imp, spaced=True)
        print('Number of pos traces passed : %d'%(len(pos_traces_passed)))
        print('Number of neg traces passed : %d'%(len(neg_traces_passed)))
        all_traces = []
        all_traces.append(pos_traces_passed)
        all_traces.append(neg_traces_passed)

        if print_all_traces:
            print("Positive Traces:")
            for traces_iterable in pos_traces_passed:
                print(traces_iterable)

            print('Negative Traces:')
            for traces_iterable in neg_traces_passed:
                print(traces_iterable)

        trace_dir = comptrace2flie(all_traces,
            predicate_list, filename='%s/traces/trace%d'%(RESULTS_DIR, count_update),
                until=True)

        output_flie = use_flie(trace_dir)
        new_formula_flie = output_flie[0]
        new_formula_gen = flie2genLTL(new_formula_flie, predicate_list)
        fileformula.write(new_formula_gen+"\n")
        # print(new_formula_flie)

        automata_path, pred_list_use = ltl2dot(new_formula_gen,
            operator_list=None, filename_dot='denemedot1')
        automatonk = dot2DFA(automata_path, pred_list_use)
        os.remove(automata_path)

    pass

time_elapsed = time.time() - time_init

filetime = open("%s/total_time.txt"%(RESULTS_DIR), "w+")
filetime.write("%.3f"%(time_elapsed))

fileformula.close()
filerewards.close()
np.save('%s/rewards_episode'%(RESULTS_DIR), sum_reward_list)
np.save('%s/finalQvalues'%(RESULTS_DIR), Q_values)

plt.plot(sum_reward_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('%s/rewardvsepisode'%(RESULTS_DIR))
