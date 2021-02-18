import numpy as np
import os
import sys
import spot
from ltlf2dfa.Translator import Translator
from ltlf2dfa.DotHandler import DotHandler
import subprocess
from tqdm import tqdm

def compress_trace(trace):
    """
    This function removes all the empty sets in the trace which is given as a
    list of sets which represents observations. Then it outputs a list with
    sets that are non-empty.
    """
    compressed_trace = []
    for observations in trace:
        if len(observations) > 0:
            compressed_trace.append(observations)
    return compressed_trace

def compress_trace_spaced(trace):
    """
    This function compresses a trace by removing consecutive empty sets and
    places one empty set instead of them. The output is a list of sets.
    """
    compressed_trace = []
    last_obs_empty = False

    for observations in trace:
        card_obs = len(observations)
        if card_obs > 0:
            compressed_trace.append(observations)
            last_obs_empty = False
        else:
            if not last_obs_empty:
                compressed_trace.append(observations)
            else:
                pass
            last_obs_empty = True

    return compressed_trace

def compress_trace_extra(trace, comp_param=2):
    """
    This function compresses traces in a way that repeating sequences are also
    removed and represented with one entity.
    """

    trace_half = compress_trace_spaced(trace)
    list_modified = True
    k = 0
    while list_modified:
        len_trace_half = len(trace_half)
        i = 0
        list_modified = False
        compressed_trace = []
        while(i <= len_trace_half-1):
            temp_list1 = trace_half[i:min(i+comp_param, len_trace_half)]
            temp_list2 = trace_half[min(len_trace_half,i+comp_param):min(len_trace_half,i+2*comp_param)]
            if temp_list1 == temp_list2 and i <= len_trace_half - 2*comp_param:
                list_modified = True
                compressed_trace += temp_list1
                i += 2*comp_param
                pass
            else:
                compressed_trace.append(temp_list1[0])
                i += 1
        k += 1
        trace_half = compressed_trace
    return compressed_trace

def compress_list_traces(traceList, spaced=True, extra=False):
    """
    This function compress a list of traces as compress_traces function
    Output is List of traces where each trace is compressed
    """
    compressed_traces = []
    for traces in traceList:
        if extra:
            comp_trace = compress_trace_extra(traces)
        elif spaced:
            comp_trace = compress_trace_spaced(traces)
        else:
            comp_trace = compress_trace(traces)
        compressed_traces.append(comp_trace)
    return compressed_traces

def comptrace2flie(traces, listPredicates, filename='Output', until=True):
    """
    This function takes the list of traces and list of Predicates and returns
    a .trace file which can be read by flie. List of traces is actually a list
    of 2 lists in which first list contains positive examples and the second
    contains negative examples.
    """

    filename_decomposed = filename.split('/')
    filedir = ''
    for i in range(len(filename_decomposed) - 1):
        filedir += filename_decomposed[i] + '/'

    if not os.path.isdir(filedir):
        os.system('mkdir -p %s' %(filedir))

    # First Create the output.trace file:
    f = open("%s" %(filename)  + ".trace", "+w")
    len_binary_vectors = len(listPredicates)

    pos_trace_list = traces[0]
    neg_trace_list = traces[1]

    for a_trace in pos_trace_list:
        trace_length = len(a_trace)
        for count_trace, observation_set in enumerate(a_trace, 1):
            for count_predicates, predicate in enumerate(listPredicates, 1):
                if predicate in observation_set:
                    pred_value = 1
                else:
                    pred_value = 0
                f.write("%d" %(pred_value))
                if count_predicates == len_binary_vectors and count_trace != trace_length:
                    f.write(";")
                elif count_predicates == len_binary_vectors and count_trace == trace_length:
                    f.write("::%d\n" %(count_trace-1))
                else:
                    f.write(",")

    f.write("---\n")
    for a_trace in neg_trace_list:
        trace_length = len(a_trace)
        for count_trace, observation_set in enumerate(a_trace, 1):
            for count_predicates, predicate in enumerate(listPredicates, 1):
                if predicate in observation_set:
                    pred_value = 1
                else:
                    pred_value = 0
                f.write("%d" %(pred_value))

                if count_predicates == len_binary_vectors and count_trace != trace_length:
                    f.write(";")
                elif count_predicates == len_binary_vectors and count_trace == trace_length:
                    f.write("::%d\n" %(count_trace-1))
                else:
                    f.write(",")
    f.write("---\n")
    if until:
        f.write("G,F,!,U,&,|,->,X")
    else:
        f.write("G,F,!,&,|,->,X")
    f.close()
    return os.path.abspath("%s" %(filename)  + ".trace")

def sample_traces_both(pos_traces, neg_traces, N_pos, N_neg=None):
    if N_neg == None:
        N_neg = N_pos

    if N_pos > len(pos_traces):
        N_pos = len(neg_traces)
    if N_pos == 1:
        selected_pos_traces = pos_traces
    elif N_pos == 0:
        selected_pos_traces = []
    else:
        selected_pos_traces = list(np.random.choice(pos_traces, replace=False, size=N_pos))

    if N_neg == 1:
        selected_neg_traces = neg_traces
    elif N_neg == 0:
        selected_neg_traces = []
    else:
        selected_neg_traces = list(np.random.choice(neg_traces, replace=False, size=N_neg))

    total_trace = []
    total_trace.append(selected_pos_traces)
    total_trace.append(selected_neg_traces)
    return total_trace

class DFAutomaton():
    """
    Deterministic Finite Automaton class which states are represented by a set
    labels with something and write the rest later
    """

    def __init__(self):
        self.states = set()
        self.edges = {}
        self.initial_state = None
        self.accept_state = None
        self.reject_state = None
        self.current_state = None
        self.labels = set()

    def add_state(self, state):
        """ Adds a state to the states and creates an entry in the set of edges
        that go from that state."""
        if state not in self.states:
            self.states.add(state)
            self.edges[state] = []

    def get_states(self):
        """Get States as a set"""
        return self.states

    def set_random_seed(self, seed):
        np.random.seed(seed)

    def gen_num_states(self):
        """Get Number of States"""
        return len(self.states)

    def add_edge(self, state_from, state_to, labels):
        """Add Edges and corresponding labels to the automata"""
        if state_from not in self.states:
            self.add_state(state_from)
        if state_to not in self.states:
            self.add_state(state_to)
        if type(labels) != list:
            print('Incorrect Format for labels, Use python list')
            return None

        self.edges[state_from].append((labels, state_to))
        for label in labels:
            if label.startswith('~'):
                label_to_add = label[1:]
            else:
                label_to_add = label
            self.labels.add(label_to_add)

    def set_init_state(self, init_state):
        """Set initial state, there can be only one initial state"""
        if init_state not in self.states:
            self.add_state(init_state)
        if self.initial_state is not None:
            print('There was another initial state but it is changed with %s' \
                %(init_state))
        self.initial_state = init_state

    def get_init_state(self):
        """Returns initial state"""
        return self.initial_state

    def set_acc_state(self, acc_state):
        """Sets an accepting state"""
        if acc_state not in self.states:
            self.add_state(acc_state)
        if self.accept_state is not None:
            print('There was another accepting state but it is changed with %s' \
                %(acc_state))
        self.accept_state = acc_state

    def get_acc_state(self):
        """Returns accepting state"""
        return self.accept_state

    def set_reject_state(self, reject_state):
        """Sets an rejecting state"""
        if reject_state not in self.states:
            self.add_state(reject_state)
        if self.reject_state is not None:
            print('There was another rejecting state but it is changed with %s' \
                %(reject_state))
        self.reject_state = reject_state

    def get_reject_state(self):
        """Returns rejecting state"""
        return self.reject_state

    def is_terminal_state(self, state):
        return (state == self.accept_state) or (state == self.reject_state)

    def reset(self):
        if self.initial_state is None:
            print('Initial does not exist, resetting is unsuccessful '
                + 'add initial states')
            return None
        if self.accept_state is None:
            print('Accepting States do not exist, set accept_state' +
                'resetting is still successful but there cannot be any' +
                ' accepted trace')
        self.current_state = self.initial_state

        return self.current_state

    def get_edges(self):
        return self.edges


    def get_next_states(self, labels, current_state=None):
        if current_state == None:
            current_state = self.current_state

        if current_state == None:
            print('Current state does not exist enter a state reset automata' +
            ' next state is not defined')
            next_states = None
            return next_states

        if type(labels) != list:
            print('Labels are not formatted correctly')
            next_states = None
            return next_states

        next_states = set([])

        for condition, possible_state in self.edges[current_state]:
            if self._is_condition_satisfied(condition, labels):
                next_states.add(possible_state)

        if len(next_states) == 0:
            next_states.add(current_state)

        return next_states

    def _is_condition_satisfied(self, condition, labels):
        """Returns true if a condition is satisfied by a set of observations. If a condition is empty, it is always
        true regardless of the observations."""
        if len(condition) == 0:  # empty conditions = unconditional transition (always taken)
            return True

        for literal in condition:
            if literal.startswith('~'):
                dummy = literal[1:]
                if dummy in labels:
                    return False
            else:
                dummy = literal
                if dummy not in labels:
                    return False
        return True

    def add_label(self, listoflabels):
        for labeltoadd in listoflabels:
            self.labels.add(labeltoadd)

    def step_automaton(self, labels):
        if self.current_state == None:
            print('Current State does not exist, Reset the Automaton.')
        else:
            next_state_set = self.get_next_states(labels=labels)

            self.current_state = np.random.choice(list(next_state_set))

        return self.current_state, self.is_terminal_state(self.current_state)

    def get_samples(self, num_traces, max_trace_length=100):

        pos_traces = []
        neg_traces = []

        for i in range(num_traces):
            self.reset()
            trace = []
            for j in range(max_trace_length):
                label_selected = np.random.choice(list(self.labels))
                label_selected_list = [label_selected]
                next_state, is_done = self.step_automaton(label_selected_list)
                trace.append(label_selected)
                if is_done or j == max_trace_length-1:
                    if next_state == self.accept_state:
                        pos_traces.append(trace)
                    else:
                        neg_traces.append(trace)
                    break

        all_traces = (pos_traces, neg_traces)

        return all_traces

def ltl2dot(formula, operator_list=None, filename_dot=None):
    declare_flag = False #True if you want to compute DECLARE assumption for the formula

    formula = formula.replace(" ", "")

    translator = Translator(formula)
    translator.formula_parser()
    translator.translate()
    translator.createMonafile(declare_flag) #it creates automa.mona file
    translator.invoke_mona() #it returns an intermediate automa.dot file

    dotHandler = DotHandler()
    dotHandler.modify_dot()
    dotHandler.output_dot() #it returns the final automa.dot file

    if filename_dot == None:
        filename_dot = "automa.dot"

    else:
        filename_dot = filename_dot + ".dot"
        former_path_dot = os.path.abspath("automa.dot")
        os.rename(former_path_dot, filename_dot)

    automata_path = os.path.abspath(filename_dot)
    automa_dir = os.path.split(automata_path)[0]
    os.remove("automa.mona")

    if operator_list == None:
        operator_list = ['F', 'G', 'X', 'U', '~', '&', '|', '(', ')',
            '-', '>', '<']

    modified_formula = ''
    for char in formula:
        if char in operator_list:
            modified_formula += '/'
        else:
            modified_formula += char

    dummylist1 = modified_formula.split('/')
    predicate_list = []
    for i in dummylist1:
        if i != '':
            predicate_list.append(i)

    return automata_path, predicate_list

def dot2DFA(dot_file_path, predicate_list):

    if os.path.isfile(dot_file_path) == False:
        print('Dot file is not found')
        return None

    target_automaton = DFAutomaton()
    file = open(dot_file_path, 'r')
    filelines = file.readlines()
    file.close()

    for line_index in range(len(filelines)):
        filelines[line_index] = filelines[line_index].strip("\n")
        filelines[line_index] = filelines[line_index].strip(";")


    add_regular_states = False
    add_edges = False
    for index in range(len(filelines)):

        if filelines[index] == "node [shape = doublecircle]":
            target_automaton.set_acc_state(filelines[index + 1])
        elif filelines[index] == "node [shape = circle]":
            add_regular_states = True
        elif add_regular_states and filelines[index] != "node [shape = box]":
            target_automaton.add_state(filelines[index])
        elif filelines[index] == "node [shape = box]":
            add_regular_states = False
        elif filelines[index] == 'init [shape = plaintext, label = ""]':
            add_edges = True

        elif add_edges and filelines[index][0:4] != "init":

            state_from = filelines[index][0]
            state_to = filelines[index][5]

            label_string = filelines[index].split('"')[1]
            label_string = label_string.replace(' ', '')
            label_string = label_string.replace(',', '')
            label_string = label_string.replace('\\','')
            label_list_raw = label_string.split('n')
            label_list = []

            for i in range(len(label_list_raw[0])):
                label_list.append([])
                for j in range(len(label_list_raw)):
                    label_list[i].append(label_list_raw[j][i])

            for label in label_list:
                condition = []
                for label_index in range(len(label)):
                    if label[label_index] == 'X':
                        pass
                    elif label[label_index] == '1':
                        condition.append(predicate_list[label_index])
                    else:
                        condition.append('~' + '%s'%(predicate_list[label_index]))
                target_automaton.add_edge(state_from, state_to, condition)

        elif filelines[index][0:7] == "init ->":
            target_automaton.set_init_state(filelines[index][-1])
            add_edges = False
        else:
            pass

    return target_automaton

def flie2genLTL(fliestr, predList):

    general_formula_str = fliestr.replace('&&', '&')
    general_formula_str = general_formula_str.replace('||', '|')
    general_formula_str = general_formula_str.replace('!', '~')
    general_formula_str = general_formula_str.replace('=', '-')
    for i in range(len(predList)):
        general_formula_str = general_formula_str.replace('x%s'%(i),'%s'%(predList[i]))

    return general_formula_str

def use_pytool(tracesdir, max_depth=None, maxNumOfFormulas=None):
    """Takes a trace as input and outputs an LTL formula """
    pytool_dir = os.path.realpath('../utils/samples2LTL-master/py_tool.py')
    os.system('export PATH="${PATH}:%s"'%(pytool_dir))
    command = 'python3 ' + pytool_dir + ' --traces=\'%s\''%(tracesdir)

    if type(max_depth) == int:
        command += '--max_depth=' + str(max_depth)
    if type(maxNumOfFormulas) == int:
        command += '--max_num_formulas=' + str(maxNumOfFormulas)

    output_pytool = subprocess.check_output(command, shell=True)
    output_pytool = str(output_pytool)
    dummy_list = output_pytool.split('\'')

    necessary_info = dummy_list[1]
    info_list = necessary_info.split('\\n')

    formula = info_list[0]
    formula = formula.replace(' ', '')
    total_exec_time = float(info_list[1])

    return formula, total_exec_time

def use_flie(tracedir, disable_until=False):
    if disable_until:
        output_flie = subprocess.check_output('flie-noUX -f ' + '%s'%(tracedir) +
         ' -v', shell=True)
    else:
        output_flie = subprocess.check_output('flie -f ' + '%s'%(tracedir) +
         ' -v', shell=True)
    output_flie = str(output_flie)
    dummy_list = output_flie.split('\'')

    necessary_info = dummy_list[1]
    info_list = necessary_info.split('\\n')

    formula = info_list[0]

    own_exec_time = float(info_list[2].split(':')[1])
    z3_exec_time = float(info_list[3].split(':')[1])
    total_exec_time = float(info_list[4].split(':')[1])

    return formula, own_exec_time, z3_exec_time, total_exec_time

if __name__ == "__main__":
    test_dot = True
    test_DFA = False
    test_flie2genLTL = False
    test_useflie = False
    test_DFA_sample = False
    test_usepytool = False
    test_compress_trace = False

    if test_usepytool:
        trace_dir = '../experiments/results0/traces/trace1.trace'
        flie_formula, t1, t2, t3 = use_flie(trace_dir)
        pytool_formula, total_time = use_pytool(trace_dir)

        print(flie_formula)
        print(pytool_formula)
        print(flie2genLTL(pytool_formula, ['a', 'b', 'c']))

    if test_DFA_sample:
        test_DFA = True

    if test_flie2genLTL:
        formula = '(x3)U((x0)U(x1&&(x1)U(x0)))'
        predList = ['yellow', 'red', 'purple', 'blue']
        print('Flie Formula is : ', end='')
        print(formula)
        genLTLformula = flie2genLTL(formula, predList)
        print('General Formula is : ', end='')
        print(genLTLformula)

    if test_dot:
        # formula = 'F(red & F(yellow)) & G(~purple)'
        # formula = 'F(yellow)'
        # formula = 'F(red & F(yellow))'
        # formula = '(Fc)->c'
        # formula = 'G~c'
        # formula = 'Fb'
        formula = '((~c)U(a))U(b)'
        formula = '(((e->g))U(g))U(f)'
        filename = "deneme3"
        path_automata, pred_list = ltl2dot(formula=formula, filename_dot=filename)

        DFA = dot2DFA(path_automata, pred_list)

        print('----TEST DFA translate function----')
        print('Formula : ', end='')
        print(formula)
        print('List of Predicates : ', end='')
        print(pred_list)
        print('Automata States : ', end='')
        print(DFA.get_states())
        print('Accepting State : ', end='')
        print(DFA.get_acc_state())
        print('Initial State : ', end='')
        print(DFA.get_init_state())
        edges = DFA.get_edges()
        print('Transitions : ')
        for key, value in sorted(edges.items()):
            print(key + ' : ', end='')
            print(value)

        subprocess.call('dot -Tps %s.dot -o %s.ps'%(filename, filename), shell=True)
        subprocess.call('gnome-open %s.ps'%(filename), shell=True)

    if test_DFA:
        if test_dot:
            dfa = DFA
        else:
            dfa = DFAutomaton()
            dfa.add_state("1")
            dfa.add_state("2")
            dfa.add_state("3")
            dfa.add_edge("1", "2", ["red"])
            dfa.add_edge("1", "1", ["~red"])
            dfa.add_edge("2", "1", ["~red", "~yellow"])
            dfa.add_edge("2", "2", ["red", "~yellow"])
            dfa.add_edge("2", "3", ["~red", "yellow"])
            dfa.add_edge("2", "3", ["red", "yellow"])
            dfa.add_edge("3", "3", [])
            dfa.set_acc_state("3")
            dfa.set_init_state("1")

        print('----TEST DFA Class----')
        print('Automata States : ', end='')
        print(dfa.get_states())
        print('Accepting State : ', end='')
        print(dfa.get_acc_state())
        print('Initial State : ', end='')
        print(dfa.get_init_state())
        edges = dfa.get_edges()
        print('Transitions : ')
        for key, value in sorted(edges.items()):
            print(key + ' : ', end='')
            print(value)
        print('Labels : ', end='')
        print(dfa.labels)
        dfa.reset()
        print(dfa.current_state)
        dfa.step_automaton(['green'])
        print(dfa.current_state)
        if test_DFA_sample:
            dfa.add_label(['yellow', 'red','blue', 'purple'])
            pos_samples, neg_samples = dfa.get_samples(num_traces=10)
            print(len(pos_samples))
            print(len(neg_samples))
            for k in pos_samples:
                print(k)
            pass

    if test_useflie:
        tracedir = '~/Desktop/flie-master/traces/0000.trace'
        formula, t1, t2, t3 = use_flie(tracedir)

        print(formula)
        print(t1)
        print(t2)
        print(t3)

    if test_compress_trace:
        import gym
        SEED = 100
        ENV_NAME = "Task2"
        np.random.seed(SEED)
        env = gym.make("gym_LTL_RL:OfficeWorldDoors%s-v0" %(ENV_NAME))
        env.action_space.seed(SEED)
        NUM_STEPS = 1000
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
                break
        # print(current_trace)
        compressed_trace = compress_trace_spaced(current_trace)
        print(compressed_trace)
        print()
        compressed_trace_extra = compress_trace_extra(current_trace)
        print(compressed_trace_extra)
