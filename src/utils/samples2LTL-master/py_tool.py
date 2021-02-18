import argparse
from solverRuns import run_solver
from utils.Traces import Trace, ExperimentTraces

parser = argparse.ArgumentParser()

parser.add_argument("--traces", type=str, default="traces/dummy.trace")
parser.add_argument("--max_depth", type=int, default='8')
parser.add_argument("--start_depth", type=int, default='1')
parser.add_argument("--max_num_formulas", type=int, default='1')
parser.add_argument("--iteration_step", type=int, default='1')
parser.add_argument("-print_all_formula", action="store_true")
parser.set_defaults()
args = parser.parse_args(print_all_formula=False)

TRACE_DIR = args.traces
MAX_DEPTH = args.max_depth
START_DEPTH = args.start_depth
MAX_NUM_FORMULAS = args.max_num_formulas
ITERATION_STEP = args.iteration_step
PRINT_ALL_FORMULA = args.print_all_formula


traces = ExperimentTraces()
traces.readTracesFromFile(TRACE_DIR)

[formulas, timePassed] = run_solver(finalDepth=MAX_DEPTH, traces=traces,
    maxNumOfFormulas = MAX_NUM_FORMULAS, startValue=START_DEPTH,
    step=ITERATION_STEP)

if PRINT_ALL_FORMULA:
    for formula1 in formulas:
        print(formula1.prettyPrint())
else:
    print(formulas[0].prettyPrint())

print(timePassed)
