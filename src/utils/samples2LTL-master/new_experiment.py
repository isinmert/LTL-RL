from solverRuns import run_solver, run_dt_solver
import argparse
from utils.Traces import Trace, ExperimentTraces


parser = argparse.ArgumentParser()
parser.add_argument("--trace", type=str, default='traces/dummy.trace')
parser.add_argument("-sat", action='store_true', default=False, dest="sat")
parser.add_argument("-dt", action='store_true', default=False, dest="dt")

args = parser.parse_args()
traces = ExperimentTraces()
traces.readTracesFromFile(args.trace)

if args.sat:
    [results, time_elapsed] = run_solver(finalDepth=10, traces=traces)
    print('RESULTS OF SAT SOLVER')
    print('-----------------')
    print('Formula : ', end='')
    print(results)
    print('Total Time : ', end='')
    print(time_elapsed)
    print('')
    print('')

if args.dt:
    [time_elapsed, numAtoms, numPrimitives] = run_dt_solver(traces=traces)
    print('RESULTS OF DECISION TREE')
    print('-----------------')
    print('Formula : ', end='')
    print(numAtoms, end='     ')
    print(numPrimitives)
    print('Total Time : ', end='')
    print(time_elapsed)
