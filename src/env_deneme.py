import gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default='Task2', type=str)
args = parser.parse_args()
task = args.environment

# print('Environment Name is %s' %(task))

# env = gym.make("gym_LTL_RL:OfficeWorldBigTask1-v0")
env = gym.make("gym_LTL_RL:OfficeWorldTaskBCA-v0")
# env = gym.make("gym_LTL_RL:OfficeWorldDoorsTask2-v0")
env.set_seed(1)
# env.set_sliprate(0.5)
env.play()
print(env.get_observables())
