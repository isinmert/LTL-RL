from matplotlib import pyplot as plt
import numpy as np
import argparse

fs=14

rewards_episode_list = []
for i in [1,3,4]:
    data = np.load('../experiments/BCAresults%d/rewards_episode.npy'%(i),
        allow_pickle=True)
    rewards_episode_list.append(data)

# print(rewards_episode_list)
rew_arr = np.array(rewards_episode_list)
# print(rew_arr)
print(rew_arr.shape)

# print(np.sum(rew_arr, 0))

mean_rew = np.sum(rew_arr, 0) / rew_arr.shape[0]
print(mean_rew.shape)
cov_arr = np.zeros(mean_rew.shape)
for i in range(cov_arr.shape[0]):
    for j in range(rew_arr.shape[0]):
        cov_arr[i] += (rew_arr[j][i] - mean_rew[i])**2
    cov_arr[i] = np.sqrt(cov_arr[i])

upper_bound = np.minimum(mean_rew + 0.2*cov_arr, 1.0)
lower_bound = np.maximum(mean_rew - 0.2*cov_arr, 0.0)

# print(mean_rew)

plt.plot(mean_rew)
# plt.plot(upper_bound)
# plt.plot(lower_bound)
plt.xlabel('Episodes', fontsize=fs)
plt.ylabel('Mean Reward', fontsize=fs)
# plt.title('Rewards in Episodes (4 run) Task-II')
plt.show()
