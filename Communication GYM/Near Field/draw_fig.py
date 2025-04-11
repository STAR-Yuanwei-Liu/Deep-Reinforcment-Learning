"""
Here is an example for plotting the rewards.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from more_itertools import chunked
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import ScalarFormatter

#------ load data ------
path = "results/"
ave_num = 50
update_steps = 512

reward = np.load(path+"PPO_nf_hybrid_alloc_reward_K_P_N_U_1_0.1_4.npy", allow_pickle=True)
reward_ppo = [sum(x) / len(x) for x in chunked(reward, ave_num)]
print("The average reward:", reward_ppo)

fig = plt.figure()
ms = 7
lw= 2

plt.plot(np.array(range(len(reward_ppo))) * ave_num * update_steps, reward_ppo, 'd-', label="PPO, $P_{\\rm max}=0$ dBm, $K=1$, $N_U = 4$"
                        , markersize=ms,linewidth=lw
                         ,  markeredgewidth=1.5)

plt.xlabel('Training Step')
# Setting the x-axis to scientific notation
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().xaxis.get_major_formatter().set_scientific(True)
plt.gca().xaxis.get_major_formatter().set_powerlimits((0, 0))

plt.ylabel('Average Reward')
plt.legend(loc='lower right')

#------Fig Show------
plt.show()
