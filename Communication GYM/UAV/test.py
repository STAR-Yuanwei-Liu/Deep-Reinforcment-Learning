from env_core import EnvCore
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from stable_baselines3 import A2C
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


env = EnvCore()
model = A2C.load('a2c.zip', env=env)

plt.ion()
fig, ax = plt.subplots()

uav_trajectories_x = [[] for _ in range(env.UAV_number)]
uav_trajectories_y = [[] for _ in range(env.UAV_number)]

done = False
obs = env.reset()

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    uav_x = env.UAV_position[0, :]
    uav_y = env.UAV_position[1, :]
    user_x = env.User_position[0, :]
    user_y = env.User_position[1, :]


    for i in range(env.UAV_number):
        uav_trajectories_x[i].append(uav_x[i])
        uav_trajectories_y[i].append(uav_y[i])


    ax.clear()
    ax.scatter(user_x, user_y, c='blue', label='Users')
    ax.scatter(uav_x, uav_y, c='red', marker='^', label='UAVs')


    for i in range(env.UAV_number):
        ax.plot(uav_trajectories_x[i], uav_trajectories_y[i], 'r--', linewidth=1)

    ax.set_xlim(-100, 500)
    ax.set_ylim(-100, 500)
    ax.set_title("UAV Trajectory with Users")
    ax.legend()
    ax.grid(True)
    plt.pause(0.1)  # 动画效果
