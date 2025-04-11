from env_core import EnvCore
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.a2c.policies import MlpPolicy
import torch
import pickle
env = EnvCore()

policy_kwargs = dict(activation_fn=torch.nn.ReLU,net_arch=dict(pi=[256,256], qf=[256,256]))
model = A2C(MlpPolicy, env,policy_kwargs=policy_kwargs, verbose=0, learning_rate=1e-4)

def evaluate(model, num_episodes=100):
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "test episodes num:", num_episodes)
    return mean_episode_reward

store = []
for _ in range(500):
    print('Episode:',_)
    model.learn(total_timesteps=100) # train model
    if _ % 10 == 0:
        mean_reward = evaluate(model, num_episodes=10) # evaluate model
        store.append(mean_reward)
        print(env.UAV_position)

with open('store.pkl', 'wb') as file:
    pickle.dump(store, file)
model.save('a2c')