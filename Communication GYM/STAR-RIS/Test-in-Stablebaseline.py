from STAR_RIS_ES import STAR_RIS_Env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

# env = gym.make('CartPole-v1')
# env = gym.make('custom_env-v0')
env = STAR_RIS_Env()

model = PPO(MlpPolicy, env, verbose=0)


# Random Agent, before training

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward befroe training:{mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for 10000 steps
model.learn(total_timesteps=10000)
# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward after training:{mean_reward:.2f} +/- {std_reward:.2f}")