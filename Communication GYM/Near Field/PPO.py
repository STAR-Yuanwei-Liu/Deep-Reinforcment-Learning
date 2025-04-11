import torch
import time
import numpy as np
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_agent import PPO_continuous
from Env_NF import Env_NF

def evaluate_policy(args, env, agent, state_norm):
    times = 1
    evaluate_reward = 0
    info = []
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)

        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)

            episode_reward += r
            info.append(_)

            s = s_

            print('data rate: ', _[0])
            # print('consumed power per user:', _[1])
            # print('data stream selection: ', action[0:env.S_d])
            # print('W beta: ', action[env.S_d: env.S_d + env.N_RF * env.S_d])
            # print('W theta:', action[env.S_d + env.N_RF * env.S_d: env.S_d + 2 * env.N_RF * env.S_d])
            # print('P theta: ', action[env.S_d + 2 * env.N_RF * env.S_d:])

        evaluate_reward += episode_reward

    return evaluate_reward / times, info

def main(args, env_name, seed):
    env = Env_NF(seed, max_step=args.max_episode_steps)
    eval_env = Env_NF(seed, max_step=args.max_episode_steps)

    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])

    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))
    print("rayleigh distance: ", env.r_dis)

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    eval_info = []
    total_steps = 0  # Record the total steps during the training
    total_episodes = 0

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    state_norm = Normalization(shape=args.state_dim)  # tate normalization
    if args.use_reward_norm:  # reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    t0 = time.time()
    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()

        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, evaluate_info = evaluate_policy(args, eval_env, agent, state_norm)

                eval_info.append(evaluate_info)
                evaluate_rewards.append(evaluate_reward)
                print('PPO_env_{}_bs{}_mbs{}_hd{}_lc{}_la{}_ce{}_ec{}_nb{}_nu{}'.format(
                    env_name, args.batch_size, args.mini_batch_size, args.hidden_width, args.lr_c, args.lr_a,
                    args.epsilon, args.entropy_coef, env.N_BS, env.N_U))
                print("\n evaluate_num:{} \t evaluate_reward:{} \t Running Time: {:.4f}".format(evaluate_num, evaluate_reward, time.time() - t0))
                # writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)

            total_episodes += 1

    # Save the rewards
    # args.batch_size, args.mini_batch_size, args.hidden_width, args.lr_c, args.lr_a, args.epsilon, args.entropy_coef, env.N_BS, env.N_U
    np.save('results/PPO_{}_reward_{}_{}.npy'.format(
        env_name, var_para, SAVE_NUM), np.array(evaluate_rewards))
    np.save('results/PPO_{}_info_{}_{}.npy'.format(
        env_name, var_para, SAVE_NUM), eval_info)

if __name__ == '__main__':
    MAX_STEPS = 5
    MAX_EPISODES = 50000

    var_para = 'K_P_N_U'
    SAVE_NUM = f"{Env_NF().K}_{Env_NF().P_max}_{Env_NF().N_U}"
    MAX_EVAL_NUM = MAX_STEPS * MAX_EPISODES // 512

    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=MAX_EPISODES * MAX_STEPS, help=" Maximum number of training steps")
    parser.add_argument("--max_episode_steps", type=int, default=MAX_STEPS, help=" Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=512, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--action_dim_scale", type=int, default=1, help="Scaling the output layer of actor to match action dim")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.5, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.0, help="policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=False, help="set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="tanh activation function")


    args = parser.parse_args()
    main(args, env_name='nf_hybrid_alloc', seed=10)




