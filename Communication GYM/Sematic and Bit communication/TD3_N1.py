import argparse
import random
import time
import os
import gym
from MU_SB_Env_N1 import MU_SB_Env_N1
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from IPython.display import clear_output
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model
from more_itertools import chunked

tfd = tfp.distributions
Normal = tfd.Normal

tl.logging.set_verbosity(tl.logging.DEBUG)

RANDOM_SEED = 2
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)  # reproducible

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#####################  hyper parameters  ####################
action_range = 1.  # scale action, [-action_range, action_range]
save_name = 'train'
save_num = 3e-4

# RL training
max_steps = 50  # maximum number of steps for one episode
max_frames = max_steps * 5000  # total number of steps for training
test_frames = max_steps  # total number of steps for testing
test_per_episode = 1  # interval of episodes for every testing
batch_size = 128  # udpate batchsize
explore_steps = 200  # steps for random action sampling in the beginning of training
update_itr = 1  # repeated updates for single step
hidden_dim = 64  # size of hidden layers for networks
q_lr = 1e-3  # q_net learning rate
policy_lr = 1e-3  # policy_net learning rate
policy_target_update_interval = 3  # delayed steps for updating the policy network and target networks
explore_noise_scale = 0.3  # range of action noise for exploration
eval_noise_scale = 0.3  # range of action noise for evaluation of action value
reward_scale = 1.  # value range of reward
replay_buffer_size = 5e5  # size of replay buffer

###############################  TD3  ####################################
class QNetwork(Model):
    ''' the network for evaluate values of state-action pairs: Q(s,a) '''

    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-1):
        super(QNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name='q3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(Model):
    ''' the network for generating non-determinstic (Gaussian distributed) action from the state input '''

    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-1):
        super(PolicyNetwork, self).__init__()

        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')

        self.output_linear1 = Dense(n_units=int(env.user_num), W_init=w_init, \
                                    b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim,
                                    name='policy_output1')

        self.output_linear2 = Dense(n_units=int(env.user_num/2-1), W_init=w_init, \
                                   b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim,
                                   name='policy_output2')

        self.output_linear3= Dense(n_units=int(env.user_num-1), W_init=w_init, \
                                    b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim,
                                    name='policy_output3')

        self.output_linear4 = Dense(n_units=int(env.user_num), W_init=w_init, \
                                    b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim,
                                    name='policy_output4')

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)

        ###########
        temp_out1 = self.output_linear1(x)
        temp_out2 = self.output_linear2(x)
        temp_out3 = self.output_linear3(x)
        temp_out4 = self.output_linear4(x)

        ref_neuron2 = tf.constant([-2.], dtype=np.float32, shape=(temp_out2.shape[0], 1))
        ref_neuron3 = tf.constant([-2.], dtype=np.float32, shape=(temp_out3.shape[0], 1))

        x2 = tf.concat([temp_out2, ref_neuron2], 1)
        x3 = tf.concat([temp_out3, ref_neuron3], 1)

        output1 = tf.nn.tanh(temp_out1)  # unit range output [-1, 1]
        output2 = tf.nn.softmax(x2)
        output3 = tf.nn.softmax(x3)
        output4 = tf.nn.sigmoid(temp_out4)


        output = tf.concat([output1, output2, output3, output4], 1)

        return output

    def evaluate(self, state, eval_noise_scale):
        ''' 
               generate action with state for calculating gradients;
               eval_noise_scale: as the trick of target policy smoothing, for generating noisy actions.
               '''
        state = state.astype(np.float32)
        action = self.forward(state)  # action range [-1,1]

        action = self.action_range * action  # action mapping

        # add noise
        normal = Normal(0, eval_noise_scale)  # normal distribution
        eval_noise_clip = eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)  # clip noise
        action = action + noise  # action plus  noise

        if eval_noise_scale != 0:
            action = action.numpy()
            for m in range(batch_size):
                action[m][0:env.user_num] = np.clip(action[m][0:env.user_num], -1, 1)

                action[m][env.user_num: int(3 / 2 * env.user_num)] = action[m][env.user_num: int(3 / 2 * env.user_num)].clip(0.01, 1)
                action[m][env.user_num: int(3 / 2 * env.user_num)] = action[m][env.user_num: int(3 / 2 * env.user_num)] \
                                                                  / np.sum(action[m][env.user_num: int(3 / 2 * env.user_num)])

                action[m][int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)] = action[m][int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)].clip(0.01, 1)
                action[m][int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)] = action[m][int(3 / 2
                                  * env.user_num): int(5 / 2 * env.user_num)] / np.sum(action[m][int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)])

                action[m][int(5 / 2 * env.user_num):] = np.clip(action[m][int(5 / 2 * env.user_num):], 0, 1)

            action = tf.convert_to_tensor(action)

        return action

    # inout state，output action
    def get_action(self, state, explore_noise_scale, greedy=False):
        ''' generate action with state for interaction with envronment '''
        action = self.forward([state])
        if greedy:
            action = action.numpy()[0]

            action[env.user_num: int(3 / 2 * env.user_num)] = action[env.user_num: int(3 / 2 * env.user_num)].clip(0.01,1)
            action[env.user_num: int(3 / 2 * env.user_num)] = action[env.user_num: int(3 / 2 * env.user_num)] \
                                                              / np.sum(action[env.user_num: int(3 / 2 * env.user_num)])

            action[int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)] = action[int(3 / 2 * env.user_num): int(
                       5 / 2 * env.user_num)].clip(0.01, 1)
            action[int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)] = action[int(3 / 2 * env.user_num): int(
                       5 / 2 * env.user_num)] / np.sum(action[int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)])

            return action
        else:
            # add noise
            action = action.numpy()[0]
            normal = Normal(0, explore_noise_scale)  # normal distribution
            noise = normal.sample(action.shape) * 0.3  # sample a noise
            action = self.action_range * action + noise  # action + noise

            action = action.numpy()
            action[0:env.user_num] = np.clip(action[0:env.user_num], -1, 1)

            action[env.user_num: int(3 / 2 * env.user_num)] = action[env.user_num: int(3 / 2 * env.user_num)].clip(0.01,1)
            action[env.user_num: int(3 / 2 * env.user_num)] = action[env.user_num: int(3 / 2 * env.user_num)] \
                                                              / np.sum(action[env.user_num: int(3 / 2 * env.user_num)])

            action[int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)] = action[int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)].clip(0.01, 1)
            action[int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)] = action[int(3 / 2
                             * env.user_num): int(5 / 2 * env.user_num)] / np.sum(action[int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)])

            action[int(5 / 2 * env.user_num):] = np.clip(action[int(5 / 2 * env.user_num):], 0, 1)

            action = tf.convert_to_tensor(action)

            return action.numpy()

    def sample_action(self, ):
        ''' generate random actions for exploration '''
        a0 = tf.random.uniform([env.user_num], -1, 1)
        a1a2 = tf.random.uniform([int(3/2 * env.user_num)], 0.01, 1)
        a3 = tf.random.uniform([env.user_num], 0, 1)
        a = tf.concat([a0, a1a2, a3], 0)

        action = a.numpy()

        action[env.user_num: int(3 / 2 * env.user_num)] = action[env.user_num: int(3 / 2 * env.user_num)] \
                                                          / np.sum(action[env.user_num: int(3 / 2 * env.user_num)])

        action[int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)] = action[int(3 / 2 * env.user_num): int(5 / 2
                         * env.user_num)] / np.sum(action[int(3 / 2 * env.user_num): int(5 / 2 * env.user_num)])

        a = tf.convert_to_tensor(action)

        return self.action_range * a.numpy()


class TD3_Trainer():

    def __init__(
            self, replay_buffer, hidden_dim, action_range, policy_target_update_interval=1, q_lr=3e-4, policy_lr=3e-4
    ):
        self.replay_buffer = replay_buffer

        # initialize all networks
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = tf.optimizers.Adam(q_lr)
        self.q_optimizer2 = tf.optimizers.Adam(q_lr)
        self.policy_optimizer = tf.optimizers.Adam(policy_lr)

    # initialization
    def target_ini(self, net, target_net):
        ''' hard-copy update for initializing target networks '''
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    # soft update
    def target_soft_update(self, net, target_net, soft_tau):
        ''' soft update the target net with Polyak averaging '''
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
                # Original Parameter Percentage + Current Parameter Percentage
            )
        return target_net

    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.99, soft_tau=1e-2):
        ''' update all networks in TD3 '''
        self.update_cnt += 1        # Calculating the number of updates
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)     # from buffer sample data

        reward = reward[:, np.newaxis]  # expand dim
        done = done[:, np.newaxis]

        # inputs',from target_policy_net calculate a'
        new_next_action = self.target_policy_net.evaluate(
            next_state, eval_noise_scale=eval_noise_scale
        )  # clipped normal noise

        # normalize reward
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (
            np.std(reward, axis=0) + 1e-10
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function。
        # two qnet，we choose the smaller one
        target_q_input = tf.concat([next_state, new_next_action], 1)  # the dim 0 is number of samples
        target_q_min = tf.minimum(self.target_q_net1(target_q_input), self.target_q_net2(target_q_input))

        # Calculate the value of target_q for updating q_net
        # There was a previous change of done from a boolean variable to an int, just so that it could be calculated directly here
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_input = tf.concat([state, action], 1)  # input of q_net

        # update q_net1
        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.trainable_weights))

        # update q_net2
        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.trainable_weights))

        # Training Policy Function
        if self.update_cnt % self.policy_target_update_interval == 0:
            # update policy_net
            with tf.GradientTape() as p_tape:
                new_action = self.policy_net.evaluate(
                    state, eval_noise_scale=0.0
                )  # no noise, deterministic policy gradients

                new_q_input = tf.concat([state, new_action], 1)
                # ''' implementation 1 '''
                # predicted_new_q_value = tf.minimum(self.q_net1(new_q_input),self.q_net2(new_q_input))
                ''' implementation 2 '''
                predicted_new_q_value = self.q_net1(new_q_input)
                policy_loss = -tf.reduce_mean(predicted_new_q_value)
            p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

            # Soft update the target nets
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

    def save_weights(self):  # save trained weights
        tl.files.save_npz(self.q_net1.trainable_weights, name='model/model_q_net1_N1.npz')
        tl.files.save_npz(self.q_net2.trainable_weights, name='model/model_q_net2_N1.npz')
        tl.files.save_npz(self.target_q_net1.trainable_weights, name='model/model_target_q_net1_N1.npz')
        tl.files.save_npz(self.target_q_net2.trainable_weights, name='model/model_target_q_net2_N1.npz')
        tl.files.save_npz(self.policy_net.trainable_weights, name='model/model_policy_net_N1.npz')
        tl.files.save_npz(self.target_policy_net.trainable_weights, name='model/model_target_policy_net_N1.npz')

    def load_weights(self):  # load trained weights
        tl.files.load_and_assign_npz(name='model/model_q_net1_N1.npz', network=self.q_net1)
        tl.files.load_and_assign_npz(name='model/model_q_net2_N1.npz', network=self.q_net2)
        tl.files.load_and_assign_npz(name='model/model_target_q_net1_N1.npz', network=self.target_q_net1)
        tl.files.load_and_assign_npz(name='model/model_target_q_net2_N1.npz', network=self.target_q_net2)
        tl.files.load_and_assign_npz(name='model/model_policy_net_N1.npz', network=self.policy_net)
        tl.files.load_and_assign_npz(name='model/model_target_policy_net_N1.npz', network=self.target_policy_net)


def plot(frame_idx, rewards):  # plot figure
    plt.ion()
    plt.cla()
    clear_output(True)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.show()
    plt.pause(0.1)
    plt.ioff()

class ReplayBuffer:
    '''
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    '''

    def __init__(self, capacity):
        self.capacity = capacity        #buffer maximun
        self.buffer = []                #buffer list
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    ''' normalize the actions to be in reasonable range '''

    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

if __name__ == '__main__':
    # initialization of env
    env = MU_SB_Env_N1(RANDOM_SEED, max_steps)
    action_dim = env.action_space.shape[0]      # actor space
    state_dim = env.observation_space.shape[0]  # state space

    # initialization of buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # initialization of trainer
    td3_trainer=TD3_Trainer(replay_buffer, hidden_dim=hidden_dim, policy_target_update_interval=policy_target_update_interval, \
    action_range=action_range, q_lr=q_lr, policy_lr=policy_lr )

    # set train mode
    td3_trainer.q_net1.train()
    td3_trainer.q_net2.train()
    td3_trainer.target_q_net1.train()
    td3_trainer.target_q_net2.train()
    td3_trainer.policy_net.train()
    td3_trainer.target_policy_net.train()

    # training loop
    if args.train:
        is_show = 0
        episode = 0
        frame_idx = 0                           # total step
        rewards = []                            # record reward
        draw_rewards = []
        info = []
        t0 = time.time()
        all_episodes = int(max_frames / max_steps) - 1  # total episodes
        while frame_idx < max_frames:           # start training
            state = env.reset()                 # initailize state
            state = state.astype(np.float32)
            episode_reward = 0
            if frame_idx < 1:                   # initialize trainer
                print('intialize')
                _ = td3_trainer.policy_net([state])  # need an extra call here to make inside functions be able to use model.forward
                _ = td3_trainer.target_policy_net([state])

            interval = (explore_noise_scale-0.001) / (all_episodes - 1000)
            now_explore_noise_scale = explore_noise_scale - episode * interval
            # training start
            for step in range(max_steps):
                if frame_idx > explore_steps:       # if less than explore_steps, randomly; else, get_action
                    action = td3_trainer.policy_net.get_action(state, explore_noise_scale=now_explore_noise_scale)  #noisy action
                else:
                    action = td3_trainer.policy_net.sample_action()

                # interaction
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                done = 1 if done ==True else 0

                # store in replay_buffer
                replay_buffer.push(state, action, reward, next_state, done)

                is_show = 1 if episode % 100 == 0 or episode == all_episodes else 0

                # if is_show:
                #     print('\nstate:', state)
                #     print('mode_sel: ', _[0][0])
                #     print('action:', _[0][1], _[0][2])
                #     print('reward:', reward)
                #     print('SNR: ', _[4])
                #     print('rate: ', _[1])

                episode_reward += reward

                if step == max_steps - 1:
                    print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
                          .format(episode, all_episodes, episode_reward, time.time() - t0))

                # next state
                state = next_state
                frame_idx += 1

                # If the data exceeds the size of a batch_size, then start updating
                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):         # Updates here can be multiple times
                        td3_trainer.update(batch_size, eval_noise_scale=eval_noise_scale, reward_scale=1.)

                if done:
                    break

            sum_rate_all = 0
            fair_index_all = 0
            if not episode % test_per_episode or episode == all_episodes:
                t1 = time.time()
                state = env.reset()  # initialize state
                state = state.astype(np.float32)
                episode_reward = 0
                for j in range(max_steps):
                    action = td3_trainer.policy_net.get_action(state, explore_noise_scale=0., greedy=True)
                    next_state, reward, done, _ = env.step(action)
                    next_state = next_state.astype(np.float32)

                    if episode == all_episodes:
                        info.append(_)

                    if is_show:
                        print('\nstate:', state)
                        print('mode_sel: ', _[0][0:env.user_num])
                        print('action:', _[0][env.user_num: int(3/2*env.user_num)], _[0][int(3/2*env.user_num):int(5/2*env.user_num)],
                              _[0][int(5/2*env.user_num):])
                        print('reward:', reward)
                        print('SINR: ', _[4])
                        print('rate: ', _[1])
                        print('user pair: ', _[5])
                        print('sum rate: ', _[2])
                        print('fair index: ', _[6])

                        sum_rate_all += _[2]
                        fair_index_all += _[6]

                    episode_reward += reward

                    if j == max_steps - 1:
                        rewards.append(episode_reward)
                        if is_show:
                            print(
                                '\nTest: '
                                'Episode: {}/{} |  '
                                'Episode Reward: {:.4f} | Running Time: {:.4f}'.format(
                                    episode, all_episodes, episode_reward, time.time() - t1))
                            print('ave sum rate all ts: ', sum_rate_all / max_steps)
                            print('ave fair index all ts: ', fair_index_all / max_steps )

                    state = next_state

            if rewards:
                ave_num = 250
                if episode and not episode % ave_num:
                    draw_rewards = [sum(x) / len(x) for x in chunked(rewards, ave_num)]
                plt.ion()
                plt.cla()
                plt.plot(np.array(range(len(draw_rewards))) * ave_num, draw_rewards,
                         'v-', c='darkred', markerfacecolor='none', markeredgewidth=1.5,
                         markersize=8)  # plot the episode vt
                plt.grid(True)
                plt.xlabel('Episode')
                plt.ylabel('Accumulative Reward')
                plt.show()
                plt.pause(0.1)

            episode = int(frame_idx / max_steps)  # current episode

        plt.ioff()
        plt.show()
        np.save("results/rewards_N1_"+save_name+str(save_num)+".npy", rewards)
        np.save("results/info_N1_"+save_name+str(save_num)+".npy", info)

    if args.test:
        frame_idx = 0
        rewards = []
        t0 = time.time()

        td3_trainer.load_weights()

        while frame_idx < test_frames:
            state = env.reset()
            state = state.astype(np.float32)
            episode_reward = 0
            if frame_idx < 1:
                print('intialize')
                _ = td3_trainer.policy_net(
                    [state]
                )  # need an extra call to make inside functions be able to use forward
                _ = td3_trainer.target_policy_net([state])

            for step in range(max_steps):
                action = td3_trainer.policy_net.get_action(state, explore_noise_scale=0, greedy=True)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                done = 1 if done == True else 0

                state = next_state
                episode_reward += reward
                frame_idx += 1

                if done:
                    break
            episode = int(frame_idx / max_steps)
            all_episodes = int(test_frames / max_steps)
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
            .format(episode, all_episodes, episode_reward, time.time()-t0 ) )
            rewards.append(episode_reward)

