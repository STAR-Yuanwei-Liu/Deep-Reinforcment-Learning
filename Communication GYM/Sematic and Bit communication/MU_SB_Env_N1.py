import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math
import scipy.io as sio

matfn = u'reg_para.mat'  # load mat file of semantic similarity approximation parameters

class MU_SB_Env_N1(gym.Env):  # Env for OMA channel-based problem
    def __init__(self, seed, MAX_STEP):
        # basic system parameters
        self.user_num = 4  # number of users
        self.P_max = 10**(  20  / 10)*1e-3  # maximum power of BS (dBm to W)
        self.B_max = 1e6  # system bandwidth (MHz)
        self.fc = 2000 * 1e6  # carrier frequency (MHz)
        self.noise_power = 10 ** (-170 / 10) * 1e-3  # power spectral density of noise (dBm/Hz to W/Hz)
        self.time_slots = MAX_STEP  # max number of time slots
        self.t = 0  # time slot t
        self.d_epsilon = 1e-20  # prevent illegal math operation
        self.state_scale = 1e9  # scale the state
        self.action_type = 4

        # semantic related parameters
        self.K_m = 4  # semantic symbols per word
        self.kappa = 40  # bit to semantic transforming factor：average bits per word
        # semantic similarity approximation parameters
        self.A_1 = sio.loadmat(matfn)['B1'][0, self.K_m - 1]
        self.A_2 = sio.loadmat(matfn)['C'][0, self.K_m - 1]
        self.C_1 = sio.loadmat(matfn)['B3'][0, self.K_m - 1]
        self.C_2 = sio.loadmat(matfn)['B2'][0, self.K_m - 1]

        # random seed for reproduction
        np.random.seed(seed)
        self.seed()

        # ndarray for saving CSI at each fading block t
        self.h_m_t = np.random.normal(scale=1, size=self.user_num) \
                     + np.random.normal(scale=1, size=self.user_num) * 1j
        self.scale = 1

        self.test_state = np.random.uniform(0,1,size=(self.user_num,self.time_slots))

        # random fading matrix for all users and time slots
        self.H = np.random.normal(scale=1, size=(self.user_num, self.time_slots)) \
                 + np.random.normal(scale=1, size=(self.user_num, self.time_slots)) * 1j

        # locations (m)
        self.l_BS = np.array([0, 0])  # location of BS
        self.l_user = np.random.uniform(-2000, 2000, size=(self.time_slots, 2, self.user_num))  # generate random locations of user

        # define action and state spaces
        self.action_dim = int((self.action_type-1) * self.user_num + self.user_num/2)  # mode selection, power and bandwidth allocation for each user
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)

        self.state_dim = self.user_num  # current SNR/SINR for each user
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.state_dim,), dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # calculate CSI for each user at time slot t
    def get_CSI(self):
        for m in range(self.user_num):
            dist_BS_m = np.linalg.norm(self.l_user[self.t, :, m] - self.l_BS)  # distance from BS to user m
            path_loss_m = 10 ** (-30 / 10) * (dist_BS_m ** (-2.2))  # path loss
            H = self.H[m, self.t]  # ** self.scale  # random fading at time slot t
            self.h_m_t[m] = np.sqrt(path_loss_m) * H

    #  interaction of agent and environment
    def step(self, action):
        # 1st is mode selection, 2nd is band allocation, 3rd is power allocation
        mode_sel = action[0: self.user_num]
        band_allo = action[self.user_num: int(3/2 * self.user_num)]
        p_allo = action[int(3/2 * self.user_num): int(5/2 * self.user_num)]
        pair = action[int(5/2 * self.user_num):]

        # user pairing indicator
        sort_index = sorted(range(len(pair)), key=lambda k: pair[k], reverse=True)
        sort_index = np.reshape(sort_index, (int(self.user_num/2), 2))

        # start to calculate sum rate
        s_rew = 0
        rate = np.zeros(self.user_num)
        SINR = np.zeros(self.user_num)
        channel_gain = abs(self.h_m_t) ** 2

        for m in range(int(self.user_num/2)):
            # consider the NOMA paor
            decoding_first = int(0 if channel_gain[sort_index[m][0]] >= channel_gain[sort_index[m][1]] else 1)
            decoding_second = int(abs(decoding_first - 1))

            SINR[sort_index[m][decoding_first]] = channel_gain[sort_index[m][decoding_first]] \
                                                  * p_allo[sort_index[m][decoding_first]] \
                                                  * self.P_max / (band_allo[m] * self.B_max * self.noise_power) if band_allo[m] != 0 else 0

            interference = channel_gain[sort_index[m][decoding_second]] * p_allo[sort_index[m][decoding_first]] * self.P_max
            SINR[sort_index[m][decoding_second]] = channel_gain[sort_index[m][decoding_second]] \
                                                  * p_allo[sort_index[m][decoding_second]] \
                                                  * self.P_max / (band_allo[m] * self.B_max * self.noise_power + interference) if band_allo[m] != 0 else 0

            # consider the mode selection between semantic and bit
            if SINR[sort_index[m][decoding_first]] != 0:
                if mode_sel[sort_index[m][decoding_first]] <= 0:  # semantic rate
                    s_s = self.A_1 + (self.A_2 - self.A_1) / (
                            1 + math.exp(-self.C_1 * 10 * math.log10(SINR[sort_index[m][decoding_first]]) - self.C_2))  # semantic similarity
                    if s_s < 0.6:
                        s_rew -= 20
                    rate[sort_index[m][decoding_first]] = (band_allo[m] * self.B_max / self.K_m) * s_s
                else:  # equivalent semantic rate by bit rate
                    bit_rate = band_allo[m] * self.B_max * math.log2(1 + SINR[sort_index[m][decoding_first]])
                    rate[sort_index[m][decoding_first]] = bit_rate / self.kappa
            else:
                rate[sort_index[m][decoding_first]] = 0

            if SINR[sort_index[m][decoding_second]] != 0:
                if mode_sel[sort_index[m][decoding_second]] <= 0:  # semantic rate
                    s_s = self.A_1 + (self.A_2 - self.A_1) / (
                            1 + math.exp(-self.C_1 * 10 * math.log10(SINR[sort_index[m][decoding_second]]) - self.C_2))  # semantic similarity
                    if s_s < 0.6:
                        s_rew -= 20
                    rate[sort_index[m][decoding_second]] = (band_allo[m] * self.B_max / self.K_m) * s_s
                else:  # equivalent semantic rate by bit rate
                    bit_rate = band_allo[m] * self.B_max * math.log2(1 + SINR[sort_index[m][decoding_second]])
                    rate[sort_index[m][decoding_second]] = bit_rate / self.kappa
            else:
                rate[sort_index[m][decoding_second]] = 0

        sum_rate = sum(rate)

        fair_index = sum_rate ** 2 / (self.user_num * sum(np.square(rate))) if sum_rate != 0 else 0
        # reward to meet rate constraints
        QoS_rew = 0
        for m in range(self.user_num):
            if rate[m] <= 1e3:
                QoS_rew -= 20
        rew = 1e-4 * fair_index * sum_rate + QoS_rew + s_rew  # set main reward as sum rate

        # next time slot
        self.t += 1
        if self.t == self.time_slots:  # judge if done
            done = True
            self.t = self.time_slots - 1
        else:
            done = False

        self.get_CSI()

        next_state = abs(self.h_m_t)**2 * self.state_scale  # next state
        # next_state = SNR  # next state
        # Optionally we can pass additional info, we are not using that for now
        info = [action, rate, sum_rate, next_state, SINR, sort_index, fair_index]

        return np.array(next_state).astype(np.float32), rew, done, info

    #  environment reset
    def reset(self):
        self.t = 0
        self.get_CSI()
        state = abs(self.h_m_t)**2 * self.state_scale
        return np.array(state).astype(np.float32)


if __name__ == '__main__':
    env = MU_SB_Env_N1()
    # wrap it
    un_wrapped_env = MU_SB_Env_N1()
    # wrapped_env = make_vec_env(lambda: un_wrapped_env, n_envs=1)
