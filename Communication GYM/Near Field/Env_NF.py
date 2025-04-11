import numpy as np
import gym
from gym import spaces
import math


class Env_NF(gym.Env):  # Env for NFC
    def __init__(self, seed=1, max_step=5):
        # random seed for reproduction
        np.random.seed(seed)
        # basic system parameters
        self.K = 2  # number of users
        self.N_BS = 256  # number of BS antennas
        self.N_U = 4  # number of user antennas
        self.S_d = self.K * self.N_U // 2  # number of data streams
        self.N_RF = self.S_d * 2  # number of RF chains
        self.BS_z = 15  # height of BS (m)
        self.P_max = 10 ** (-10 / 10)  # maximum power of BS (dBm to mW)
        self.P_unit = self.P_max / self.N_BS
        self.B_max = 1e6  # system bandwidth (1MHz)
        self.fc = 10 * 1e9  # carrier frequency (Hz)
        self.light_speed = 3e8
        self.lambda_c = self.light_speed / self.fc
        self.noise_power = 10 ** (-105 / 10)   # power spectral density of noise (dBm to mW)
        self.time_slots = max_step  # max number of time slots
        self.t = 0  # time slot t
        self.d_epsilon = 1e-20  # prevent illegal math operation
        self.state_scale = 1e5  # scale the state
        self.rate_qos = 1
        self.s_k = int(self.S_d / self.K)  # data stream

        # array for saving sinr
        self.sinr = np.zeros(shape=(self.K, self.N_U, self.N_U)) + np.zeros(shape=(self.K, self.N_U, self.N_U)) * 1j

        # locations (m)
        self.l_BS = np.array([0, 0, 0])  # location of reference antenna of BS-UPA
        self.l_user = np.random.uniform(-30, 30,
                                        size=(self.time_slots, self.K, 3))  # generate random locations of user
        self.l_user[:, :, 2] = -self.BS_z

        d_fac = 1
        k1_traj = [0.038 / d_fac, 0.694 / d_fac, 0.155 / d_fac, 0.625 / d_fac, 0.264 / d_fac, 0.561 / d_fac,
                   0.373 / d_fac, 0.5 / d_fac, 0.495 / d_fac, 0.428 / d_fac]

        d_fac2 = 1
        k2_traj = [0.672 / d_fac2, 0.638 / d_fac2, 0.692 / d_fac2, 0.549 / d_fac2, 0.709 / d_fac2,
                   0.462 / d_fac2, 0.726 / d_fac2, 0.367 / d_fac2, 0.743 / d_fac2, 0.291 / d_fac2]

        if self.K == 1:
            d_fac = 1
            k1_traj = [0.038 / d_fac, 0.694 / d_fac, 0.155 / d_fac, 0.625 / d_fac, 0.264 / d_fac, 0.561 / d_fac,
                       0.373 / d_fac, 0.5 / d_fac, 0.495 / d_fac, 0.428 / d_fac]

            l_user_xy = np.array(k1_traj)[0:self.time_slots * 2]
        else:
            l_user_xy = np.array([k1_traj,
                                  k2_traj])[:, 0:self.time_slots * 2]
        l_user_xy = l_user_xy.reshape(self.K, self.time_slots, 2).transpose(1, 0, 2)
        self.l_user[:, :, 0:2] = l_user_xy * 30

        # CSI matrix
        self.H = np.random.normal(scale=1, size=(self.K, self.N_U, self.N_BS)) + np.random.normal(scale=1, size=(self.K, self.N_U, self.N_BS)) * 1j
        self.all_H = (np.random.normal(scale=1, size=(self.time_slots, self.K, self.N_U, self.N_BS)) +
                      np.random.normal(scale=1, size=(self.time_slots, self.K, self.N_U, self.N_BS)) * 1j)

        # coordinates of antennas
        self.d_c = 1 / 2 * self.lambda_c
        self.Nv = 2
        self.Nh = self.N_BS // self.Nv
        self.Nv_ = 2
        self.Nh_ = self.N_U // self.Nv_
        self.p_hv = np.zeros(shape=(self.Nh, self.Nv, 3))
        self.q_t_k_hv = np.zeros(shape=(self.time_slots, self.K, self.Nh_, self.Nv_, 3))

        # calculate coordinates
        for h in range(self.Nh):
            for v in range(self.Nv):
                self.p_hv[h, v, :] = np.array([h* self.d_c, 0, v * self.d_c])

        for h_ in range(self.Nh_):
            for v_ in range(self.Nv_):
                self.q_t_k_hv[:, :, h_, v_, :] = np.array([h_ * self.d_c, 0, v_ * self.d_c]) + self.l_user

        self.p_BS = self.p_hv.reshape((self.N_BS, 3))
        self.q_U = self.q_t_k_hv.reshape((self.time_slots, self.K, self.N_U, 3))

        # calculate all CSI
        # distance
        self.dist_BS_U = np.zeros(shape=(self.time_slots, self.K))
        self.dist_p_q = np.zeros(shape=(self.time_slots, self.K, self.N_U, self.N_BS))

        for t in range(self.time_slots):
            for k in range(self.K):
                self.dist_BS_U[t, k] = np.linalg.norm(self.l_user[t, k, :] - self.l_BS)

        for t in range(self.time_slots):
            for i in range(self.N_BS):
                for j in range(self.N_U):
                    for k in range(self.K):
                        self.dist_p_q[t, k, j, i] = np.linalg.norm(self.p_BS[i] - self.q_U[t, k, j])

        # channel gain
        self.g = self.light_speed / (4 * np.pi * self.fc * self.dist_BS_U)

        for t in range(self.time_slots):
            for k in range(self.K):
                self.all_H[t, k, :, :] = self.g[t, k] * np.exp(
                    -1j * 2 * np.pi * self.fc / self.light_speed * self.dist_p_q[t, k, :, :])

        # rayleigh distance
        self.aperture_BS = np.sqrt((self.d_c * self.Nv) ** 2 + (self.d_c * self.Nh) ** 2)
        self.aperture_U = np.sqrt((self.d_c * self.Nv_) ** 2 + (self.d_c * self.Nh_) ** 2)
        self.r_dis = 2 * (self.aperture_BS + self.aperture_U) ** 2 / self.lambda_c

        # define action and state spaces
        self.action_dim = 2 * self.N_RF * self.S_d + self.N_BS * self.N_RF
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

        self.state_dim = 2 * self.K * self.N_U * self.N_BS
        self.observation_space = spaces.Box(low=-50, high=50, shape=(self.state_dim,), dtype=np.float32)

    def get_csi(self):
        self.H = self.all_H[self.t]

    def get_state(self):
        # state = np.ndarray([self.state_dim, ])
        H_vec = self.H.ravel()
        H_state = np.append(np.real(H_vec), np.imag(H_vec))
        state = H_state
        return state

    def step(self, action):
        # baseband beamformer
        New_W_beta = (action[0: self.N_RF * self.S_d] + 1) / 2
        New_W_beta = New_W_beta / (sum(New_W_beta) + self.d_epsilon)
        New_W_beta = New_W_beta.reshape(self.N_RF, self.S_d)

        New_W_theta_pi = action[self.N_RF * self.S_d: 2 * self.N_RF * self.S_d] * np.pi
        New_W_theta_pi = New_W_theta_pi.reshape(self.N_RF, self.S_d)

        New_W = np.cos(New_W_theta_pi) * New_W_beta + np.sin(New_W_theta_pi) * New_W_beta * 1j

        # analog beamformer
        New_P_theta_pi = action[2 * self.N_RF * self.S_d:] * np.pi
        New_P = np.cos(New_P_theta_pi) + np.sin(New_P_theta_pi) * 1j  # calculate coefficient
        P_mat = np.reshape(New_P, (self.N_BS, self.N_RF))
        P_mat_H = P_mat.conj().T

        # W_ to satisfy power constraint
        New_W_ = np.sqrt(self.P_max) * New_W / (np.linalg.norm(P_mat @ New_W) + self.d_epsilon)

        eig = np.zeros(shape=(self.K, self.N_U))
        power_consumed = np.zeros(self.K)

        # calculate data rate
        rate_all = np.zeros(self.K)
        for k in range(self.K):
            W_k_mat = New_W_[:,k * self.s_k: (k+1) * self.s_k]
            H_k = self.H[k, :, :]
            H_k_H = H_k.conj().T
            precoding_mat = P_mat @ W_k_mat
            precoding_mat_H = precoding_mat.conj().T
            power_consumed[k] = np.linalg.norm(precoding_mat) ** 2
            signal_power = H_k @ precoding_mat @ precoding_mat_H @ H_k_H
            interference_power = self.noise_power * (np.eye(self.N_U) + 0j)

            for m in range(self.K):
                if not m == k:
                    W_m_mat = New_W_[:, m * self.s_k: (m+1) * self.s_k]
                    W_m_mat_H = W_m_mat.conj().T
                    interference_m = H_k @ P_mat @ W_m_mat @ W_m_mat_H @ P_mat_H @ H_k_H
                    interference_power += interference_m

            sinr_k = signal_power @ np.linalg.inv(interference_power)
            eig[k] = np.real(np.linalg.eig(sinr_k)[0])
            self.sinr[k, :, :] = sinr_k

            data_rate_k = math.log2(np.real(np.linalg.det(np.eye(self.N_U) + sinr_k)))  # calculate data rate
            rate_all[k] = data_rate_k

        # QoS constraint
        qos_satisfied = 0
        for k in range(self.K):
            if rate_all[k] >= self.rate_qos:
                qos_satisfied += 1
        QoS_penalty = (self.K - qos_satisfied) * 1

        # sum up data rate
        sum_rate = sum(rate_all)

        reward = sum_rate - QoS_penalty

        self.t += 1
        if self.t == self.time_slots:  # judge if done
            done = True
            self.t -= 1
        else:
            # update CSI and state
            done = False

        self.get_csi()
        next_state = self.get_state()

        # Optionally we can pass additional info
        info = [rate_all, New_W_, P_mat, power_consumed, eig]

        return np.array(next_state).astype(np.float32), reward, done, info

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent at the right of the grid
        self.t = 0
        self.get_csi()
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        # self.sinr = np.zeros(shape=(self.K, self.N_U, self.N_U)) + np.zeros(shape=(self.K, self.N_U, self.N_U)) * 1j
        state = self.get_state()
        return np.array(state).astype(np.float32)

if __name__ == '__main__':
    env = Env_NF()

