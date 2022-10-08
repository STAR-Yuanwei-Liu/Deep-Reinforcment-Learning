from gym import spaces
# import Paras
import copy
import math
import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import gym

from gym import spaces
import copy
import math
import os
import glob
import time
from datetime import datetime






"""
STAR-RIS environment for DRL agent training 
Author: Ruikang Zhong 
E-mail: r.zhong@qmul.ac.uk
"""
class STAR_RIS_Env(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self):
        super(STAR_RIS_Env, self).__init__()
        self.KR = 3 # Number of reflection users
        self.KT = 3 # Number of transmission users
        self.K = self.KR + self.KT # total users
        self.M = 4  # antenna number at BS
        self.N = 10 # the number of STAR-elements
        self.Nx = 2 # the number of element per row
        self.B = 1  # Mhz
        self.fc = 2 #Ghz
        self.T = 50 #max_steps
        self.power_unit = 100 # 最大发射功率 100mW,26dbm
        self.noise_power = 3* 10 ** (-13) # noise power
        self.Sum_rate = 0
        self.power_consumption = 0
        self.unsatisfied_user = 0
        self.W_list = np.ones(shape=(self.M, self.K)) + 0 * 1j
        self.scale = 10000 # a coefficient for scaling the value of CSI

        #ndarray for saving CSI at each fading block t, randomly initialized, BS to KR, BS to KT, STAR-RIS to KR, STAR-RIS to KT, and BS to STAR-RIS
        self.H_B_KR = np.random.normal(scale=1, size=(self.M, self.KR)) + np.random.normal(scale=1, size=(self.M, self.KR)) * 1j
        self.H_B_KT = np.random.normal(scale=1, size=(self.M, self.KT)) + np.random.normal(scale=1, size=(self.M, self.KT)) * 1j
        self.H_R_KR = np.random.normal(scale=1, size=(self.N, self.KR)) + np.random.normal(scale=1, size=(self.N, self.KR)) * 1j
        self.H_R_KT = np.random.normal(scale=1, size=(self.N, self.KT)) + np.random.normal(scale=1, size=(self.N, self.KT)) * 1j
        self.H_B_R = np.random.normal(scale=1, size=(self.N, self.M)) + np.random.normal(scale=1, size=(self.N, self.M)) * 1j

        # random fading matrix for all time slots
        self.G_B_KR = np.random.normal(scale=1, size=(self.M, self.KR, self.T)) + np.random.normal(scale=1, size=(self.M, self.KR, self.T)) * 1j
        self.G_B_KT = np.random.normal(scale=1, size=(self.M, self.KT, self.T)) + np.random.normal(scale=1, size=(self.M, self.KT, self.T)) * 1j
        self.G_R_KR = np.random.normal(scale=1, size=(self.N, self.KR, self.T)) + np.random.normal(scale=1, size=(self.N, self.KR, self.T)) * 1j
        self.G_R_KT = np.random.normal(scale=1, size=(self.N, self.KT, self.T)) + np.random.normal(scale=1, size=(self.N, self.KT, self.T)) * 1j
        self.G_B_R = np.random.normal(scale=1, size=(self.N, self.M, self.T)) + np.random.normal(scale=1, size=(self.N, self.M, self.T)) * 1j
        # fading intensity
        self.fading_scale_BS = 0.1
        self.fading_scale_RIS = 0.2
        self.G_B_KR = self.fading_scale_BS * self.G_B_KR
        self.G_B_KT = self.fading_scale_BS * self.G_B_KT
        self.G_R_KR = self.fading_scale_RIS  * self.G_R_KR
        self.G_R_KT = self.fading_scale_RIS  * self.G_R_KT
        self.G_B_R = self.fading_scale_RIS  * self.G_B_R
        self.Rice = 5 # Rician factor

        # postions
        self.P_BS = np.array([2000, 2000, 15])
        self.P_R = np.array([0, 0, 5])
        self.P_KR_list = np.random.normal(scale=3, size=(3, self.KR)) + 3
        self.P_KT_list = np.random.normal(scale=3, size=(3, self.KT)) - 3
        self.P_KR_list[2, :] = 0 #R user hight
        self.P_KT_list[2, :] = 0 #T user hight
        self.P_KR_list_initial = copy.deepcopy(self.P_KR_list)
        self.P_KT_list_initial = copy.deepcopy(self.P_KT_list)
        self.t = 0

        # create vectors for storing the R/T coefficients
        self.theta_R = np.random.normal(scale=1, size=(self.N)) + np.random.normal(scale=1, size=(self.N)) * 1j
        self.theta_T = np.random.normal(scale=1, size=(self.N)) + np.random.normal(scale=1, size=(self.N)) * 1j
        self.Theta_eye_R = np.eye(self.N)
        self.Theta_eye_T = np.eye(self.N)
        self.data_rate_list_R = np.zeros(self.KR)
        self.data_rate_list_T = np.zeros(self.KT)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        self.action_dim = 3 * self.N + 2 * self.M * self.K
        # self.action_num = 4
        # self.action_array = np.ones(self.action_dim) * self.action_num
        # self.action_space = spaces.MultiDiscrete(self.action_array)
        self.action_space = spaces.Box(low=0, high=1,
                                        shape=(self.action_dim,), dtype=np.float32)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.num_states = 2*self.M * self.K + 2*self.N * self.K + 2*self.M*self.N
        self.observation_space = spaces.Box(low=0, high=100,
                                        shape=(self.num_states,), dtype=np.float32)


    def calculate_CSI(self):
        # BS to RIS
        d_B_R = np.linalg.norm(self.P_BS - self.P_R) # distance
        PL_B_R_line = 10 ** (-30 / 10) * (d_B_R ** (-2.2)) # calculate pathloss
        # AOA/AOD
        azimuth_B_R_varphi_A = (self.P_R[1] - self.P_BS[1]) / np.sqrt((self.P_R[0] - self.P_BS[0]) ** 2 + (self.P_R[1] - self.P_BS[1]) ** 2)
        azimuth_B_R_varphi_D = np.sqrt(1-azimuth_B_R_varphi_A**2)
        elevation_B_R_psi_A = (self.P_R[2] - self.P_BS[2]) / np.sqrt((self.P_R[0] - self.P_BS[0]) ** 2 + (self.P_R[1] - self.P_BS[1]) ** 2)
        elevation_B_R_psi_A_cos = np.sqrt(1-elevation_B_R_psi_A**2)
        a_R = np.ones(self.N) + 0*1j
        a_B = np.ones(self.M) + 0*1j

        # assuming da/lamda =1, DOA, AOA
        for m in range(self.M):
            real = np.cos(2*math.pi * (m - 1 + 1) * azimuth_B_R_varphi_D)  # +1是因为计算机从0开始计数而element从1开始
            imag = np.sin(2*math.pi * (m - 1 + 1) * azimuth_B_R_varphi_D)
            a_B[m] = (real + 1j * imag)

        for n in range(self.N):
            real = np.cos(2*math.pi * (n - 1 + 1) * (int(n/self.Nx)*azimuth_B_R_varphi_A*elevation_B_R_psi_A+(n-int(n/self.Nx)*self.Nx)*elevation_B_R_psi_A*elevation_B_R_psi_A_cos))  # +1是因为计算机从0开始计数而element从1开始
            imag = np.sin(2*math.pi * (n - 1 + 1) * (int(n/self.Nx)*azimuth_B_R_varphi_A*elevation_B_R_psi_A+(n-int(n/self.Nx)*self.Nx)*elevation_B_R_psi_A*elevation_B_R_psi_A_cos))
            a_R[n] =  (real + 1j * imag)

        a_R = np.mat(a_R).conj().T
        a_B = np.mat(a_B)
        L_B_R_LOS = a_R * a_B
        L_B_R_NLOS = self.G_B_R[:,:,self.t]
        self.H_B_R = np.sqrt(PL_B_R_line) * (np.sqrt(self.Rice / (1 + self.Rice)) * L_B_R_LOS + np.sqrt(1 / (1 + self.Rice)) * L_B_R_NLOS)

        # channel for R users
        for k_R in range(self.KR):
            # positions and distance
            self.P_KR = self.P_KR_list[:, k_R].T
            d_B_KR = np.linalg.norm(self.P_KR - self.P_BS)
            d_R_KR = np.linalg.norm(self.P_KR - self.P_R)
            if d_B_KR < 1:
                d_B_KR = 1
            if d_R_KR < 1:
                d_R_KR = 1

            # path loss
            PL_B_KR_line= 10**(-30/10)*(d_B_KR**(-3.5)) #alpha
            PL_R_KR_line = 10**(-30/10)*(d_R_KR**(-2.2))
            # find the BS/RIS to R users channel
            G_B_KR = self.G_B_KR[:, k_R, self.t]
            self.H_B_KR[:, k_R] = np.sqrt(PL_B_KR_line) * G_B_KR
            G_R_KR = self.G_R_KR[:, k_R, self.t]
            self.H_R_KR[:, k_R] = np.sqrt(PL_R_KR_line) * G_R_KR

        # channel for T users
        for k_T in range(self.KR):
            # positions and distance
            self.P_KT = self.P_KT_list[:, k_T].T
            d_B_KT = np.linalg.norm(self.P_KT - self.P_BS)
            d_R_KT = np.linalg.norm(self.P_KT - self.P_R)
            if d_B_KT<1:
                d_B_KT = 1
            if d_R_KT<1:
                d_R_KT = 1

            # path loss
            PL_B_KT_line= 10**(-30/10)*(d_B_KT**(-3.5))
            PL_R_KT_line = 10**(-30/10)*(d_R_KT**(-2.2))
            # find the BS/RIS to T users channel
            G_B_KT = self.G_B_KT[:,k_T,self.t]
            self.H_B_KT[:,k_T] = np.sqrt(PL_B_KT_line) * G_B_KT
            G_R_KT = self.G_R_KT[:,k_T,self.t]
            self.H_R_KT[:,k_T] = np.sqrt(PL_R_KT_line) * G_R_KT


    def calculate_data_rate(self):

        self.calculate_CSI() # renew CSI

        for k_R in range(self.KR):
            # Computing cascaded channels
            H_R_KR_H = self.H_R_KR[:,k_R].conj().T
            h_mid = np.dot(H_R_KR_H, self.Theta_eye_R)
            # direct_link = self.H_B_KR[:,k_R]
            # RIS_link = np.dot(h_mid, self.H_B_R)
            h_k_R = self.H_B_KR[:,k_R] + np.dot(h_mid, self.H_B_R) #
            Signal_power_k_R = abs(np.dot(h_k_R,self.W_list[:,k_R].T))**2 # calculate signal power
            Interference_power_k_R = 0 - abs(np.dot(h_k_R,self.W_list[:,k_R].T))**2 # calculate interference power
            for j_R in range(self.K):
                Interference_power_k_R += abs(np.dot(h_k_R,self.W_list[:,j_R].T))**2

            SINR_k_R = Signal_power_k_R / (Interference_power_k_R+self.noise_power)
            data_rate_k_R = self.B * math.log((1 + SINR_k_R), 2) # calculate data rate
            self.data_rate_list_R[k_R] = data_rate_k_R # saving data rate

        # calculate data rate for T users following the same way for R users
        for k_T in range(self.KR):
            H_R_KT_H = self.H_R_KT[:,k_T].conj().T
            h_mid = np.dot(H_R_KT_H, self.Theta_eye_T)
            direct_link = self.H_B_KR[:,k_R]
            RIS_link = np.dot(h_mid, self.H_B_R)
            h_k_T = self.H_B_KT[:,k_T] + np.dot(h_mid, self.H_B_R)
            Signal_power_k_T = abs(np.dot(h_k_T,self.W_list[:,self.KR+k_T].T))**2
            Interference_power_k_T = 0 - abs(np.dot(h_k_T,self.W_list[:,k_T].T))**2
            for j_T in range(self.K):
                Interference_power_k_T += abs(np.dot(h_k_T,self.W_list[:,j_T].T))**2

            SINR_k_T = Signal_power_k_T / (Interference_power_k_T+self.noise_power)
            data_rate_k_T = self.B * math.log((1 + SINR_k_T), 2)
            self.data_rate_list_T[k_T] = data_rate_k_T


    def user_move(self):
        # perform random movement for users
        self.P_KR_list[0, :] = self.P_KR_list[0, :] + np.random.normal(scale=0.5, size=(1, self.KR))
        self.P_KR_list[1, :] = self.P_KR_list[1, :] + np.random.normal(scale=0.5, size=(1, self.KR))
        self.P_KT_list[0, :] = self.P_KT_list[0, :] + np.random.normal(scale=0.5, size=(1, self.KT))
        self.P_KT_list[1, :] = self.P_KT_list[1, :] + np.random.normal(scale=0.5, size=(1, self.KT))


    def get_state(self):
        # 复数输入
        state = np.ndarray([self.num_states, ])

        H_B_KR_array = self.H_B_KR.ravel()
        H_B_KT_array = self.H_B_KT.ravel()
        H_R_KR_array = self.H_R_KR .ravel()
        H_R_KT_array = self.H_R_KT.ravel()
        H_B_R_array = self.H_B_R.ravel()
        H_B_R_array = np.array(H_B_R_array)

        H_B_KR_state = np.append(np.real(H_B_KR_array), np.imag(H_B_KR_array))
        H_B_KT_state = np.append(np.real(H_B_KT_array), np.imag(H_B_KT_array))
        H_R_KR_state = np.append(np.real(H_R_KR_array), np.imag(H_R_KR_array))
        H_R_KT_state = np.append(np.real(H_R_KT_array), np.imag(H_R_KT_array))
        H_B_R_state = np.append(np.real(H_B_R_array), np.imag(H_B_R_array))

        H_B_K_state = np.append(H_B_KR_state, H_B_KT_state)
        H_R_K_state = np.append(H_R_KR_state, H_R_KT_state)

        state[0: 2 * self.N * self.M] = H_B_R_state * self.scale
        state[2 * self.N * self.M : 2 * (self.N*self.M + self.K*self.M)] = H_B_K_state * self.scale
        state[2 * (self.N*self.M + self.K*self.M) : 2 * (self.N*self.M + self.K*self.M + self.N*self.K)] = H_R_K_state * self.scale
        return state

    def step(self,action):
        # calculate R coefficient
        action = action.reshape(-1)
        New_theta_pi_R = action[0:self.N] * math.pi # theta in radian system
        New_beta_R = (action[self.N:2*self.N] +1)/2
        New_theta_R = np.cos(New_theta_pi_R) * New_beta_R + np.sin(New_theta_pi_R) * New_beta_R * 1j # calculate R coefficient

        self.theta_R = New_theta_R
        self.Theta_eye_R = np.eye(self.N) * self.theta_R # transfer R coefficient into matrix

        # determain the T coefficient based on R coefficient
        action_T = copy.deepcopy(action[2*self.N:3*self.N])
        action_T[action_T >= 0] = 1 * math.pi/2
        action_T[action_T >= 0] = 1 * math.pi/2
        New_theta_pi_T = New_theta_pi_R + action_T
        New_beta_T = np.sqrt(1-New_beta_R**2)
        self.Theta_T = np.cos(New_theta_pi_T) * New_beta_T + np.sin(New_theta_pi_T) * New_beta_T * 1j
        self.Theta_eye_T = np.eye(self.N) * self.theta_T

        # BS beamforming w
        w_theta = action[3*self.N:3*self.N+self.M*self.K] * math.pi
        w_beta = (action[3*self.N+self.M*self.K:3*self.N+2*self.M*self.K] + 1)/2 * self.power_unit
        w_array = np.cos(w_theta) * w_beta + np.sin(w_theta) * w_beta * 1j
        self.W_list = np.reshape(w_array, (self.M, self.K))

        self.calculate_data_rate()
        self.Sum_rate = sum(self.data_rate_list_R)+sum(self.data_rate_list_T)
        reward = self.Sum_rate # set sumrate as reward

        self.user_move() # user move
        self.calculate_CSI() # renew CSI
        next_state = self.get_state() # calculate next state
        self.t += 1 # next time step
        if self.t == (self.T): # judge if done
            done = True
        else:
            done = False
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([next_state]).astype(np.float32), reward, done, info,


    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent at the right of the grid
        self.P_KR_list = copy.deepcopy(self.P_KR_list_initial)
        self.P_KT_list = copy.deepcopy(self.P_KT_list_initial)
        self.t = 0
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        state = self.get_state()
        return np.array([state]).astype(np.float32)


    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        # print(self.Sum_rate, end="")


    def close(self):
        pass




if __name__ == '__main__':
    env = STAR_RIS_Env()
    # wrap it
    un_wrapped_env = STAR_RIS_Env()
    # wrapped_env = make_vec_env(lambda: un_wrapped_env, n_envs=1)

