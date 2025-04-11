import copy
import numpy as np
import math
import gym
from gym import spaces
import torch
from sklearn.preprocessing import StandardScaler
np.random.seed(42)

'''A simple static ISAC scenario where a ULA with K antennas serves M communication users and N sensing users '''

K = 8 # ULA antenna number
M = 2  # communication user number
N = 1  # sensing user number

Fc = 2.6 * 10**9 #Hz
wavelength = 3*10**8 / Fc # meter
Antenna_Separation = wavelength/2
Bandwidth = 1 # Mhz
ULA_Power = 60# 18dBW
Noise_Power = 1e-8# mW
#Beampattern_Threshold= 1e-5  # 1e-5

Maxstep = 100




def normalize_csi_with_scaler(csi_data):
    real_part = csi_data.real
    imag_part = csi_data.imag
    scaler_real = StandardScaler()
    scaler_imag = StandardScaler()
    real_scaled = scaler_real.fit_transform(real_part)
    imag_scaled = scaler_imag.fit_transform(imag_part)
    normalized_csi = real_scaled + 1j * imag_scaled
    return normalized_csi


class Env_core(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self):

        self.com = M
        self.sen = N
        x_values = np.linspace(-200, 200,self.com)
        user_positions = np.zeros((3, self.sen+self.com))
        user_positions[0, :-1] = x_values
        self.user_position = user_positions
        self.antenna = K
        self.ULA_separation = Antenna_Separation
        self.ULA_position = np.zeros((3, self.antenna))
        self.ULA_position[0,:] = np.linspace((-self.antenna//2)*self.ULA_separation, (-self.antenna//2 + self.antenna)*self.ULA_separation, self.antenna)
        self.ULA_position[2,:] = 200

        self.power = ULA_Power
        self.noise_power = Noise_Power
        self.Distance = np.zeros((self.antenna, self.com + self.sen), dtype=float)
        self.calculate_distance()
        self.CSI = np.random.normal(scale=1, size=(self.antenna, self.com )) \
                   + np.random.normal(scale=1, size=(self.antenna, self.com)) * 1j
        self.calculate_CSI()
        self.Beamformer = np.random.normal(scale=1, size=(self.antenna, self.com + self.sen*2)) \
                              + np.random.normal(scale=1, size=(self.antenna, self.com + self.sen*2)) * 1j
        self.Beamformer[:,:self.com + self.sen] = self.power * self.Beamformer[:,:self.com + self.sen]/np.sum(np.abs(self.Beamformer[:,:self.com + self.sen]))
        self.Beamformer[:,self.com + self.sen:] = self.Beamformer[:,self.com + self.sen:] / np.sum(np.abs(self.Beamformer[:,self.com + self.sen:]))


        self.State = np.zeros([1, (self.antenna*(self.com + self.sen*2))*2 + 1],dtype=float)
        self.episodes = 0
        self.t = 0
        self.T = Maxstep -1
        self.reward = 0
        self.Max_datarate = 0
        self.threshold = 5#15db
        self.Data_rate = np.zeros(self.com)
        self.calculate_datarate()
        self.Beam_gain = np.zeros(self.sen)
        self.calculate_beamgain()
        self.Beamformer_init = copy.deepcopy(self.Beamformer)
        self.Total = []

        self.observation_space = spaces.Box(low=-1, high=+1,
                        shape=((self.antenna * (self.com + self.sen)) * 2 + 1,),dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([8] * self.antenna * (self.com + self.sen * 2) * 2)


    def calculate_distance(self):
        for i in range(self.antenna):
            for j in range(self.com+self.sen):
                self.Distance[i, j] = np.linalg.norm(self.ULA_position[:, i] - self.user_position[:, j])
        return

    def calculate_CSI(self):
        path_loss = 1 / (self.Distance[:, :self.com] ** 2 + 1)  # Example path loss model
        fading = (np.random.randn(self.antenna, self.com) + 1j * np.random.randn(self.antenna, self.com)) / np.sqrt(2)
        CSI_com = path_loss * fading
        CSI_sen =  np.sqrt(10**-13) * (np.cos( np.pi *  self.Distance[:, self.com:]/wavelength ) - np.sin(
                2 * np.pi *  self.Distance[:, self.com:]/wavelength ) * 1j)
        self.CSI = np.hstack((CSI_com,CSI_sen))
        return

    def calculate_datarate(self):
        for c in range(self.com):
            signal_power = np.abs(np.dot(self.CSI[:,c].T.conjugate(), self.Beamformer[:,c]))**2
            interference = -signal_power
            for c_ in range(self.com):
                interference+= np.abs(np.dot(self.CSI[:,c].T.conjugate(), self.Beamformer[:,c_]))**2
            interference_2 = 0
            for s in range(self.sen):
                interference_2 += np.abs(np.dot(self.CSI[:,c].T.conjugate(), self.Beamformer[:,self.com+s]))**2
            SINR = signal_power/(self.noise_power + interference + interference_2)
            self.Data_rate[c] = np.array([[Bandwidth *math.log((1+SINR),2)]])
        return

    def calculate_beamgain(self):
        arrayresponse = (np.cos(2 * np.pi * self.Distance[:, self.com:self.com + self.sen] / wavelength) - np.sin(
            2 * np.pi * self.Distance[:, self.com:self.com + self.sen] / wavelength) * 1j)
        A = np.dot(arrayresponse, arrayresponse.T.conjugate())
        Q1 = [np.dot(self.Beamformer[:,v], self.Beamformer[:,v].T.conjugate()) for v in range(self.com)]
        Q2 = [np.dot(self.Beamformer[:,v+self.com], self.Beamformer[:,v+self.com].T.conjugate()) for v in range(self.sen)]
        Q = Q1[0] + Q1[1] +Q2
        HSI =  10**-11
        for s in range(self.sen):
            signal_power = np.sum(np.abs(10**-13 * self.Beamformer[:,self.com+self.sen + s].T.conjugate()*A* Q* self.Beamformer[:,self.com+self.sen + s]))
            interference = HSI + np.sum(np.abs(np.dot(self.Beamformer[:,self.com+self.sen + s].T.conjugate(),self.noise_power))**2)
            SINR = signal_power/interference
        self.Beam_gain =SINR
        return

    def get_state(self):
        CSI = normalize_csi_with_scaler(copy.deepcopy(self.CSI)) #4*1

        self.State = np.concatenate( [np.real(CSI).reshape((self.antenna * (self.com + self.sen))),
                                          np.imag(CSI).reshape((self.antenna * (self.com + self.sen))),
                                          np.array(self.reward).reshape(1)
                                          ],axis=0)
        return self.State

    def reset(self):
        self.Beamformer = copy.deepcopy(self.Beamformer_init)
        self.calculate_datarate()
        self.calculate_beamgain()
        self.t = 0
        state = self.get_state()

        return state

    def step(self, actions):

        mag = actions[:self.antenna * (self.com + self.sen * 2)]
        pha = actions[self.antenna * (self.com + self.sen * 2):]
        beamformer = mag + pha*1j
        self.Beamformer = beamformer.reshape((self.antenna , (self.com + self.sen*2)))
        self.Beamformer[:, :self.com + self.sen] = self.power * self.Beamformer[:,
                        :self.com + self.sen] / np.linalg.norm(self.Beamformer[:, :self.com + self.sen], ord=2)
        self.Beamformer[:, self.com + self.sen:] = self.Beamformer[:, self.com + self.sen:] / np.linalg.norm(
                        self.Beamformer[:, self.com + self.sen:], ord=2)

        datarate = np.sum(self.Data_rate)
        if datarate > self.Max_datarate:
            self.Max_datarate = datarate

        self.calculate_datarate()
        self.calculate_beamgain()
        next_state = self.get_state()

        if self.t == (self.T):
            done = True
            self.Max_datarate = 0
        else:
            done = False
        self.t += 1


        if np.sum(self.Data_rate) >= 0.9*self.Max_datarate and self.Beam_gain >= self.threshold and np.all(self.Data_rate > 0.3): # 0.3 for 4 antenna
            self.reward = np.sum(self.Data_rate)
        elif np.sum(self.Data_rate) < 0.9*self.Max_datarate and self.Beam_gain >= self.threshold and np.all(self.Data_rate > 0.3):
            self.reward = 0
        else:
            self.reward = -1
        return next_state, self.reward, done, {}

    def render(self):
        pass

    def close(self):
        pass


env= Env_core()
print(env.Data_rate, env.Beam_gain)
