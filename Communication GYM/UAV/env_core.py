import numpy as np
import copy
import math
import gym
from gym import spaces
np.random.seed(42)

from sklearn.cluster import KMeans
# 1. physical world settings


ServiceZone_X = 500  # environment design
ServiceZone_Y = 500
ServiceHeight_Z = 100

NumberofUAV = 2   # UAV design
UAV_Speed = 5  # m/s
UAV_power_unit = 100  # 100mW = 20 dBm

NumberofUser = 4    # user design
NumberofComUser = 4
NumberofSenUser = NumberofUser-NumberofComUser
User_Speed = 0.5  # m/s

# 2. ISAC system settings

M = 4 #antenna number
Fc = 2  # GHz
Bandwidth = 1 # kHz
noise_power = 10**(-11) # mW
Maxstep = 100


# adjustable parameter
Threshold = 1e-06
punish =1


class EnvCore(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):


        '''initialization about physical world'''
        # initialize area
        self.Zone_border_X = ServiceZone_X
        self.Zone_border_Y = ServiceZone_Y
        self.Zone_border_Z = ServiceHeight_Z

        # initialize UAVs
        self.UAV_number = NumberofUAV
        self.UAV_idx = np.arange(self.UAV_number)
        self.UAV_speed = UAV_Speed
        self.UAV_position = np.zeros((3, self.UAV_number))
        self.UAV_position[0, :] = np.array([0, 0])
        self.UAV_position[1, :] = np.array([0, 0])
        self.UAV_position[2, :] = np.array([0, 0])

        # initialize Users
        self.User_number = NumberofUser
        self.User_idx = np.arange(self.User_number)
        self.User_speed = User_Speed
        self.ComUser_number = NumberofComUser
        self.User_position = np.zeros((3, self.User_number))
        self.User_position[0, :] = np.array([0, 10, 490, 500])
        self.User_position[1, :] = np.array([500,490, 10,0])
        self.User_position[2, :] = 0

        '''initialization about communication system'''
        self.Antenna_number = M
        self.Power_unit = UAV_power_unit
        self.Noise_power = noise_power
        self.Beamformer = np.random.normal(scale=1, size=(
        self.UAV_number, self.ComUser_number, self.Antenna_number)) + np.random.normal(scale=1, size=(
        self.UAV_number, self.ComUser_number, self.Antenna_number)) * 1j
        self.User_association = np.array([[1,1,0,0],[0,0,1,1],])
        self.Distance = np.zeros((self.UAV_number, self.User_number), dtype=float)
        self.Distance = self.calculate_distance()
        self.Channel_estimation = np.random.normal(scale=1,size=(self.UAV_number, self.User_number, self.Antenna_number)) \
                                  + np.random.normal(scale=1,size=(self.UAV_number, self.User_number, self.Antenna_number)) * 1j


        '''items about RL'''
        self.t = 1
        self.T = Maxstep
        self.Data_rate = np.zeros(self.ComUser_number)
        self.agent_num = NumberofUAV
        self.observation_space = spaces.Box(low=-1, high=+1,
                shape=((self.ComUser_number * self.Antenna_number * 2 + 3)*self.UAV_number,),dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(( (self.User_number * self.Antenna_number) *2+3)*self.UAV_number,), dtype=np.float32)
        self.User_association_init = copy.deepcopy(self.User_association)
        self.UAV_position_init = copy.deepcopy(self.UAV_position)
        self.User_position_init = copy.deepcopy(self.User_position)
        self.Beamformer_init = copy.deepcopy(self.Beamformer)
        self.Distance_init = copy.deepcopy(self.Distance)

    def calculate_distance(self):
        for i in range(self.UAV_number):
            for j in range(self.User_number):
                self.Distance[i, j] = np.linalg.norm(self.UAV_position[:, i] - self.User_position[:, j])
        return self.Distance

    def user_roaming(self):
        self.User_position[[0,1], :] += np.random.randn(2, self.User_number) * self.User_speed
        self.User_position[0, :][self.User_position[0, :] > self.Zone_border_X] = self.Zone_border_X
        self.User_position[0, :][self.User_position[0, :] < 0] = 0
        self.User_position[1, :][self.User_position[1, :] > self.Zone_border_Y] = self.Zone_border_Y
        self.User_position[1, :][self.User_position[1, :] < 0] = 0
        return


    def calculate_data_rate(self):
        alpha = 10**(-6) # -60dB
        for i in range(self.UAV_number):
            for j in range(self.ComUser_number):
                dx = self.UAV_position[0, i] - self.User_position[0, j]
                dy = self.UAV_position[1, i] - self.User_position[1, j]
                dz = self.UAV_position[2, i] - self.User_position[2, j]
                distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                self.Distance[i, j] = distance
                cos_phi = dz / distance
                phase_shifts = np.pi * np.arange(self.Antenna_number) * cos_phi
                array_response = np.exp(1j * phase_shifts)
                self.Channel_estimation[i, j] = (np.sqrt(alpha) / distance) * array_response

        for c in range(self.ComUser_number):
            uav = np.argmax(self.User_association[:, c]) #user association
            signal_power = abs(np.dot(self.Channel_estimation[uav, c], self.Beamformer[uav, c].T))**2
            intra_interference = 0 - abs(np.dot(self.Channel_estimation[uav, c], self.Beamformer[uav, c].T))**2
            for c_ in np.where(self.User_association[uav,:] == 1)[0][0:2]:
                intra_interference += abs(np.dot(self.Channel_estimation[uav, c], self.Beamformer[uav, c_].T))**2
            inter_interference = 0
            for u_e in range(self.UAV_number):
                if u_e == uav:
                    continue
                for c_e in np.where(self.User_association[u_e, :] == 1)[0][0:2]:
                    inter_interference += abs(np.dot(self.Channel_estimation[u_e, c], self.Beamformer[u_e, c_e].T))**2
            SINR = signal_power / (intra_interference + inter_interference + self.Noise_power)
            self.Data_rate[c] = Bandwidth * math.log((1 + SINR), 2)
        return self.Data_rate

    def get_state(self):
        channel_estimination_array = self.Channel_estimation.ravel()
        channel_estimination_state = np.append(np.real(channel_estimination_array), np.imag(channel_estimination_array))
        uav_position = self.UAV_position
        self.State = np.append(channel_estimination_state, uav_position.ravel())
        return self.State



    def reset(self):
        self.User_association = copy.deepcopy(self.User_association_init)
        self.UAV_position = copy.deepcopy(self.UAV_position_init)
        self.User_position = copy.deepcopy(self.User_position_init)
        self.Beamformer = copy.deepcopy(self.Beamformer_init)
        self.Distance = copy.deepcopy(self.Distance_init)
        self.calculate_data_rate()
        self.t = 0


        obs = np.array(self.get_state()).astype(np.float32)

        return obs

    def step(self, actions):
        action = np.array(actions).reshape(-1)

        # action1 : beamformer
        Beamformer_real = action[0:self.UAV_number * self.ComUser_number * self.Antenna_number] * math.sqrt(self.Power_unit)
        Beamformer_imag = action[self.UAV_number * self.ComUser_number * self.Antenna_number:self.UAV_number *
                                self.ComUser_number * self.Antenna_number * 2] * math.sqrt(self.Power_unit)

        for u in range(self.UAV_number):
            total_power = np.sum(Beamformer_real[u * self.ComUser_number * self.Antenna_number:(u + 1) * self.ComUser_number * self.Antenna_number] ** 2) + \
                          np.sum(Beamformer_imag[u * self.ComUser_number * self.Antenna_number:(u + 1) * self.ComUser_number * self.Antenna_number] ** 2)
            # print(total_power)
            if total_power > self.Power_unit:
                Beamformer_real[u * self.ComUser_number * self.Antenna_number:(u + 1) * self.ComUser_number * self.Antenna_number] /= math.sqrt(
                    (total_power / self.Power_unit))
                Beamformer_imag[u * self.ComUser_number * self.Antenna_number:(u + 1) * self.ComUser_number * self.Antenna_number] /= math.sqrt(
                    (total_power / self.Power_unit))

        self.Beamformer = (Beamformer_real + Beamformer_imag * 1j).reshape(
            (self.UAV_number, self.ComUser_number, self.Antenna_number))

        # action2 : UAV trajectory design  3*3
        trajectory = action[self.UAV_number * self.ComUser_number * self.Antenna_number * 2:
                            self.UAV_number * self.ComUser_number * self.Antenna_number * 2 + 3 * self.UAV_number]

        for u in range(self.UAV_number):
            self.UAV_position[0, u] += (trajectory[3 * u] ) * self.UAV_speed
            self.UAV_position[1, u] += (trajectory[3 * u + 1] ) * self.UAV_speed
            self.UAV_position[2, u] += (trajectory[3 * u + 2] ) * self.UAV_speed
            if self.UAV_position[0, u] > self.Zone_border_X:
                self.UAV_position[0, u] = self.Zone_border_X
            elif self.UAV_position[0, u] < 0:
                self.UAV_position[0, u] = 0
            if self.UAV_position[1, u] > self.Zone_border_Y:
                self.UAV_position[1, u] = self.Zone_border_Y
            elif self.UAV_position[1, u] < 0:
                self.UAV_position[1, u] = 0
            if self.UAV_position[2, u] > self.Zone_border_Z:
                self.UAV_position[2, u] = self.Zone_border_Z
            elif self.UAV_position[2, u] < 100:
                self.UAV_position[2, u] = 100

        self.user_roaming()
        self.calculate_distance()
        self.calculate_data_rate()


        self.t += 1  # next time step
        if self.t == (self.T):  # judge if done
            done = True
        else:
            done = False


        obs=np.array(self.get_state()).astype(np.float32)
        reward = np.sum(self.Data_rate)

        return obs, reward, done, {}
