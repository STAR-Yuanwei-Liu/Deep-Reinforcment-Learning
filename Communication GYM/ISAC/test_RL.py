from ISAC_Env import Env_core
import numpy as np
from stable_baselines3 import A2C
from plot_beam import beam_pattern, plot_beampattern
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


env = Env_core()
model = A2C.load('a2c_8.zip', env=env)


d = 0.5  # Element spacing in wavelengths
Fc = 2.6 * 10**9 #Hz
wavelength = 3*10**8 / Fc # meter
angles = np.linspace(-90, 90, 180)  # Angle range from -90 to 90 degrees

done = False
obs = env.reset()
while not done:
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, info = env.step(action)
    print(info)
    _, beampattern = beam_pattern(env.Beamformer[:,0:2], d,  angles)
    plot_beampattern(angles,beampattern, polar=0)




