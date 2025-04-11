import numpy as np
import matplotlib.pyplot as plt
from more_itertools import chunked
from matplotlib.ticker import MultipleLocator


#------ load data frrm the path ------
path = "results"

"""
You can refer to the plot function in the file TD3_N1.py for plotting the rewards.
Use np.load to load data from the "results" folder after training.
To smooth the curve, you can average 400/etc. rounds of rewards for plotting.
"""