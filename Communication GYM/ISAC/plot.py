import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

with open('store_4.pkl', 'rb') as f1:
    vector1 = pickle.load(f1)

x1 = list(range(len(vector1)))
plt.figure(figsize=(10, 6))
plt.plot([i*10 for i in x1], vector1, linestyle='-', marker='o',color='b',markerfacecolor='none', markeredgecolor='b',)
plt.xlabel('Episodes')
plt.ylabel('Reward')


plt.legend()


plt.grid(True)
plt.show()