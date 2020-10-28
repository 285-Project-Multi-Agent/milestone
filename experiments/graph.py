import pickle
import matplotlib.pyplot as plt
r = pickle.load(open("./learning_curves/baselineRun_rewards.pkl", "rb"))

checkpoints = 540
size = 1000
plt.plot([(1 + i) * 1000 for i in range(checkpoints)], [r[int(i * len(r) / checkpoints)] for i in range(checkpoints)])
plt.plot([(1 + i) * 1000 for i in range(checkpoints)], [r[int(1 + i * len(r) / checkpoints)] for i in range(checkpoints)])
plt.plot([(1 + i) * 1000 for i in range(checkpoints)], [r[int(2 + i * len(r) / checkpoints)] for i in range(checkpoints)])
# plt.plot([(1 + i) * 1000 for i in range(checkpoints)], [r[i] for i in range(len(r))])
plt.show()
