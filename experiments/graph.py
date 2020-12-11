import pickle
import matplotlib.pyplot as plt
import os

files = ['behavior_analysis_2_no_persistent_agRewards.pkl', 'behavior_analysis_2_no_persistent_Rewards.pkl', 'behavior_analysis_2_proximity_20_agRewards.pkl', 'behavior_analysis_2_proximity_20_Rewards.pkl', 'behavior_analysis_2_proximity_fraction_final_agRewards.pkl', 'behavior_analysis_2_proximity_fraction_final_Rewards.pkl', 'ddpg_for_adversary_agRewards.pkl', 'ddpg_for_adversary_Rewards.pkl', 'ddpg_for_agents_agRewards.pkl', 'ddpg_for_agents_Rewards.pkl', 'ddpg_for_both_agRewards.pkl', 'ddpg_for_both_Rewards.pkl', 'kill_cooldown_no_persistent_agRewards.pkl', 'kill_cooldown_no_persistent_Rewards.pkl', 'behavior_analysis_1_Rewards.pkl']

mean_Rewards = [pickle.load(open("./learning_curves/{}".format(path), "rb")) for path in files if 'agRewards' not in path]
titles = [path for path in files if 'agRewards' not in path]

base_images = "C:\\Users\\TheDonut\\Documents\\Berkeley\\285\\project\\final_paper_images\\"

proximity = [mean_Rewards[i] for i in range(len(mean_Rewards)) if i in [1, 2]]
x = [1000 * i for i in range(10)]
baseline = mean_Rewards[-1]

def normalize(data):
    smallest = min(data)
    ret = []
    scale = 100 / (max(data) - smallest)
    for i in data:
        ret.append(scale * (i - smallest))
    return ret

def get_image_path(file_name):
    return base_images + file_name + ".png"

plt.figure(figsize=(12,9))
plt.plot(x, normalize(baseline), label="Baseline")
plt.plot(x, normalize(proximity[0]), label="Proximity Rewards increased")
plt.plot(x, normalize(proximity[1]), label="Proximity Rewards decreased")
plt.legend()
plt.ylabel("Rewards (Scaled)")
plt.xlabel("Iterations")
plt.title("Effects of Scaling Proximity Rewards")
plt.savefig(get_image_path("Proximity_Scaling"))

plt.show()

plt.figure(figsize=(12,9))
plt.plot(x, baseline, label="Baseline")
plt.plot(x, proximity[0], label="Proximity Rewards increased")
plt.plot(x, proximity[1], label="Proximity Rewards decreased")
plt.legend()
plt.ylabel("Rewards")
plt.xlabel("Iterations")
plt.title("Effects of Scaling Proximity Rewards")
plt.savefig(get_image_path("Proximity_Scaling_Raw"))

plt.show()

plt.figure(figsize=(12,9))
plt.plot(x, baseline, label="Baseline (Persistent Reward)")
plt.plot(x, mean_Rewards[0], label="One Time Reward")
plt.legend()
plt.ylabel("Rewards (Scaled)")
plt.xlabel("Iterations")
plt.title("Effects of Persistent vs One Time Reward")
plt.savefig(get_image_path("Persistent_Reward"))

plt.show()


plt.figure(figsize=(12,9))
plt.plot(x, baseline, label="Baseline (maddpg for both)")
plt.plot(x, mean_Rewards[3], label="ddpg for impostors, maddpg for crewmates")
plt.plot(x, mean_Rewards[4], label="ddpg for crewmates, maddpg for impostors")
plt.plot(x, mean_Rewards[5], label="ddpg for both")
plt.legend()
plt.ylabel("Rewards (Scaled)")
plt.xlabel("Iterations")
plt.title("DDPG vs MADDPG Performance")
plt.savefig(get_image_path("DDPG_vs_MADDPG"))

plt.show()

plt.figure(figsize=(12,9))
plt.plot(x, normalize(baseline), label="Baseline (Persistent Reward)")
plt.plot(x, normalize(mean_Rewards[-2]), label="Kill Cooldown")
plt.legend()
plt.ylabel("Rewards (Scaled)")
plt.xlabel("Iterations")
plt.title("Effect of Kill Cooldown (Scaled)")
plt.savefig(get_image_path("Kill_Cooldown_Scaled"))

plt.show()
