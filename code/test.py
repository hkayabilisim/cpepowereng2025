from opendss_env import OpenDSS
import gymnasium as gym
import ray
from ray.tune.registry import register_env
from gymnasium.envs.registration import register
from ray.rllib.algorithms import ppo, sac, dqn
from ray.tune.logger import pretty_print
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
import matplotlib.pyplot as plt

import pandas as pd
import random
import argparse
import json

parser = argparse.ArgumentParser(description="Testing script")
parser.add_argument("--training_hour", type=int, help="Training hour", required=True)
parser.add_argument("--testing_hour", type=int, help="Testing hour", required=True)

args = parser.parse_args()

training_hour = args.training_hour
testing_hour = args.testing_hour

print("Training hour:", training_hour)
print("Testing hour:", testing_hour)


ray.init(include_dashboard=False)

register_env('opendss', lambda config: OpenDSS(chosen_hour=testing_hour))

policy_name = f"../data/checkpoints/hour_{training_hour}"
algo = Algorithm.from_checkpoint(policy_name)

env = OpenDSS()

obs, info = env.reset()
done = False
truncated = False
sum_reward = 0

step = 0

while step<10:
    action = algo.compute_single_action(obs)
    next_obs, reward, done, truncated, info= env.step(action)
    sum_reward += reward
    if np.equal(obs, next_obs).all():
        print(f"Converged at step {step}")
        break
    obs = next_obs
    step += 1

result = {'training_hour': training_hour, 
          'testing_hour': testing_hour, 
          'converged_at': step,
          'reward': reward,
          'action': action.tolist(),
          'next_obs': next_obs.tolist() }

for key, val in info.items():
    result[key] = val
result['p1_std'] = np.std(result['p1'])
result['p2_std'] = np.std(result['p2'])
result['p3_std'] = np.std(result['p3'])

out_file = f"../results/training_{training_hour}_testing_{testing_hour}.txt"
with open(out_file, "w") as file:
    for k, v in result.items():
        print(f"{k:30s}: {v}", file=file)
print(f"Result is recorded: {out_file}")


plt.figure(figsize=(12, 6))
plt.plot(result['p1'], label='Phase A', marker='o', markersize=3, linewidth=2)
plt.plot(result['p2'], label='Phase B', marker='o', markersize=3, linewidth=2)
plt.plot(result['p3'], label='Phase C', marker='o', markersize=3, linewidth=2)

plt.ylabel('Voltage magnitude (p.u.)', fontsize=14)
plt.xlabel('Node Number', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14) 

plt.legend()
plt.grid()

out_pdf = f"../results/training_{training_hour}_testing_{testing_hour}.pdf"
plt.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")




