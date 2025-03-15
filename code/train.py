from opendss_env import OpenDSS
import gymnasium as gym
import ray
from ray.tune.registry import register_env
from gymnasium.envs.registration import register
from ray.rllib.algorithms import ppo, sac, dqn
from ray.tune.logger import pretty_print
import numpy as np

import argparse
import pandas as pd
import random
import torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--hour", type=int, help="Chosen hour", required=True)

    args = parser.parse_args()

    chosen_hour = args.hour
    print("Chosen hour:", chosen_hour)

    num_iterations = 1000

    # For reproducibility
    seed = 42
    # Python’s built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ray.init(include_dashboard=False)
    
    register_env('opendss', lambda config: OpenDSS(chosen_hour=chosen_hour))
    
    config_ppo = (ppo.PPOConfig()
                .environment(env="opendss",env_config={"seed": seed})
                .rollouts(num_rollout_workers=8, enable_connectors=False, num_envs_per_worker=1)
                .resources(num_gpus=0, num_cpus_per_worker=1)
                .training(train_batch_size=256 ,sgd_minibatch_size=16,
                          model={"fcnet_hiddens": [32,32]},
                          lr=0.00001)
                )
    
    algo = config_ppo.build()
    
    for i in range(num_iterations):
        result = algo.train()

        learner_stats = result.get('info', {}).get('learner', {}).get('default_policy', {}).get('learner_stats', {})

        total_loss = learner_stats.get('total_loss', 'Not Found')
        policy_loss = learner_stats.get('policy_loss', 'Not Found')
 
        print(f'Iteration: {i:5d} Total Loss: {total_loss:10.8f} Policy Loss: {policy_loss:10.8f}')
        
    path_to_checkpoint = algo.save(checkpoint_dir = f'../results/checkpoints/hour_{chosen_hour}') 
    print(f"Policy is recorded: '../results/checkpoints/hour_{chosen_hour}'")
            

    algo.stop()
    ray.shutdown()