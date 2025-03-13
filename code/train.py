from opendss_env import OpenDSS
import gymnasium as gym
import ray
from ray.tune.registry import register_env
from gymnasium.envs.registration import register
from ray.rllib.algorithms import ppo, sac, dqn
from ray.tune.logger import pretty_print
import numpy as np

import pandas as pd


if __name__ == '__main__':
    num_iterations = 2
    evaluating_interval = 2
    evaluation_duration = 4
    save_interval = 100
    
    ray.init(include_dashboard=False)
    
    register_env('opendss', lambda config: OpenDSS())
    
    test = OpenDSS()
    # config_sac = (sac.SACConfig()
    #             .environment(env="opendss")   
    #             .rollouts(num_rollout_workers=8, enable_connectors=False, num_envs_per_worker=1)
    #             .resources(num_gpus=number_of_gpus, num_cpus_per_worker=1)
    #             .training(train_batch_size=256,
    #                       model={"fcnet_hiddens": [32,32]},
    #                       lr=0.00001) 
    #     # .callbacks(MetricCallbacks)
    # )
    
    # config_dqn = (dqn.DQNConfig()
    #             .environment(env="opendss")        
    #             .rollouts(num_rollout_workers=8, enable_connectors=False, num_envs_per_worker=1)
    #             .resources(num_gpus=number_of_gpus, num_cpus_per_worker=1)
    #             .training(train_batch_size=256,
    #                       model={"fcnet_hiddens": [32,32]}) 
    #     # .callbacks(MetricCallbacks)
    # )
    
    config_ppo = (ppo.PPOConfig()
                .environment(env="opendss")
                .env_runners(num_env_runners=4, num_cpus_per_env_runner=1)
                .training(train_batch_size=256,
                          model={"fcnet_hiddens": [32,32]},
                          lr=0.00001)
                .evaluation(evaluation_interval=evaluating_interval, evaluation_duration = evaluation_duration)
        # .callbacks(MetricCallbacks)
        )
    
    algo = config_ppo.build()
    logdir = algo.logdir
    for i in range(num_iterations):
        print("------------- Iteration", i+1, "-------------")
        result = algo.train()
        if((i+1) % save_interval) == 0:
            path_to_checkpoint = algo.save(checkpoint_dir = logdir) 
            print("----- Checkpoint -----")
            print(f"An Algorithm checkpoint has been created inside directory: {path_to_checkpoint}.")
            
        
        learner_stats = result.get('info', {}).get('learner', {}).get('default_policy', {}).get('learner_stats', {})

        total_loss = learner_stats.get('total_loss', 'Not Found')
        policy_loss = learner_stats.get('policy_loss', 'Not Found')
 
        print("Total Loss:", total_loss)
        print("Policy Loss:", policy_loss)

    algo.stop()
    ray.shutdown()