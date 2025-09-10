import os

import git
import torch
from gym import spaces
from omegaconf import OmegaConf

from PIL import Image

import habitat
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config
from helpers.print import print_episode_info, print_step_info, print_episode_summary, print_training_summary
from helpers.cfg import init_config
from agent.agent import Agent
from helpers.visualize import save_rgb_observation_to_png,create_gif_from_pngs,save_depth_observation_to_png

if __name__ == "__main__":
    # GPU kontrolü
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    max_episode_steps = 500
    epoch = 100
    config = init_config(max_episode_steps=max_episode_steps,split="val_mini")
    
    print("Initializing environment...")
    env = habitat.Env(config=config)
    print("Environment initialized!")
    
    # Check available episodes
    #print(f"\nTotal episodes in dataset: {len(env.episodes)}")
    #print("Available episode IDs:")
    #for i, episode in enumerate(env.episodes):
    #    print(f"Index {i}: Episode ID {episode.episode_id}, Scene: {episode.scene_id}, Object: {episode.object_category}")
    
    total_rewards = []
    """
    rgb after cnn: 512
    depth after cnn: 512
    compass: 1
    gps: 2
    objectgoal: 1
    total = 1028
    """
    agent = Agent(goal_category=None,state_dim=1028,action_dim=2)
    agent.load("outputs/last.pt")
    for e in range(epoch):
        for i, episode in enumerate(env.episodes):
            obs = env.reset()
            current_episode = env.current_episode
            if current_episode.scene_id != "data/scene_datasets/hm3d_v0.2/hm3d_v0.2/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb" or current_episode.episode_id != "8":
                continue
            log_probs, values, rewards, states, actions = [], [], [], [], []
            done = False
            step = 0
            agent.goal_category=obs['objectgoal']
            episode_reward=0
            while not done and step < max_episode_steps:
                # Select action (currently random)
                action, log_prob = agent.action_selector(obs)
                action_np = action.squeeze().detach().cpu().numpy()
                # Scale the actions to reasonable velocities
                linear_vel = action_np[0] * 0.5  # -0.5 to 0.5 m/s (was -1 to 1)
                angular_vel = action_np[1] * 1.0  # -1.0 to 1.0 rad/s (was -1 to 1)
                print(f"Action: {action_np}, Scaled: linear={linear_vel:.3f}, angular={angular_vel:.3f}")
                next_obs = env.step(action = {"action": "velocity_control","action_args": {"linear_velocity": linear_vel,"angular_velocity": angular_vel}})
                # Get episode info
                done = env.episode_over
                info = env.get_metrics()
                reward = agent.calculate_reward(info,done)
                obs = next_obs
                episode_reward += reward
                if step % 10 == 0:
                    save_rgb_observation_to_png(obs["rgb"],output_path="outputs/episode_"+str(episode.episode_id),filename=str(step)+"_rgb.png")
                    pass
                step += 1
            total_rewards.append(episode_reward)


    env.close()
    agent.close_tensorboard()  # Close TensorBoard writer
    print("\nEnvironment closed!")