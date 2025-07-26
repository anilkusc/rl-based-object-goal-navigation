import os
import random

import git
import numpy as np
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


def create_action():
    """Create a random action for the agent"""
    # Random linear and angular velocities
    linear_velocity = random.uniform(-1.0, 1.0)
    angular_velocity = random.uniform(-1.0, 1.0)
    
    return {
        "action": "velocity_control",
        "action_args": {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity
        }
    }


def print_episode_info(env):
    """Print detailed episode information from config"""
    episode = env.current_episode
    print(f"\n=== Episode Information ===")
    print(f"Episode ID: {episode.episode_id}")
    
    # Extract scene name from scene_id
    scene_id = episode.scene_id
    if "/" in scene_id:
        scene_name = scene_id.split("/")[-2]  # Get the scene folder name
    else:
        scene_name = scene_id
    
    print(f"Scene: {scene_name}")
    print(f"Scene ID: {episode.scene_id}")
    print(f"Object Category: {episode.object_category}")
    print(f"Start Position: {episode.start_position}")
    print(f"Start Rotation: {episode.start_rotation}")
    print(f"Start Room: {episode.start_room}")


def run_episode(env, episode, max_steps=100):
    """Run a single episode with specific episode"""
    # Reset environment
    obs = env.reset()
    print(f"\nStarting Episode {episode.episode_id}")
    
    # Print episode information from config
    print_episode_info(env)
    
    input("Press Enter to continue...")
    print(f"Initial object goal: {obs['objectgoal']}")
    
    episode_reward = 0
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Select action (currently random)
        action = create_action()
        
        # Take action in environment
        obs = env.step(action)
        
        # Get episode info
        done = env.episode_over
        info = env.get_metrics()
        
        # Calculate reward (you can customize this)
        reward = 0
        if done:
            if info.get('success', False):
                reward = 10  # Success reward
            else:
                reward = -1  # Failure penalty
        else:
            reward = -0.01  # Small penalty for each step
        
        # Accumulate reward
        episode_reward += reward
        
        # Print step information
        print(f"\nStep {step}:")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Object goal: {obs['objectgoal']}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        step += 1
    
    # Episode summary
    print(f"\nEpisode {episode.episode_id} finished after {step} steps")
    print(f"Total reward: {episode_reward}")
    metrics = env.get_metrics()
    print(f"Episode metrics: {metrics}")
    
    return episode_reward, metrics


if __name__ == "__main__":
    config = habitat.get_config(
        config_path="/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch.yaml"
    )
    
    # Defrost config to make it modifiable
    OmegaConf.set_readonly(config, False)
    
    # Override config to use multiple scenes
    #config.habitat.dataset.content_scenes = ["TEEsavR23oF","wcojb4TFT35"]  # Using two scenes
    config.habitat.dataset.data_path = "data/datasets/objectnav/hm3d/v2/val_mini/val_mini.json.gz"
    config.habitat.dataset.scenes_dir = "data/scene_datasets/hm3d_v0.2"
    
    # Other overrides
    config.habitat.environment.max_episode_steps = 3  # Increased for RL setting
    config.habitat.environment.iterator_options.shuffle = True  # Enable shuffling for training

    print("Initializing environment...")
    env = habitat.Env(config=config)
    print("Environment initialized!")
    
    # Check available episodes
    print(f"\nTotal episodes in dataset: {len(env.episodes)}")
    print("Available episode IDs:")
    for i, episode in enumerate(env.episodes):
        print(f"Index {i}: Episode ID {episode.episode_id}, Scene: {episode.scene_id}, Object: {episode.object_category}")
    
    total_rewards = []
    
    try:
        for i, episode in enumerate(env.episodes):
            print(f"\nEpisode {i} (ID: {episode.episode_id}) started")
            episode_reward, metrics = run_episode(env, episode)
            total_rewards.append(episode_reward)
            
        # Print training summary
        print("\n=== Training Summary ===")
        print(f"Average reward over {len(env.episodes)} episodes: {np.mean(total_rewards):.2f}")
        print(f"Standard deviation of rewards: {np.std(total_rewards):.2f}")
        print(f"Min reward: {min(total_rewards):.2f}")
        print(f"Max reward: {max(total_rewards):.2f}")

        
    finally:
        env.close()
        print("\nEnvironment closed!")


