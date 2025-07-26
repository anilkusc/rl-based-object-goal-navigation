import os

import git

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
from helpers.agent import Agent

def run_episode(env, episode, max_steps=100):
    """Run a single episode with specific episode"""
    # Reset environment
    obs = env.reset()
    print(f"\nStarting Episode {episode.episode_id}")
    
    # Print episode information from config
    print_episode_info(env)
    
    #input("Press Enter to continue...")
    print(f"Initial object goal: {obs['objectgoal']}")
    
    episode_reward = 0
    done = False
    step = 0
    agent = Agent()
    while not done and step < max_steps:
        # Select action (currently random)
        action = agent.action_selector(obs)
        
        # Take action in environment
        obs = env.step(action)
        
        # Get episode info
        done = env.episode_over
        info = env.get_metrics()
        
        # Calculate reward (you can customize this)
        reward = agent.calculate_reward(info,done,obs)
        
        # Accumulate reward
        episode_reward += reward
        
        # Print step information
        print_step_info(step,action,reward,obs,done,info)
        step += 1
    metrics = env.get_metrics()
    # Episode summary
    print_episode_summary(episode.episode_id,episode_reward,metrics,step)
    
    return episode_reward, metrics


if __name__ == "__main__":
    config = init_config(split="val_mini")
    
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
        print_training_summary(total_rewards,env.episodes)

    finally:
        env.close()
        print("\nEnvironment closed!")