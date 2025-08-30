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

def run_episode(env,agent, episode, max_steps=100):
    """Run a single episode with specific episode"""
    # Reset environment
    obs = env.reset()
    log_probs, values, rewards, states, actions = [], [], [], [], []
    print(f"\nStarting Episode {episode.episode_id}")
    
    # Print episode information from config
    print_episode_info(env)
    
    episode_reward = 0
    done = False
    step = 0

    agent.goal_category=obs['objectgoal']
    
    while not done and step < max_steps:
        # Select action (currently random)

        action, log_prob = agent.action_selector(obs)
        value = agent.critic_selector(obs)
        action_np = action.squeeze().cpu().numpy()
        next_obs = env.step(action = {"action": "velocity_control","action_args": {"linear_velocity": action_np[0],"angular_velocity": action_np[1]}})
        # Get episode info
        done = env.episode_over
        info = env.get_metrics()
        reward = agent.calculate_reward(info,done)
        states.append(agent.process_state(obs))
        actions.append(torch.tensor(action))
        log_probs.append(log_prob)
        values.append(value.item())
        rewards.append(reward)
        obs = next_obs
        episode_reward += reward
        # Print step information
        #print_step_info(step,action,reward,obs,done,info)
        #if step % 100 == 0:
        #    save_rgb_observation_to_png(obs["rgb"],output_path="outputs/episode_"+str(episode.episode_id),filename=str(step)+"_rgb.png")
        #    #save_depth_observation_to_png(obs["depth"],output_path="outputs/episode_"+str(episode.episode_id),filename=str(step)+"_depth.png")
        #    pass
        step += 1
        print(f"Step: {step}, Action: {action},Log prob: {log_prob},Value: {value},Reward: {reward}")

    returns, advs, states_tensor, actions_tensor, old_log_probs_tensor = agent.calculate_advantage_returns(rewards,values,states,actions,log_probs)
    print(f"Returns: {len(returns)}, Advs: {len(advs)}, States tensor: {states_tensor.shape}, Actions tensor: {actions_tensor.shape}, Old log probs tensor: {old_log_probs_tensor.shape}")
    agent.current_actor_loss = agent.actor_loss(states_tensor,actions_tensor,old_log_probs_tensor,advs)
    print(f"Actor loss: {agent.current_actor_loss}")
    agent.current_critic_loss = agent.critic_loss(returns, values,states_tensor)
    print(f"Critic loss: {agent.current_critic_loss}")
    input("Press Enter to continue...")
    agent.optimize_actor()
    agent.optimize_critic()

    metrics = env.get_metrics()
    # Episode summary
    print_episode_summary(episode.episode_id,episode_reward,metrics,step)
    #create_gif_from_pngs(output_gif_path="outputs/episode_"+str(episode.episode_id)+".gif",png_directory="outputs/episode_"+str(episode.episode_id))
    #agent.train()
    input("1Press Enter to continue...")
    return episode_reward, metrics,agent


if __name__ == "__main__":
    config = init_config(max_episode_steps=500,split="val_mini")
    
    print("Initializing environment...")
    env = habitat.Env(config=config)
    print("Environment initialized!")
    
    # Check available episodes
    print(f"\nTotal episodes in dataset: {len(env.episodes)}")
    print("Available episode IDs:")
    for i, episode in enumerate(env.episodes):
        print(f"Index {i}: Episode ID {episode.episode_id}, Scene: {episode.scene_id}, Object: {episode.object_category}")
    
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
    try:
        for i, episode in enumerate(env.episodes):
            print(f"\nEpisode {i} (ID: {episode.episode_id}) started")
            episode_reward, metrics,agent = run_episode(env,agent, episode)
            total_rewards.append(episode_reward)
            
        # Print training summary
        print_training_summary(total_rewards,env.episodes)

    finally:
        env.close()
        print("\nEnvironment closed!")