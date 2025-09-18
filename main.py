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
    
    max_episode_steps = 20000
    epoch = 1000
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
    agent = Agent(goal_category=None,state_dim=516,action_dim=2,lr_actor=3e-4,lr_critic=3e-4,epsilon=0.8,epsilon_min=0.1,epsilon_decay=0.995)
    for e in range(epoch):
        for i, episode in enumerate(env.episodes):
            obs = env.reset()
            ep = env.current_episode
            if not (ep.scene_id == "data/scene_datasets/hm3d_v0.2/hm3d_v0.2/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb" and ep.episode_id == "2"):
                continue
            values, rewards, states, actions = [], [], [], []
            print(f"\nStarting Episode {episode.episode_id}")
            # Print episode information from config
            print_episode_info(env)
            episode_reward = 0
            done = False
            step = 0
            agent.goal_category=obs['objectgoal']
            agent.reset_episode()  # Reset episode-specific state
            while not done and step < max_episode_steps:
                # Select action (currently random)
                action = agent.action_selector(obs)
                value = agent.critic_selector(obs)
                action_np = action.squeeze().detach().cpu().numpy()
                # Scale the actions to reasonable velocities (linear: 0-1, angular: -1 to 1)
                next_obs = env.step(action = {"action": "velocity_control","action_args": {"linear_velocity": action_np[0],"angular_velocity": action_np[1]}})
                info = env.get_metrics()
                info["success"] = 1 if info["distance_to_goal"] < 0.2 else 0
                done = True if info["distance_to_goal"] < 0.2 else False

                reward = agent.calculate_reward(info,done)
                states.append(agent.process_state(obs))
                actions.append(action)  # action zaten GPU'da
                values.append(value.item())
                rewards.append(reward)
                obs = next_obs
                episode_reward += reward
                # Print step information
                #print_step_info(step,action,reward,obs,done,info)
                #if step % 10 == 0:
                #    save_rgb_observation_to_png(obs["rgb"],output_path="outputs/episode_"+str(episode.episode_id),filename=str(step)+"_rgb.png")
                    #save_depth_observation_to_png(obs["depth"],output_path="outputs/episode_"+str(episode.episode_id),filename=str(step)+"_depth.png")
                #    pass
                step += 1
                if step % 500 == 0:
                    print("#######################################################")
                    print(info)
                    print(action_np)
                    print(value)
                    print(obs["compass"])
                    print(obs["gps"])
                    print("#######################################################")
                    actor_loss, critic_loss, total_loss = agent.optimize_models(rewards, values, states, actions)
                    values, rewards, states, actions = [], [], [], []
                    torch.cuda.empty_cache()

                #print(f"Step: {step}, Action: {action},Log prob: {log_prob},Value: {value},Reward: {reward}")
            # Get losses from optimization and log to TensorBoard
            if len(values) > 0:
                actor_loss, critic_loss, total_loss = agent.optimize_models(rewards, values, states, actions)
            
            metrics = env.get_metrics()
            agent.log_to_tensorboard(episode_reward, actor_loss, critic_loss, total_loss, step, rewards, metrics)
            agent.save(episode_reward)
            # Decay epsilon after each episode
            agent.decay_epsilon()
            print(f"Episode {episode.episode_id} finished with total reward {episode_reward}")
            # Episode summary
            #print_episode_summary(episode.episode_id,episode_reward,metrics,step)
            total_rewards.append(episode_reward)
            #input("Press Enter to continue...")
        # Log final training summary
        #agent.log_final_summary(total_rewards)
        # Print training summary
        #print_training_summary(total_rewards,env.episodes)
        print(f"=======================================================Epoch {e} finished=========================================================================")


    env.close()
    agent.close_tensorboard()  # Close TensorBoard writer
    print("\nEnvironment closed!")