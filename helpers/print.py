import numpy as np

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

def print_step_info(step,action,reward,obs,done,info):
    print(f"\nStep {step}:")
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"Object goal: {obs['objectgoal']}")
    print(f"Done: {done}")
    print(f"Info: {info}")

def print_episode_summary(episode_id,episode_reward,metrics,step):
    print(f"\nEpisode {episode_id} finished after {step} steps")
    print(f"Total reward: {episode_reward}")
    print(f"Episode metrics: {metrics}")

def print_training_summary(total_rewards,episodes):
    print("\n=== Training Summary ===")
    print(f"Average reward over {len(episodes)} episodes: {np.mean(total_rewards):.2f}")
    print(f"Standard deviation of rewards: {np.std(total_rewards):.2f}")
    print(f"Min reward: {min(total_rewards):.2f}")
    print(f"Max reward: {max(total_rewards):.2f}")