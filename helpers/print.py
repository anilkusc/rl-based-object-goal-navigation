
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
