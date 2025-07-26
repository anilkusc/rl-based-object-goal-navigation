import habitat
from omegaconf import OmegaConf

def init_config(split, max_episode_steps=3, shuffle=True):
    config = habitat.get_config(
        config_path="/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch.yaml"
    )
    # Defrost config to make it modifiable
    OmegaConf.set_readonly(config, False)
    config.habitat.dataset.data_path = f"data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz"
    config.habitat.dataset.scenes_dir = "data/scene_datasets/hm3d_v0.2"
    
    # Other overrides
    config.habitat.environment.max_episode_steps = max_episode_steps  # Increased for RL setting
    config.habitat.environment.iterator_options.shuffle = shuffle
    return config