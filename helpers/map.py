import habitat
import numpy as np
import matplotlib.pyplot as plt
from typing import cast
from habitat.utils.visualizations import maps

def get_topdown_map(env, filename="top_down_map.png"):
    import os
    save_path = os.path.join("outputs", filename)
    top_down_map = maps.get_topdown_map_from_sim(cast("HabitatSim", env.sim), map_resolution=1024)
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    plt.imsave(save_path, top_down_map)