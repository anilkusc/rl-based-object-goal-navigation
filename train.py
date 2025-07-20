import numpy as np
import habitat_sim
from helpers import display_sample_save,make_cfg,set_sim_settings
from data_loader import scene_loader
import random

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.add_argument("--max-frames", dest="max_frames", action="store_false")
    parser.set_defaults(show_video=True, make_video=True,max_frames=200)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    do_make_video = args.make_video
    max_frames= args.max_frames
else:
    show_video = False
    do_make_video = False
    display = False
    max_frames = 200

scenes = scene_loader("data/train/")

for scene in scenes:

    sim_settings = set_sim_settings(scene["basis.glb"])
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    success = sim.pathfinder.load_nav_mesh(scene["navmesh"])
    if not success:
        raise RuntimeError("NavMesh yüklenemedi!")
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    # random in the scene
    valid_positions = sim.pathfinder.get_random_navigable_point()
    agent_state.position = valid_positions
    agent.set_state(agent_state)
    agent_state = agent.get_state()
    print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("Discrete action space: ", action_names)
    total_frames = 0
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    while total_frames < max_frames:
        action = random.choice(action_names)
        print("action", action)
        observations = sim.step(action)
        agent_state = sim.agents[0].get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
        rgb = observations["color_sensor"]
        #semantic = observations["semantic_sensor"]
        depth = observations["depth_sensor"]
        if display:
            if total_frames % 10 == 0:
                basename = scene["basis.glb"].split("/")[2]
                display_sample_save(rgb_obs=rgb,output_dir="outputs/"+basename,file_name=basename+"-"+str(total_frames), depth_obs=depth)
        total_frames += 1
    total_frames = 0
    sim.close()