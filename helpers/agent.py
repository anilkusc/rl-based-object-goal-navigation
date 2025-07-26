
import random

class Agent():
    def __init__(self):
        pass
    def calculate_reward(self,info,done,obs):
        reward = 0
        if done:
            if info.get('success', False):
                reward = 10  # Success reward
            else:
                reward = -1  # Failure penalty
        else:
            reward = -0.01  # Small penalty for each step
        return reward

    def action_selector(self,obs):
        #linear_velocity = random.uniform(-1.0, 1.0)
        #angular_velocity = random.uniform(-1.0, 1.0)
        linear_velocity = 0.5
        angular_velocity = 0.5

        return {
            "action": "velocity_control",
            "action_args": {
                "linear_velocity": linear_velocity,
                "angular_velocity": angular_velocity
            }
        }
    def save(self):
        pass
    def load(self):
        pass