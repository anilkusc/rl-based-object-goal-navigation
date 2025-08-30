
import random
from helpers.cnn import Encoder
import torch
import torch.nn as nn
from helpers.sac import Actor, Critic

class Agent():
    def __init__(self,total_objects,replay_buffer_size=100000,gamma=0.99,tau=0.005,batch_size=32):
        self.encoder = Encoder()
        self.total_objects = total_objects
        self.input_size = self.encoder.total_output_size + 1 + 1 # 1 for compass, 1 for object goal
        self.action_size = 2 # linear velocity and angular velocity
        #self.encoder.load("models/encoder.pth")
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        self.actor = Actor(self.input_size,self.action_size)
        self.critic = Critic(self.input_size,self.action_size)
        self.optimizer = torch.optim.Adam(self.actor.parameters(),lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure all models use float32
        self.actor.to(self.device).float()
        self.critic.to(self.device).float()
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=0.001)
        self.loss_fn = nn.MSELoss()
    def add_to_replay_buffer(self,state,action,reward,next_state,done):
        state = self.concat_input(state)
        next_state = self.concat_input(next_state)
        self.replay_buffer.append((state,action,reward,next_state,done))
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

    def sample_from_replay_buffer(self,batch_size):
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.stack([s1 for (s1, a, r, s2, d) in minibatch]).to(self.device)
        action_batch = torch.stack([a for (s1, a, r, s2, d) in minibatch]).to(self.device)
        reward_batch = torch.tensor([r for (s1, a, r, s2, d) in minibatch], dtype=torch.float32).to(self.device)
        next_state_batch = torch.stack([s2 for (s1, a, r, s2, d) in minibatch]).to(self.device)
        done_batch = torch.tensor([1 if d else 0 for (s1, a, r, s2, d) in minibatch], dtype=torch.float32).to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def train(self):
        # Only train if we have enough samples in the replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return
        state_batch,action_batch,reward_batch,next_state_batch,done_batch = self.sample_from_replay_buffer(self.batch_size)

        current_q_value = self.critic(state_batch,action_batch)
        current_q_value = current_q_value.to(self.device)
        with torch.no_grad():
            next_q_values = self.critic(next_state_batch,action_batch).max(1)[0]
            next_q_values = next_q_values.to(self.device)
            target_q_value = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        input("20 Press Enter to continue...")
        loss = self.loss_fn(current_q_value, target_q_value)
        #loss = self.critic(state_batch,action_batch) - self.critic(next_state_batch,action_batch) + reward_batch * (~done_batch).float()
        input("30 Press Enter to continue...")
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        input("4 Press Enter to continue...")

    def calculate_reward(self,info,done,obs):
        #print(info.keys()) #dict_keys(['distance_to_goal', 'success', 'spl', 'soft_spl', 'num_steps', 'collisions', 'distance_to_goal_reward'])
        #print(obs.keys()) # dict_keys(['rgb', 'depth', 'objectgoal', 'compass', 'gps'])
        reward = 0
        if done:
            if info.get('success', False):
                reward = 10  # Success reward
            else:
                reward = -1  # Failure penalty
        else:
            reward = -0.01  # Small penalty for each step
        return reward

    def concat_input(self,obs):
        fuse = self.encoder.process(obs["rgb"], obs["depth"])
        compass = torch.tensor([[obs["compass"][0]]], dtype=torch.float32)  # Ensure float32 dtype
        object_goal = torch.tensor([[obs["objectgoal"][0] / self.total_objects]], dtype=torch.float32)  # Ensure float32 dtype
        agent_input = torch.cat([fuse, compass, object_goal], dim=-1)
        return agent_input

    def action_selector(self,obs):
        #linear_velocity = random.uniform(-1.0, 1.0)
        #angular_velocity = random.uniform(-1.0, 1.0)
        agent_input = self.concat_input(obs)
        action, _ = self.actor.forward(agent_input.to(self.device))
        action = action.cpu().squeeze(0).detach()  # Remove the batch dimension and detach gradients
        return action