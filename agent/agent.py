from .models import Actor, Critic, Encoder
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
import torch
class Agent():
    def __init__(self,goal_category,state_dim,action_dim,gamma=0.99,lam=0.95,lr_actor=3e-4,lr_critic=1e-3,eps_clip = 0.2,lr_rgb_encoder=3e-4,lr_depth_encoder=3e-4):
        self.goal_category = goal_category
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.rgb_encoder = Encoder()
        self.depth_encoder = Encoder()
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.current_actor_loss = None
        self.current_critic_loss = None
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.rgb_encoder_optimizer = optim.Adam(self.rgb_encoder.parameters(), lr=lr_rgb_encoder)
        self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=lr_depth_encoder)
    
    def calculate_reward(self,info,done):
        #Info: {'distance_to_goal': 2.3431520462036133, 'success': 0.0, 'spl': 0.0, 'soft_spl': 0.05283481905361087, 'num_steps': 15, 'collisions': {'count': 0, 'is_collision': False}, 'distance_to_goal_reward': 0.011035680770874023}
        dist_reward = -info['distance_to_goal']  # distance küçüldükçe reward artar
        dist_reward *= 1.0  # ağırlık
        # 2. Success reward: hedefe ulaşıldığında büyük ödül
        success_reward = 10.0 if info['success'] > 0 else 0.0
        # 3. Step penalty: kısa yolları teşvik
        step_penalty = -0.01 * info['num_steps']
        # 4. Collision penalty
        collision_penalty = -0.2 * info['collisions']['count']
        # 5. Distance to goal bonus (ortamdan gelen küçük ek sinyal)
        distance_goal_bonus = info.get('distance_to_goal_reward', 0.0)
        # 6. Done penalization (opsiyonel)
        # Eğer episode başarısız ve done = True ise ekstra ceza
        done_penalty = -1.0 if done and info['success'] == 0 else 0.0
        # Toplam reward
        reward = dist_reward + success_reward + step_penalty + collision_penalty + distance_goal_bonus + done_penalty
        return reward

    def action_selector(self,obs):
        #linear_velocity = random.uniform(-1.0, 1.0)
        #angular_velocity = random.uniform(-1.0, 1.0)
        #print(f"Goal category: {self.goal_category}")
        #print(f"Obs: {obs}")
        #linear_velocity = 0.5
        #angular_velocity = 0.5
        #return linear_velocity, angular_velocity
        state = self.process_state(obs)
        action, log_prob = self.actor.act(state)
        return action, log_prob

    def critic_selector(self,obs):
        state = self.process_state(obs)
        return self.critic(state)

    def compute_returns_advantages(self,rewards, values):
        returns, advs = [], []
        gae = 0
        next_value = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            delta = r + self.gamma * next_value - v
            gae = delta + self.gamma * self.lam * gae
            advs.insert(0, gae)
            returns.insert(0, gae + v)
            next_value = v
        return returns, advs

    def critic_loss(self,returns, values,states_tensor):
        values_pred = self.critic(states_tensor).squeeze()
        return F.mse_loss(values_pred, returns)
    
    def actor_loss(self,states_tensor,actions_tensor,old_log_probs_tensor,advs):
        mu, std = self.actor(states_tensor)
        dist = D.Normal(mu, std)
        log_probs_new = dist.log_prob(actions_tensor).sum(-1)
        ratio = torch.exp(log_probs_new - old_log_probs_tensor)
        obj1 = ratio * advs
        obj2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advs
        actor_loss = -torch.min(obj1, obj2).mean()
        return actor_loss

    def optimize_actor(self):
        self.actor_optimizer.zero_grad()
        self.current_actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

    def optimize_critic(self):
        self.critic_optimizer.zero_grad()
        self.current_critic_loss.backward()
        self.critic_optimizer.step()

    def optimize_rgb_encoder(self):
        pass

    def optimize_depth_encoder(self):
        pass

    def calculate_advantage_returns(self,rewards,values,states,actions,log_probs):
        returns, advs = self.compute_returns_advantages(rewards, values)
        returns = torch.tensor(returns, dtype=torch.float32)
        advs = torch.tensor(advs, dtype=torch.float32)

        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        old_log_probs_tensor = torch.stack(log_probs)
        return returns, advs, states_tensor, actions_tensor, old_log_probs_tensor

    def process_state(self,obs):
        gray_frame = obs["rgb"].mean(axis=-1, keepdims=True)
        rgb_frame = torch.from_numpy(gray_frame).float()  # (H,W,C)
        depth_frame = torch.from_numpy(obs["depth"]).float()  # (H,W,1)

        rgb_frame = rgb_frame.unsqueeze(0)   # (1,H,W,C)
        depth_frame = depth_frame.unsqueeze(0)
        rgb_feature = self.rgb_encoder(rgb_frame)
        depth_feature = self.depth_encoder(depth_frame)
        compass = torch.from_numpy(obs["compass"]).float().unsqueeze(0)     # (1,3)
        gps = torch.from_numpy(obs["gps"]).float().unsqueeze(0)             # (1,2)
        objectgoal = torch.from_numpy(obs["objectgoal"]).float().unsqueeze(0) # (1,num_object_classes)

        state = torch.cat([rgb_feature, depth_feature, compass, gps, objectgoal], dim=1)
        return state

    def save(self):
        pass
    def load(self):
        pass