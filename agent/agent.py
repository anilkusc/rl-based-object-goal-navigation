from .models import Actor, Critic, Encoder
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
import torch
class Agent():
    def __init__(self,goal_category,state_dim,action_dim,gamma=0.99,lam=0.95,lr_actor=3e-4,lr_critic=1e-3,eps_clip = 0.2,lr_encoder=1e-4):
        self.goal_category = goal_category
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.rgb_encoder = Encoder()
        self.depth_encoder = Encoder()
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.reward_max = None
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.encoder_optimizer = torch.optim.Adam(list(self.rgb_encoder.parameters()) + list(self.depth_encoder.parameters()),lr=lr_encoder)
    
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

    def critic_loss(self,returns, states_tensor):
        values_pred = self.critic(states_tensor).squeeze()
        cl = F.mse_loss(values_pred, returns)
        return cl
    
    def actor_loss(self,states_tensor,actions_tensor,old_log_probs_tensor,advs):
        mu, std = self.actor(states_tensor)
        dist = D.Normal(mu, std)
        log_probs_new = dist.log_prob(actions_tensor).sum(-1)
        ratio = torch.exp(log_probs_new - old_log_probs_tensor)
        obj1 = ratio * advs
        obj2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advs
        al = -torch.min(obj1, obj2).mean()
        return al

    def optimize_models(self,rewards,values,states,actions,log_probs):
        returns, advs, states_tensor, actions_tensor, old_log_probs_tensor = self.calculate_advantage_returns(rewards,values,states,actions,log_probs)
        al = self.actor_loss(states_tensor,actions_tensor,old_log_probs_tensor,advs)
        cl = self.critic_loss(returns, states_tensor)
        total_loss = al + cl

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        total_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.encoder_optimizer.step()

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

    def save(self, total_reward,filepath="./outputs/"):
        """
        Save all model weights, optimizers, and training state to a file.
        
        Args:
            filepath (str): Path where to save the checkpoint
        """
        checkpoint = {
            # Model states
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'rgb_encoder_state_dict': self.rgb_encoder.state_dict(),
            'depth_encoder_state_dict': self.depth_encoder.state_dict(),
            
            # Optimizer states
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            
            # Training parameters
            'gamma': self.gamma,
            'lam': self.lam,
            'eps_clip': self.eps_clip,
            
        }
        torch.save(checkpoint, filepath+"last.pt")
        if self.reward_max is None or total_reward > self.reward_max:
            torch.save(total_reward, filepath+"best.pt")
        print(f"Model saved successfully to {filepath}")
    
    def load(self, filepath):
        """
        Load all model weights, optimizers, and training state from a file.
        
        Args:
            filepath (str): Path to the checkpoint file
        """
        checkpoint = torch.load(filepath)
        
        # Load model states
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.rgb_encoder.load_state_dict(checkpoint['rgb_encoder_state_dict'])
        self.depth_encoder.load_state_dict(checkpoint['depth_encoder_state_dict'])
        
        # Load optimizer states
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        
        # Load training parameters
        self.gamma = checkpoint['gamma']
        self.lam = checkpoint['lam']
        self.eps_clip = checkpoint['eps_clip']
        
        
        print(f"Model loaded successfully from {filepath}")