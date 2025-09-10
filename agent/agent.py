from .models import Actor, Critic, ResNetEncoder
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
import torch
import os
import pynvml
import random
from torch.utils.tensorboard import SummaryWriter

class Agent():
    def __init__(self,goal_category,state_dim,action_dim,gamma=0.99,lam=0.95,lr_actor=1e-4,lr_critic=1e-4,eps_clip = 0.2,lr_encoder=1e-4,epsilon=0.9,epsilon_min=0.1,epsilon_decay=0.999):
        self.goal_category = goal_category
        
        # Exploration parameters
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate
        self.exploration_noise_std = 0.5  # Standard deviation for exploration noise (increased from 0.3)
        
        # GPU kontrolü ve cihaz seçimi
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name()}")
        
        # Modelleri GPU'ya taşı
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.rgb_encoder = ResNetEncoder(output_dim=512, pretrained=True).to(self.device)
        #self.depth_encoder = ResNetEncoder(output_dim=512, pretrained=True).to(self.device)
        
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.reward_max = None
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        #self.encoder_optimizer = torch.optim.Adam(list(self.rgb_encoder.parameters()) + list(self.depth_encoder.parameters()),lr=lr_encoder)
        
        # TensorBoard setup
        self.log_dir = "./outputs/tensorboard_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.episode_count = 0
    
    def decay_epsilon(self):
        """Decay exploration rate over time"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self,info,done):
        #Info: {'distance_to_goal': 2.3431520462036133, 'success': 0.0, 'spl': 0.0, 'soft_spl': 0.05283481905361087, 'num_steps': 15, 'collisions': {'count': 0, 'is_collision': False}, 'distance_to_goal_reward': 0.011035680770874023}
        
        # Improved reward calculation
        # 1. Distance reward: larger penalty for being far from goal
        dist_reward = -info['distance_to_goal'] * 0.1  # Larger penalty (was -info['distance_to_goal'] / 10.0)
        
        # 2. Success reward: much larger reward for success
        success_reward = 50.0 if info['success'] > 0 else 0.0  # Much larger success reward (was 10.0)
        
        # 3. Step penalty: larger penalty for taking too many steps
        step_penalty = -0.05 * info['num_steps']  # Larger step penalty (was -0.01)
        
        # 4. Done penalty: larger penalty for failing
        done_penalty = -5.0 if done and info['success'] == 0 else 0.0  # Larger done penalty (was -2.0)
        
        # 5. Collision penalty: larger penalty for collisions
        collision_penalty = -1.0 * info['collisions']['count'] if 'collisions' in info else 0.0  # Larger collision penalty (was -0.5)
        
        reward = dist_reward + success_reward + step_penalty + done_penalty + collision_penalty
        
        # Reward'u -20 ile 60 arasında clamp et (wider range for larger rewards)
        reward = torch.clamp(torch.tensor(reward), -20.0, 60.0).item()
        
        return reward

    def action_selector(self,obs):
        state = self.process_state(obs)
        policy_action, policy_log_prob = self.actor.act(state)

        if random.random() < self.epsilon:
            # Noise ekle
            linear_noise = torch.randn(1) * self.exploration_noise_std
            angular_noise = torch.randn(1) * self.exploration_noise_std
            noise = torch.tensor([linear_noise, angular_noise], dtype=torch.float32).to(self.device)
            action = policy_action + noise
            action = torch.clamp(action, -1.0, 1.0)

            # Noisy action için yeni log prob hesapla
            mu, std = self.actor.forward(state)
            dist = D.Normal(mu, std)
            log_prob = dist.log_prob(action).sum(-1)
        else:
            action = policy_action
            log_prob = policy_log_prob

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

        # Advantage normalization
        if len(advs) > 1:
            advs_tensor = torch.tensor(advs, dtype=torch.float32)
            advs_mean = advs_tensor.mean()
            advs_std = advs_tensor.std()
            if advs_std > 1e-8:  # Avoid division by zero
                advs = [(adv - advs_mean) / advs_std for adv in advs]

        return returns, advs

    def critic_loss(self,returns, states_tensor):
        values_pred = self.critic(states_tensor).squeeze()
        
        # Huber loss kullan (MSE yerine) - daha robust
        cl = F.smooth_l1_loss(values_pred, returns)
        
        # Value clipping ekle - critic'in çok büyük değerler üretmesini engelle
        values_clipped = torch.clamp(values_pred, -10.0, 20.0)
        cl_clipped = F.smooth_l1_loss(values_clipped, returns)
        
        # İkisini birleştir
        cl = 0.5 * cl + 0.5 * cl_clipped
        
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

        # Actor loss hesapla
        al = self.actor_loss(states_tensor,actions_tensor,old_log_probs_tensor,advs)

        # Critic loss hesapla
        cl = self.critic_loss(returns, states_tensor)

        # Actor'ı ayrı optimize et
        self.actor_optimizer.zero_grad()
        al.backward(retain_graph=True)  # retain_graph=True çünkü aynı tensor'ları kullanıyoruz
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)  # Gradient clipping
        self.actor_optimizer.step()

        # Critic'i ayrı optimize et
        self.critic_optimizer.zero_grad()
        cl.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  # Critic için daha yüksek gradient clipping
        self.critic_optimizer.step()

        return al.item(), cl.item(), (al + cl).item()

    def log_to_tensorboard(self, episode_reward, actor_loss, critic_loss, total_loss, step_count, rewards, metrics):
        """
        Log training data to TensorBoard
        
        Args:
            episode_reward (float): Total reward for the episode
            actor_loss (float): Actor loss value
            critic_loss (float): Critic loss value
            total_loss (float): Total loss value
            step_count (int): Number of steps in episode
            rewards (list): List of rewards for each step
            values (list): List of critic values for each step
            actions (list): List of actions taken
            metrics (dict): Environment metrics
        """
        # Training metrics
        self.writer.add_scalar('Training/Episode_Reward', episode_reward, self.episode_count)
        self.writer.add_scalar('Training/Actor_Loss', actor_loss, self.episode_count)
        self.writer.add_scalar('Training/Critic_Loss', critic_loss, self.episode_count)
        self.writer.add_scalar('Training/Total_Loss', total_loss, self.episode_count)
        self.writer.add_scalar('Training/Episode_Length', step_count, self.episode_count)
        self.writer.add_scalar('Training/Epsilon', self.epsilon, self.episode_count)
        self.writer.add_scalar('Training/Average_Step_Reward', episode_reward / max(step_count, 1), self.episode_count)
        
        # Environment metrics
        self.writer.add_scalar('Metrics/Distance_to_Goal', metrics.get('distance_to_goal', 0), self.episode_count)
        self.writer.add_scalar('Metrics/Success', metrics.get('success', 0), self.episode_count)
        self.writer.add_scalar('Metrics/SPL', metrics.get('spl', 0), self.episode_count)
        self.writer.add_scalar('Metrics/Soft_SPL', metrics.get('soft_spl', 0), self.episode_count)
        self.writer.add_scalar('Metrics/Collisions', metrics.get('collisions', {}).get('count', 0), self.episode_count)
        
        
        # Reward statistics
        if rewards:
            rewards_tensor = torch.tensor(rewards).to(self.device)
            self.writer.add_scalar('Rewards/Mean', rewards_tensor.mean().item(), self.episode_count)
            self.writer.add_scalar('Rewards/Std', rewards_tensor.std().item(), self.episode_count)
            self.writer.add_scalar('Rewards/Min', rewards_tensor.min().item(), self.episode_count)
            self.writer.add_scalar('Rewards/Max', rewards_tensor.max().item(), self.episode_count)
        
        # GPU metrics from pynvml
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = mem_info.used / 1024**2
            mem_total_mb = mem_info.total / 1024**2
            mem_usage_percent = (mem_info.used / mem_info.total) * 100
            
            # GPU utilization
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage_percent = gpu_util.gpu
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            self.writer.add_scalar('GPU/Memory_Used_MB', mem_used_mb, self.episode_count)
            self.writer.add_scalar('GPU/Memory_Total_MB', mem_total_mb, self.episode_count)
            self.writer.add_scalar('GPU/Memory_Usage_Percent', mem_usage_percent, self.episode_count)
            self.writer.add_scalar('GPU/Utilization_Percent', gpu_usage_percent, self.episode_count)
            self.writer.add_scalar('GPU/Temperature_Celsius', temp, self.episode_count)
            
        except:
            pass  # GPU monitoring not available
        
        self.episode_count += 1

    def log_final_summary(self, total_rewards):
        """
        Log final training summary to TensorBoard
        
        Args:
            total_rewards (list): List of all episode rewards
        """
        if total_rewards:
            self.writer.add_scalar('Training/Final_Average_Reward', sum(total_rewards) / len(total_rewards), 0)
            self.writer.add_scalar('Training/Best_Reward', max(total_rewards), 0)
            self.writer.add_scalar('Training/Worst_Reward', min(total_rewards), 0)
            self.writer.add_scalar('Training/Total_Episodes', len(total_rewards), 0)

    def close_tensorboard(self):
        """Close TensorBoard writer"""
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.log_dir}")
        print("To view TensorBoard, run: tensorboard --logdir=outputs/tensorboard_logs")

    def calculate_advantage_returns(self,rewards,values,states,actions,log_probs):
        returns, advs = self.compute_returns_advantages(rewards, values)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advs = torch.tensor(advs, dtype=torch.float32).to(self.device)

        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        old_log_probs_tensor = torch.stack(log_probs)
        return returns, advs, states_tensor, actions_tensor, old_log_probs_tensor

    def process_state(self,obs):
        # RGB frame - ResNet expects 3 channels, so we keep the original RGB
        rgb_frame = torch.from_numpy(obs["rgb"]).float().to(self.device)  # (H,W,3)
        # Depth frame - convert to 3 channels for ResNet compatibility
        #depth_frame = torch.from_numpy(obs["depth"]).float().to(self.device)  # (H,W,1)
        #depth_frame = depth_frame.repeat(1, 1, 3)  # (H,W,3) - repeat depth channel 3 times

        rgb_frame = rgb_frame.unsqueeze(0)   # (1,H,W,3)
        #depth_frame = depth_frame.unsqueeze(0)  # (1,H,W,3)
        rgb_feature = self.rgb_encoder(rgb_frame)
        #depth_feature = self.depth_encoder(depth_frame)
        compass = torch.from_numpy(obs["compass"]).float().unsqueeze(0).to(self.device)     # (1,3)
        gps = torch.from_numpy(obs["gps"]).float().unsqueeze(0).to(self.device)             # (1,2)
        objectgoal = torch.from_numpy(obs["objectgoal"]).float().unsqueeze(0).to(self.device) # (1,num_object_classes)

        state = torch.cat([rgb_feature, compass, gps, objectgoal], dim=1)
        return state

    def save(self, total_reward,filepath="./outputs/checkpoints/"):
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
            #'depth_encoder_state_dict': self.depth_encoder.state_dict(),
            
            # Optimizer states
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            #'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            
            # Training parameters
            'gamma': self.gamma,
            'lam': self.lam,
            'eps_clip': self.eps_clip,
            'device': str(self.device),
            
        }
        os.makedirs(filepath, exist_ok=True)
        torch.save(checkpoint, filepath+"last.pt")
        if self.reward_max is None or total_reward > self.reward_max:
            torch.save(total_reward, filepath+"best.pt")
            self.reward_max = total_reward
        #print(f"Model saved successfully to {filepath}")
    
    def load(self, filepath):
        """
        Load all model weights, optimizers, and training state from a file.
        
        Args:
            filepath (str): Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model states
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.rgb_encoder.load_state_dict(checkpoint['rgb_encoder_state_dict'])
        #self.depth_encoder.load_state_dict(checkpoint['depth_encoder_state_dict'])
        
        # Load optimizer states
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        #self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        
        # Load training parameters
        self.gamma = checkpoint['gamma']
        self.lam = checkpoint['lam']
        self.eps_clip = checkpoint['eps_clip']
        
        print(f"Model loaded successfully from {filepath}")