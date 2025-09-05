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
    def __init__(self,goal_category,state_dim,action_dim,gamma=0.99,lam=0.95,lr_actor=3e-4,lr_critic=5e-4,eps_clip = 0.2,lr_encoder=1e-4,epsilon=1.0,epsilon_min=0.01,epsilon_decay=0.995):
        self.goal_category = goal_category
        
        # Exploration parameters
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate
        self.exploration_noise_std = 0.3  # Standard deviation for exploration noise
        
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
        dist_reward = -info['distance_to_goal']  # distance küçüldükçe reward artar
        # 2. Success reward: hedefe ulaşıldığında büyük ödül
        success_reward = 100.0 if info['success'] > 0 else 0.0
        # 3. Step penalty: kısa yolları teşvik
        step_penalty = -0.001 * info['num_steps']
        # 4. Soft SPL penalty: soft SPL küçüldükçe reward azalır
        #soft_spl_reward = info['soft_spl']
        # 6. Done penalization (opsiyonel)
        # Eğer episode başarısız ve done = True ise ekstra ceza
        done_penalty = -20.0 if done and info['success'] == 0 else 0.0
        reward = dist_reward + success_reward + step_penalty  + done_penalty
        return reward

    def action_selector(self,obs):
        #linear_velocity = random.uniform(-1.0, 1.0)
        #angular_velocity = random.uniform(-1.0, 1.0)
        # Always get state and policy action first
        state = self.process_state(obs)
        policy_action, log_prob = self.actor.act(state)
        
        # Epsilon-greedy exploration: add separate noise for linear and angular velocities
        if random.random() < self.epsilon:
            # Add separate Gaussian noise for each action component
            linear_noise = torch.randn(1) * self.exploration_noise_std
            angular_noise = torch.randn(1) * self.exploration_noise_std
            
            # Add noise to policy action components separately
            action = policy_action + torch.tensor([linear_noise, angular_noise], dtype=torch.float32).to(self.device)
            
            # Clamp to valid action range [-1, 1]
            action = torch.clamp(action, -1.0, 1.0)
        else:
            # Use policy action directly (no noise)
            action = policy_action
        
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
        #self.encoder_optimizer.zero_grad()

        total_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        #self.encoder_optimizer.step()
        
        # Return loss values for TensorBoard logging
        return al.item(), cl.item(), total_loss.item()

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