import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, 64)
        self.mu = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # log std learnable

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        mu = torch.tanh(self.mu(x))  # tanh for [-1,1] range
        std = torch.exp(self.log_std)
        return mu, std

    def act(self, state):
        mu, std = self.forward(state)
        dist = distributions.Normal(mu, std)
        action = dist.sample()
        action_clipped = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action_clipped.detach(), log_prob.detach()

# === Critic (Value) Network ===
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        return self.out(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        # 640x480 resim girince conv’lar sonunda kaç feature kalacağını elle hesaplamak zor.
        #dummy input gönderip çıktının shape[1] (feature sayısı) alınarak fc layer buna göre ayarlanıyor.
        # Böylece flatten işleminden sonra toplamda kaç çıktı olacağı hesaplanıyor.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 640, 480)
            n_flatten = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(n_flatten, 512)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B,H,W,C) → (B,C,H,W) # Ortamdan gelen frame (batch, height, width, channels) yani (B, H, W, C) formatında. PyTorch Conv2d (B, C, H, W) ister.O yüzden channel’ı 3. indexten 1. indexe taşıyoruz.
        x = self.cnn(x)
        x = F.relu(self.fc(x))
        return x  # feature vektörü
        