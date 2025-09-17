import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import torchvision.models as models

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.mu = nn.Linear(64, action_dim)
        
        # Layer normalization instead of batch normalization
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(64)
        
        # Dropout ekle
        self.dropout = nn.Dropout(0.1)
        
        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for module in [self.fc1, self.fc2, self.fc3, self.mu]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.tanh(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.tanh(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.tanh(self.ln3(self.fc3(x)))
        action = torch.tanh(self.mu(x))  # Direkt action çıktısı
        return action

    def act(self, state):
        action = self.forward(state)
        return action.detach()

# === Critic (Value) Network ===
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        
        # Layer normalization instead of batch normalization
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        self.ln4 = nn.LayerNorm(64)
        
        # Dropout
        self.dropout = nn.Dropout(0.15)
        
        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for module in [self.fc1, self.fc2, self.fc3, self.fc4]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        # Output layer için farklı initialization
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)  # Küçük gain
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.ln4(self.fc4(x)))
        # Output layer'da aktivasyon yok - negatif değerlere izin ver
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


class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=512, pretrained=True):
        super().__init__()
        # Pretrained ResNet18 al (ImageNet üzerinde eğitilmiş)
        resnet = models.resnet18(pretrained=pretrained)

        # FC katmanını at → sadece convolutional backbone kalsın
        modules = list(resnet.children())[:-1]  # son FC katmanını çıkar
        self.backbone = nn.Sequential(*modules)

        # Çıkış boyutu resnet18 için 512, bunu istediğin latent boyuta çevirebilirsin
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        # x: (B,H,W,C)
        x = x.permute(0, 3, 1, 2)  # (B,C,H,W)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = self.backbone(x)  # (B,512,1,1)
        x = torch.flatten(x, 1)  # (B,512)
        x = self.fc(x)           # (B,output_dim)
        return x