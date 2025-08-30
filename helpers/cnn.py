import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

class Encoder(nn.Module):
    def __init__(self,rgb_input_size=3,depth_input_size=1,rgb_output_size=32,depth_output_size=32,rgb_hidden_size=16,depth_hidden_size=16):
        super().__init__()
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(rgb_input_size, rgb_hidden_size, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(rgb_hidden_size, rgb_output_size, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_input_size, depth_hidden_size, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(depth_hidden_size, depth_output_size, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # RGB için kullanacağımız transform
        self.rgb_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1,1]
        ])
        self.total_output_size = rgb_output_size + depth_output_size

        # Depth için boyutlandırmayı torchvision ile yapamayacağımız için (tek kanal tensorde),
        # aşağıda manuel resize kullanmak istersen torchvision.transforms.functional'ı kullanabilirsin.
        # Burada basitçe torch.nn.functional.interpolate kullanacağız.

    def forward(self, rgb, depth):
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)
        return rgb_features, depth_features

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

    @torch.no_grad()
    def process(self, rgb_np: np.ndarray, depth_np: np.ndarray, device: str = "cpu"):
        """
        rgb_np   : (H, W, 3), uint8 veya float
        depth_np : (H, W) veya (H, W, 1), float (örn. 0-1 veya metre cinsinden)
        return   : fused embedding  -> shape: [1, 64]  (rgb:32 + depth:32)
        """
        self.eval()
        self.to(device).float()

        # ---- RGB ----
        if rgb_np.dtype != np.uint8:
            # 0-1 aralığında float geliyorsa 0-255'e çevirmeye gerek yok, ToTensor zaten handle eder.
            # Ama uint8 değilse ve 0-255 değilse normalize edeceğiz:
            rgb_np = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
        rgb_t = self.rgb_transform(rgb_np).unsqueeze(0).to(device)  # [1,3,224,224]

        # ---- Depth ----
        # depth_np shape fix
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np.squeeze(-1)
        # torch tensöre çevir
        depth_t = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

        # normalize (örnek: 0-10m arası varsayımı; kendi aralığına göre değiştir)
        depth_t = depth_t.clamp(min=0.0, max=10.0) / 10.0

        # 224x224'e resize
        depth_t = torch.nn.functional.interpolate(depth_t, size=(224, 224), mode="bilinear", align_corners=False)

        # ---- Forward ----
        rgb_feat, depth_feat = self.forward(rgb_t, depth_t)  # [1,32,1,1] ve [1,32,1,1]
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)       # [1,32]
        depth_feat = depth_feat.view(depth_feat.size(0), -1) # [1,32]

        fused = torch.cat([rgb_feat, depth_feat], dim=1)     # [1,64]
        return fused