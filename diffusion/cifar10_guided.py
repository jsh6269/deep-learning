import math
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

# Hyperparameters
img_size = 32
batch_size = 128
num_timesteps = 1000
epochs = 20
lr = 2e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
guidance_scale = 3.0  # classifier-free guidance scale

# Utility to display and save sample grids
def show_images(images, labels=None, rows=2, cols=10, save_path='cifar_guided.png'):
    fig = plt.figure(figsize=(cols, rows))
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = images[i]
        # Convert tensor or PIL to numpy array
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = np.array(img)
        # Rescale if in [0,1]
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        ax.imshow(img_np)
        if labels is not None:
            ax.set_title(str(labels[i]))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# Positional encoding for time steps
def _pos_encoding(time_idx, D, device='cpu'):
    v = torch.zeros(D, device=device)
    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))
    v[0::2] = torch.sin(time_idx / div_term[0::2])
    v[1::2] = torch.cos(time_idx / div_term[1::2])
    return v

def pos_encoding(timesteps, D, device='cpu'):
    v = torch.zeros(len(timesteps), D, device=device)
    for idx, t in enumerate(timesteps):
        v[idx] = _pos_encoding(t, D, device)
    return v

# Convolutional block with time embedding
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, H, W = x.shape
        t_emb = self.mlp(v).view(N, C, 1, 1)
        return self.convs(x + t_emb)

# U-Net conditioning on time + optional label (for classifier-free)
class UNetCond(nn.Module):
    def __init__(self, in_ch=3, time_embed_dim=256, num_labels=10):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.down1 = ConvBlock(in_ch, 128, time_embed_dim)
        self.down2 = ConvBlock(128, 256, time_embed_dim)
        self.bot1 = ConvBlock(256, 512, time_embed_dim)
        self.up2 = ConvBlock(256 + 512, 256, time_embed_dim)
        self.up1 = ConvBlock(256 + 128, 128, time_embed_dim)
        self.out = nn.Conv2d(128, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # label embedding for conditional branch
        self.label_emb = nn.Embedding(num_labels, time_embed_dim)

    def forward(self, x, timesteps, labels=None):
        t = pos_encoding(timesteps, self.time_embed_dim, device=x.device)
        if labels is not None:
            t = t + self.label_emb(labels)

        x1 = self.down1(x, t)
        x2 = self.down2(self.maxpool(x1), t)
        xb = self.bot1(self.maxpool(x2), t)

        xu = self.upsample(xb)
        xu = self.up2(torch.cat([xu, x2], dim=1), t)
        xu = self.upsample(xu)
        xu = self.up1(torch.cat([xu, x1], dim=1), t)
        return self.out(xu)

# Diffusion scheduler and sampling with classifier-free guidance
class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t):
        idx = t - 1
        a_bar = self.alpha_bars[idx].view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
        return xt, noise

    def p_sample(self, x, t, eps):
        idx = t - 1
        a = self.alphas[idx].view(-1,1,1,1)
        a_bar = self.alpha_bars[idx].view(-1,1,1,1)
        a_bar_prev = self.alpha_bars[idx-1].view(-1,1,1,1) if idx[0]>0 else torch.ones_like(a)
        mu = (x - ((1 - a) / torch.sqrt(1 - a_bar)) * eps) / torch.sqrt(a)
        std = torch.sqrt((1 - a) * (1 - a_bar_prev) / (1 - a_bar))
        noise = torch.randn_like(x)
        noise[t == 1] = 0
        return mu + noise * std

    def sample(self, model, x_shape=(20,3,32,32), labels=None, guidance_scale=0.0):
        model.eval()
        x = torch.randn(x_shape, device=self.device)
        B = x_shape[0]
        if labels is None:
            labels = torch.randint(0, model.label_emb.num_embeddings, (B,), device=self.device)

        with torch.no_grad():
            for step in tqdm(range(self.num_timesteps, 0, -1)):
                t = torch.full((B,), step, device=self.device, dtype=torch.long)
                eps_cond = model(x, t, labels)
                eps_uncond = model(x, t, None)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                x = self.p_sample(x, t, eps)
        model.train()
        to_pil = transforms.ToPILImage()
        imgs = [to_pil(x[i].clamp(-1,1).add(1).div(2)) for i in range(x.shape[0])]
        return imgs, labels

# Prepare CIFAR-10 dataset
preprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

dataset = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize model, diffuser, optimizer
diffuser = Diffuser(num_timesteps, device=device)
model = UNetCond(in_ch=3, time_embed_dim=256, num_labels=10).to(device)
optimizer = Adam(model.parameters(), lr=lr)

# Training loop with classifier-free loss
losses = []
for epoch in range(epochs):
    running_loss = 0.0
    for imgs, labels in tqdm(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        t = torch.randint(1, num_timesteps+1, (imgs.size(0),), device=device)

        x_noisy, noise = diffuser.add_noise(imgs, t)
        noise_pred = model(x_noisy, t, labels)
        noise_pred_uncond = model(x_noisy, t, None)
        loss = (F.mse_loss(noise_pred, noise) + F.mse_loss(noise_pred_uncond, noise)) * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# Plot training loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# CIFAR-10 class names
cifar10_labels = dataset.classes

# Generate one image per class
num_classes = len(cifar10_labels)
labels_to_gen = torch.arange(num_classes, device=device)
sampled_imgs, _ = diffuser.sample(model, x_shape=(num_classes,3,32,32), labels=labels_to_gen, guidance_scale=guidance_scale)

# Show one image per class with labels
show_images(sampled_imgs, labels=cifar10_labels, rows=2, cols=5, save_path='cifar10_guided.png')
