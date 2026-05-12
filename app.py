import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image as PILImage
import requests
from io import BytesIO
import os

# ── Config ───────────────────────────────────────────────────────────────
IMAGE_SIZE = 128
CHANNELS   = 3
T          = 400
BASE_CH    = 64
CH_MULT    = (1, 2, 4)
NUM_RES    = 2
DROPOUT    = 0.1
DDIM_STEPS = 50
USE_AMP    = False

# Hugging Face model path
MODEL_URL = "https://huggingface.co/aneelaBashir22f3414/document-to-markdown-generation/resolve/main/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Helpers ───────────────────────────────────────────────────────────────
def denormalize(t):
    return ((t.clamp(-1, 1) + 1) / 2)

def download_model_from_hf(url, local_path="best_model.pth"):
    """Download model from Hugging Face if not exists locally"""
    if not os.path.exists(local_path):
        print(f"Downloading model from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Model downloaded successfully to {local_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    else:
        print(f"Model already exists at {local_path}")
    return local_path

# ── Noise Schedule ────────────────────────────────────────────────────────
class NoiseSchedule:
    def __init__(self, T=400, beta_start=1e-4, beta_end=0.02,
                 schedule_type="cosine", device="cpu"):
        self.T      = T
        self.device = device
        if schedule_type == "cosine":
            steps           = T + 1
            t               = torch.linspace(0, T, steps) / T
            f_t             = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_bar_raw  = f_t / f_t[0]
            betas           = 1 - (alphas_bar_raw[1:] / alphas_bar_raw[:-1])
            betas           = betas.clamp(min=1e-5, max=0.9999)
        else:
            betas = torch.linspace(beta_start, beta_end, T)

        self.betas               = betas.to(device)
        self.alphas              = (1.0 - self.betas)
        self.alphas_bar          = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev     = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_bar     = self.alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1 - self.alphas_bar).sqrt()
        self.posterior_variance  = (
            self.betas * (1 - self.alphas_bar_prev) / (1 - self.alphas_bar)
        ).clamp(min=1e-20)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab     = self.sqrt_alphas_bar[t].view(-1,1,1,1)
        sqrt_1m_ab  = self.sqrt_one_minus_alphas_bar[t].view(-1,1,1,1)
        return sqrt_ab * x0 + sqrt_1m_ab * noise, noise

# ── Building Blocks ───────────────────────────────────────────────────────
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim   = self.dim // 2
        emb        = math.log(10000) / (half_dim - 1)
        emb        = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb        = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2))
        self.block1   = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(),
                                       nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.block2   = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(),
                                       nn.Dropout(dropout),
                                       nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, time_emb):
        h            = self.block1(x)
        t_out        = self.time_mlp(time_emb)
        scale, shift = t_out.unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h            = h * (scale + 1) + shift
        return self.block2(h) + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm   = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)
        self.heads  = num_heads
        self.scale  = (channels // num_heads) ** -0.5
    def forward(self, x):
        B, C, H, W = x.shape
        h   = self.norm(x).reshape(B, C, H*W).permute(0, 2, 1)
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, H*W, self.heads, C//self.heads)
                       .permute(0,2,1,3), qkv)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.permute(0,2,1,3).reshape(B, H*W, C)
        return x + self.to_out(out).permute(0,2,1).reshape(B, C, H, W)

class Downsample(nn.Module):
    def __init__(self, ch): super().__init__(); self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                   nn.Conv2d(ch, ch, 3, padding=1))
    def forward(self, x): return self.conv(x)

# ── U-Net ─────────────────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, ch_mult=(1,2,4),
                 num_res=2, dropout=0.1, T=400):
        super().__init__()
        time_emb_dim = base_ch * 4
        ch_sizes     = [base_ch * m for m in ch_mult]

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_ch),
            nn.Linear(base_ch, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.init_conv   = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self._enc_out_chs = [base_ch]
        in_ch = base_ch

        for lvl, out_ch in enumerate(ch_sizes):
            for blk in range(num_res):
                self.down_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
                self._enc_out_chs.append(in_ch)
                if lvl == len(ch_sizes)-1 and blk == num_res-1:
                    self.down_blocks.append(AttentionBlock(in_ch))
            if lvl < len(ch_sizes)-1:
                self.downsamples.append(Downsample(in_ch))
                self._enc_out_chs.append(in_ch)

        self.mid_res1 = ResidualBlock(in_ch, in_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_res2 = ResidualBlock(in_ch, in_ch, time_emb_dim, dropout)

        self.up_blocks  = nn.ModuleList()
        self.upsamples  = nn.ModuleList()
        skip_chs = list(self._enc_out_chs)

        for lvl, out_ch in enumerate(reversed(ch_sizes)):
            for blk in range(num_res + 1):
                skip_ch = skip_chs.pop()
                self.up_blocks.append(ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
                if lvl == 0 and blk == 0:
                    self.up_blocks.append(AttentionBlock(in_ch))
            if lvl < len(ch_sizes)-1:
                self.upsamples.append(Upsample(in_ch))

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, in_channels, 1)

    def forward(self, x, t):
        temb  = self.time_embed(t)
        h     = self.init_conv(x)
        skips = [h]
        d_iter  = iter(self.down_blocks)
        ds_iter = iter(self.downsamples)

        for lvl in range(len(CH_MULT)):
            for blk in range(NUM_RES):
                h = next(d_iter)(h, temb); skips.append(h)
                if lvl == len(CH_MULT)-1 and blk == NUM_RES-1:
                    h = next(d_iter)(h)
            if lvl < len(CH_MULT)-1:
                h = next(ds_iter)(h); skips.append(h)

        h = self.mid_res1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, temb)

        u_iter  = iter(self.up_blocks)
        us_iter = iter(self.upsamples)

        for lvl in range(len(CH_MULT)):
            for blk in range(NUM_RES + 1):
                h = torch.cat([h, skips.pop()], dim=1)
                h = next(u_iter)(h, temb)
                if lvl == 0 and blk == 0:
                    h = next(u_iter)(h)
            if lvl < len(CH_MULT)-1:
                h = next(us_iter)(h)

        return self.out_conv(self.out_act(self.out_norm(h)))

# ── Load Model from Hugging Face ────────────────────────────────────────────
print("Downloading/Loading model from Hugging Face...")
model_path = download_model_from_hf(MODEL_URL)

noise_schedule = NoiseSchedule(T=T, schedule_type="cosine", device=device)

model = UNet(in_channels=CHANNELS, base_ch=BASE_CH, ch_mult=CH_MULT,
             num_res=NUM_RES, dropout=DROPOUT, T=T).to(device)

# Load model checkpoint
ckpt  = torch.load(model_path, map_location=device, weights_only=False)
state = ckpt.get('ema_state', ckpt.get('model_state', ckpt))

# Remove "module." prefix if present (from DataParallel)
new_state = {}
for k, v in state.items():
    new_key = k.replace("module.", "")
    new_state[new_key] = v

model.load_state_dict(new_state)
model.eval()
print(f"Model ready on {device}!")

# ── DDIM Sampling ─────────────────────────────────────────────────────────
@torch.no_grad()
def ddim_sample(n_samples=1, ddim_steps=50, eta=0.0):
    T_sched   = noise_schedule.T
    step_size = T_sched // ddim_steps
    timesteps = list(range(0, T_sched, step_size))[::-1]
    x         = torch.randn(n_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
    intermediates = []

    for i, t_val in enumerate(timesteps):
        t          = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        alpha_bar  = noise_schedule.alphas_bar[t_val]
        alpha_bar_prev = (noise_schedule.alphas_bar[timesteps[i+1]]
                         if i+1 < len(timesteps) else torch.tensor(1.0, device=device))
        pred_noise = model(x, t)
        pred_x0    = (x - (1-alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        pred_x0    = pred_x0.clamp(-1, 1)
        dir_xt     = (1 - alpha_bar_prev).sqrt() * pred_noise
        noise_add  = eta * torch.randn_like(x) if eta > 0 else 0
        x          = alpha_bar_prev.sqrt() * pred_x0 + dir_xt + noise_add
        if i % max(ddim_steps // 5, 1) == 0 or i == len(timesteps)-1:
            intermediates.append(denormalize(x[0].cpu()).permute(1,2,0).numpy())
    return x, intermediates

# ── Gradio Functions ──────────────────────────────────────────────────────
def generate_from_noise(num_images, ddim_steps_val, eta_val):
    gen_imgs, intermediates = ddim_sample(int(num_images), int(ddim_steps_val), float(eta_val))
    gen_cpu = denormalize(gen_imgs.cpu())
    result  = []
    for i in range(int(num_images)):
        arr = (gen_cpu[i].permute(1,2,0).numpy() * 255).astype(np.uint8)
        result.append(PILImage.fromarray(arr))

    # Steps grid
    n    = len(intermediates)
    imgs = [PILImage.fromarray((intermediates[i]*255).astype(np.uint8)) for i in range(n)]
    w, h = imgs[0].size
    grid = PILImage.new('RGB', (w*n, h))
    for i, im in enumerate(imgs):
        grid.paste(im, (i*w, 0))

    return result[0] if len(result)==1 else result, grid

def reconstruct_uploaded(uploaded_img, start_t_val, ddim_steps_val):
    if uploaded_img is None:
        return None, None, None

    img_resized = uploaded_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np      = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor  = torch.from_numpy(img_np).permute(2,0,1)
    img_tensor  = (img_tensor - 0.5) / 0.5
    x0          = img_tensor.unsqueeze(0).to(device)
    start_t     = int(start_t_val)
    t_tensor    = torch.tensor([start_t], device=device, dtype=torch.long)
    noisy_x, _  = noise_schedule.q_sample(x0, t_tensor)

    step_sz   = max(start_t // int(ddim_steps_val), 1)
    timesteps = list(range(0, start_t, step_sz))[::-1]
    x         = noisy_x.clone()

    with torch.no_grad():
        for i, t_val in enumerate(timesteps):
            t          = torch.full((1,), t_val, device=device, dtype=torch.long)
            alpha_bar  = noise_schedule.alphas_bar[t_val]
            alpha_bar_prev = (noise_schedule.alphas_bar[timesteps[i+1]]
                             if i+1<len(timesteps) else torch.tensor(1.0, device=device))
            pred_noise = model(x, t)
            pred_x0    = (x - (1-alpha_bar).sqrt()*pred_noise)/alpha_bar.sqrt()
            pred_x0    = pred_x0.clamp(-1,1)
            x          = alpha_bar_prev.sqrt()*pred_x0 + (1-alpha_bar_prev).sqrt()*pred_noise

    orig_pil  = PILImage.fromarray(np.array(img_resized))
    noisy_pil = PILImage.fromarray((denormalize(noisy_x.squeeze(0).cpu()).permute(1,2,0).numpy()*255).astype(np.uint8))
    recon_pil = PILImage.fromarray((denormalize(x.squeeze(0).cpu()).permute(1,2,0).numpy()*255).astype(np.uint8))
    return orig_pil, noisy_pil, recon_pil

# ── Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="DDPM Face Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎨 DDPM Face Generator\nGenerate from pure noise **OR** reconstruct your own image!")
    gr.Markdown(f"**Model Source:** [Hugging Face]({MODEL_URL})")

    with gr.Tabs():
        with gr.Tab("🎲 Generate from Noise"):
            with gr.Row():
                with gr.Column():
                    num_img  = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
                    ddim_sl  = gr.Slider(10, 100, value=50, step=10, label="DDIM Steps")
                    eta_sl   = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Eta (0=sharp, 1=diverse)")
                    gen_btn  = gr.Button("Generate!", variant="primary")
                with gr.Column():
                    gen_out  = gr.Image(label="Generated Image", type="pil")
                    step_out = gr.Image(label="Denoising Steps", type="pil")
            gen_btn.click(fn=generate_from_noise,
                          inputs=[num_img, ddim_sl, eta_sl],
                          outputs=[gen_out, step_out])

        with gr.Tab("🖼️ Reconstruct Your Image"):
            with gr.Row():
                with gr.Column():
                    up_img   = gr.Image(label="Upload Your Image", type="pil")
                    t_sl     = gr.Slider(50, 350, value=250, step=50, label="Noise Level")
                    ddim_sl2 = gr.Slider(10, 100, value=50, step=10, label="DDIM Steps")
                    rec_btn  = gr.Button("Reconstruct!", variant="primary")
                with gr.Column():
                    orig_out  = gr.Image(label="Original", type="pil")
                    noisy_out = gr.Image(label="Noisy", type="pil")
                    recon_out = gr.Image(label="Reconstructed", type="pil")
            rec_btn.click(fn=reconstruct_uploaded,
                          inputs=[up_img, t_sl, ddim_sl2],
                          outputs=[orig_out, noisy_out, recon_out])

    gr.Markdown("**Model:** DDPM + U-Net | **Dataset:** CelebA-HQ 128×128 | **Sampling:** DDIM")

# For Hugging Face Spaces compatibility
if __name__ == "__main__":
    demo.launch()
