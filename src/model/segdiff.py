import math
from abc import abstractmethod
from collections import namedtuple
from functools import partial

import monai
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from src.model.archs.RRDB import RRDB, make_layer

ModelPrediction = namedtuple(
    "ModelPrediction",
    [
        "pred_noise",
        "pred_x_0",
    ],
)
# ModelPrediction: 모델 예측 결과를 저장하는 named tuple (pred_noise: 예측된 noise, pred_x_0: 예측된 x_0)
# prediction = ModelPrediction(pred_noise, pred_x_0)
# prediction.pred_noise, prediction.pred_x_0


def normalize_to_zero_to_one(img):
    return img.clamp(0, 1)


def metatensor_to_tensor(input):
    if isinstance(input, monai.data.MetaTensor):
        return input.as_tensor()
    elif isinstance(input, list):
        return metatensor_to_tensor(input[0])
    else:
        return input


def tensor_to_numpy(x):
    return x.to("cpu").detach().numpy()


def normalize_to_neg_one_to_one(t):
    # 0->-1, 1->1
    return t * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None


def default_load(val, d):
    """
    Returns the value if it exists, otherwise returns the result of the callable or the default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d


class F_model(nn.Module):
    """
    F: creating image embeddings from previous output x_t
    a single-channel input and an output of C channels
    """

    def __init__(self, in_channels=1, out_channels=128):
        super(F_model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class G_model(nn.Module):
    """
    G: encoding the input image I, RRDBNet architecture
    """

    def __init__(self, in_channels=3, out_channels=128, nf=64, nb=1, gc=32):
        super(G_model, self).__init__()
        RRDB_block_f = partial(RRDB, nf, gc)

        self.conv_first = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, I):
        fea = self.conv_first(I)
        fea = self.RRDB_trunk(fea)
        trunk = self.trunk_conv(fea)
        fea = fea + trunk
        fea = self.HRconv(fea)
        fea = self.lrelu(fea)
        out = self.conv_last(fea)
        return out


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    : param timesteps: a 1-D Tensor of N indices, one per batch element.
    These may be fractional.
    : param dim: the dimension of the output.
    : param max_period: controls the minimum frequency of the embeddings.
    : return: an[N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0,
                      end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LUT(nn.Module):
    """
    Lookup table for time embeddings.
    """

    def __init__(self, embedding_dim: int = 64, out_channels: int = 64):
        super(LUT, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(out_channels, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, t):
        t = timestep_embedding(t, dim=self.embedding_dim)
        t = self.mlp(t)
        return t


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Block(nn.Module):
    """
    Basic block containing a convolutional layer, group normalization, and SiLU activation.
    """

    def __init__(self, in_channels, out_channels, groups=8):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """
    Containing two convolutional blocks, with optional time embedding via an MLP if provided, and a residual connection to adjust input channels.
    The output is the sum of the second block's output and the input transformed by a 1x1 convolution or identity mapping.
    """

    def __init__(self, in_channels, out_channels, *, time_emb_dim=None, groups=8):
        super(ResidualBlock, self).__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2),
            )
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(in_channels, out_channels, groups=groups)
        self.block2 = Block(out_channels, out_channels, groups=groups)
        self.res_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttentionBlock(nn.Module):
    """
    Reshape the input tensor to (batch_size, channels, spatial_dim1 * spatial_dim2) and apply group normalization.
    Apply QKV 1x1 convolution to generate queries, keys, and values, perform self-attention, then reshape the output back and add it to the input.
    """

    def __init__(self, channels, num_heads=1):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.attention = QKVAttention()
        self.out = zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])

        h = self.attention(qkv)
        h = h.reshape(b, c, -1)
        h = self.out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    Split the input qkv into queries, keys, and values, and scale the queries and keys by 1/sqrt(d_k).
    Compute attention weights by taking the dot product between queries and keys, apply softmax, and compute the output as a weighted sum of the values.
    """

    def forward(self, qkv):
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(q.shape[-1])
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        return torch.einsum("bts,bcs->bct", weight, v)


class Downsample(nn.Module):
    """
    Downsample the input tensor by a factor of 2 using either a convolutional layer or average pooling.
    """

    def __init__(self, channels, stride=2, use_conv=True):
        super(Downsample, self).__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.layer = nn.Conv2d(
                channels, channels, kernel_size=3, stride=stride, padding=1)
        else:
            self.layer = nn.AvgPool2d(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.layer(x)


class Bottleneck(nn.Module):
    """
    A bottleneck block containing two residual blocks.
    """

    def __init__(self):
        super(Bottleneck, self).__init__()
        self.res_block = ResidualBlock()

    def forward(self, x):
        x = self.res_block(x)
        x = self.res_block(x)
        return x


class Upsample(nn.Module):
    """
    Upsample the input tensor by a factor of 2 using either a convolutional layer or bilinear interpolation.
    """

    def __init__(self, channels, use_conv=True):
        super(Upsample, self).__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.layer = nn.Conv2d(
                channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.layer(x)
        return x


class E_dot_D(nn.Module):
    """
    E_dot_D: encoder-decoder part, contains a series of downsample, bottleneck, and upsample blocks.
    get time embeddings from LUT and apply them to the residual blocks with F+G.
    """

    def __init__(
        self,
        in_channels=64,
        out_channels=1,
        num_res_blocks=4,
        channel_mul=(1, 2, 4, 8),
        num_heads=1,
        conv_resample=True,
    ):
        super(E_dot_D, self).__init__()

        self.num_resolutions = len(channel_mul)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = [12, 6]

        # Downsample
        self.input_block = TimestepEmbedSequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList()
        ch = in_channels
        emb_dim = in_channels

        for i in range(self.num_resolutions):
            out_ch = channel_mul[i] * emb_dim
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(
                    ch, out_ch, time_emb_dim=emb_dim))
                ch = out_ch
                if ch in self.attention_resolutions:
                    self.down_blocks.append(AttentionBlock(ch))

            if i != self.num_resolutions - 1:
                self.down_blocks.append(Downsample(ch, use_conv=conv_resample))

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim=emb_dim),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_emb_dim=emb_dim),
        ])

        # Upsample
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            out_ch = channel_mul[max(i - 1, 0)] * emb_dim
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(
                    ch, out_ch, time_emb_dim=emb_dim))
                ch = out_ch
                if ch in self.attention_resolutions:
                    self.up_blocks.append(AttentionBlock(ch))
            if i != 0:
                self.up_blocks.append(Upsample(ch, use_conv=conv_resample))

        # Final conv
        self.conv_out = nn.Sequential(nn.SiLU(), nn.Conv2d(
            ch, out_channels, kernel_size=3, padding=1))

    def forward(self, F_x, G_I, t):
        # Downsampling
        x = self.input_block(F_x, t)
        x += G_I
        skips = []
        for _, block in enumerate(self.down_blocks):
            if isinstance(block, Downsample):
                x = block(x)
                skips.append(x)
            else:
                x = block(x, t)

        # Bottleneck
        for _, block in enumerate(self.bottleneck):
            x = block(x) if isinstance(block, AttentionBlock) else block(x, t)

        # Upsampling
        for _, block in enumerate(self.up_blocks):
            if isinstance(block, Upsample):
                skip = skips.pop()
                x = x + skip
                x = block(x)
            else:
                x = block(x, t)

        # Final conv
        x = self.conv_out(x)
        return x


class SegDiffUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, nf=64):
        super(SegDiffUnet, self).__init__()
        self.F_model = F_model(in_channels=1, out_channels=nf)
        self.G_model = G_model(in_channels=in_channels, out_channels=nf)
        self.LUT = LUT(embedding_dim=nf, out_channels=nf)
        self.E_dot_D = E_dot_D(in_channels=nf, out_channels=out_channels)

    def forward(self, x_t, I, t):
        F_output = self.F_model(x_t)
        G_output = self.G_model(I)
        LUT_output = self.LUT(t)
        E_dot_D_output = self.E_dot_D(F_output, G_output, LUT_output)
        return E_dot_D_output


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps  # 1000/1000 = 1
    beta_start = scale / 10000  # 1/10000 = 10^(-4)
    beta_end = 2 * scale / 50  # 2*1/500 = 2*10^(-2)
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule
    as proposed in
    Improved Denoising Diffusion Probabilistic Models, A Nichol et al.
    https://doi.org/10.48550/arXiv.2102.09672
    Eqauation 17
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    f = torch.cos(((x / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1.0, clip_min=1e-9):
    """
    Sigmoid schedule
    as proposed in
    Scalable Adaptive Computation for Iterative Generation, A Jabri et al.
    https://doi.org/10.48550/arXiv.2212.11972
    Algorithm 4
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    v_start = torch.Tensor(start / tau).sigmoid()
    v_end = torch.Tensor(end / tau).sigmoid()
    f = (-((t * (end - start) + start) / tau) + v_end) / (v_end - v_start)
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0, 0.999)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters()}
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.device)
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        self.backup = {name: param.data.clone() for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class SegDiff(nn.Module):
    def __init__(
        self,
        device,
        arch,
        criterion='mse',
        beta_schedule="linear",
        min_snr_loss_weight="sigmoid",
        min_snr_gamma=5,
        objective="pred_noise",
        timesteps=100,
        cfg=None,
    ):
        super().__init__()

        if cfg is None:
            cfg = {}
        self.arch = arch

        # Criterion mapping (supports string or nn.Module / callable)
        if isinstance(criterion, str):
            crit_map = {
                'mse': nn.MSELoss(),
                'l1': nn.L1Loss(),
                'bce': nn.BCEWithLogitsLoss(),
            }
            criterion_obj = crit_map.get(criterion.lower(), nn.MSELoss())
        elif isinstance(criterion, nn.Module) or callable(criterion):
            criterion_obj = criterion
        else:
            criterion_obj = nn.MSELoss()

        lr = cfg.get("lr", 1e-4) if isinstance(cfg, dict) else getattr(cfg, 'lr', 1e-4)
        self.optimizer = torch.optim.Adam(self.arch.parameters(), lr=lr)

        self.timesteps = timesteps
        self.objective = objective
        self.criterion = criterion_obj
        self.device = device

        self.cfg = cfg
        # Robust access for config attribute or dict
        self.use_ddim = getattr(cfg, 'ddim_sample', False)
        self.use_ema = getattr(cfg, 'use_ema', False)
        if self.use_ema:
            decay = getattr(cfg, 'ema_decay', 0.999)
            self.ema = EMA(self.arch, decay=decay)

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")

        def rb(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # q_sample buffers
        rb("betas", betas)
        rb("alphas_cumprod", alphas_cumprod)
        rb("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        rb("sqrt_one_minus_alphas_cumprod", 1 - torch.sqrt(alphas_cumprod))

        # posterior mean / variance buffers
        rb("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        rb("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / (1 - alphas_cumprod)))
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        rb("posterior_variance", posterior_variance)
        rb("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-9)))
        rb("posterior_mean_coef1", torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod))
        rb("posterior_mean_coef2", torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        # loss weight based on SNR
        snr = alphas_cumprod / (1.0 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(min=min_snr_gamma)
        if objective == "pred_noise":
            rb("loss_weight", maybe_clipped_snr / snr)
        elif objective == "pred_x0":
            rb("loss_weight", maybe_clipped_snr)
        elif objective == "pred_v":
            rb("loss_weight", maybe_clipped_snr / (snr + 1))
            
    def register_opt(self, optimizer):
        self.optimizer = optimizer

    """
    Equation 5
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
    """

    def predict_x0_from_noise(self, x_t, t, noise):
        """
        (x_t - sqrt(1-alpha_bar_t) * epsilon) / sqrt(alpha_bar_t) = x_0
        """
        sqrt_recip_alphas_cumprod = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        x_0 = (x_t - sqrt_recipm1_alphas_cumprod * noise) * sqrt_recip_alphas_cumprod
        return x_0

    def predict_noise_from_x0(self, x_t, t, x_0):
        """
        epsilon = (x - sqrt(alpha_bar_t) * x_0) / sqrt(1-alpha_bar_t)
        """
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        pred_noise = (x_t - sqrt_alphas_cumprod * x_0) * sqrt_recipm1_alphas_cumprod
        return pred_noise

    # def predict_v(self, x_t, t, noise):
    #     """Predict v parameterization (used in latent diffusion):
    #     v = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * x_0
    #     Rearranged via x_0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)
    #     Input: x_t, timestep t, noise
    #     Returns v
    #     """
    #     # need x_0 first
    #     x_0 = self.predict_x0_from_noise(x_t, t, noise)
    #     sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
    #     sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    #     v = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * x_0
    #     return v

    def q_posterior(self, x_0, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, cond, clamp_x_start=False):
        model_output = self.arch(x, cond, t)
        maybe_clamp = partial(torch.clamp, min=-1.0, max=1.0) if clamp_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_0 = self.predict_x0_from_noise(x, t, model_output)
        elif self.objective == "pred_x0":
            x_0 = model_output
            x_0 = maybe_clamp(x_0)
            pred_noise = self.predict_noise_from_x0(x, t, x_0)

        return ModelPrediction(pred_noise, x_0)

    def p_mean_variance(self, x, t, cond, clip_denoised=True):
        """
        mu_theta & sigma_theta
        """
        preds = self.model_predictions(x, t, cond)
        x_start = preds.pred_x_0

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance_clipped = self.q_posterior(x_0=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance_clipped, x_start

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True):
        """
        Equation 12
        p(x_{t-1} | x_t, I,t)
        """
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

        # Calculate mean and variance
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, cond=cond, clip_denoised=clip_denoised)

        # Add noise to the image
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, cond):
        """
        Reverse Process to obtain image from noise
        """
        shape = cond.shape
        batch = cond.shape[0]

        x_t = torch.randn((batch, 1, shape[2], shape[3]), device=self.device)
        x_hat_0 = None

        for t in tqdm(
            reversed(range(0, self.timesteps)),
            desc="sampling loop time step",
            total=self.timesteps,
            leave=False,
        ):
            # noise / timestep / condition = image
            x_t, x_hat_0 = self.p_sample(x=x_t, t=t, cond=cond)
            if t == self.timesteps - 1 or t % (self.timesteps // 20) == 0:
                """
                image save
                """
        # Keep diffusion space range (optionally map to [0,1] outside)
        return x_t

    @torch.no_grad()
    def ddim_sample(self, x, t, t_prev, cond, eta=0.0):
        """
        DDIM Sampling
        x_{t-1} = \sqrt{\alpha_{t-1}} \hat{x}_0 + \sqrt{1 - \alpha_{t-1} - \sigma^2} \epsilon_theta + \sigma \epsilon
        \hat{x}_0 = (x_t-\sqrt{1-\alpha_t}\epsilon_theta)/(\sqrt{\alpha_t})
        """
        preds = self.model_predictions(x, t, cond, clamp_x_start=True)
        pred_x0 = preds.pred_x_0
        pred_noise = preds.pred_noise

        alpha_cumprod = extract(self.alphas_cumprod, t, x.shape)
        alpha_cumprod_prev = extract(self.alphas_cumprod, t_prev, x.shape)

        sigma = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * (1 - alpha_cumprod / alpha_cumprod_prev))

        dir = torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * pred_noise
        x_prev = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir

        if sigma > 0:
            noise = torch.randn_like(x)
            x_prev += sigma * noise

        return x_prev, pred_x0

    @torch.no_grad()
    def ddim_sample_loop(self, cond, num_steps=50, eta=0.0):
        """
        DDIM을 이용한 빠른 샘플링 루프
        """
        shape = cond.shape
        batch = cond.shape[0]

        # 샘플링 스텝 시퀀스 생성 (예: 1000 스텝 -> 50 스텝)
        times = torch.linspace(-1, self.timesteps - 1, steps=num_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_t = torch.randn((batch, 1, shape[2], shape[3]), device=self.device)

        for t, t_prev in tqdm(time_pairs, desc="DDIM sampling loop", leave=False):
            # 현재 시점 t의 tensor 생성
            time_cond = torch.full((batch,), t, device=self.device, dtype=torch.long)

            # 이전 시점 t_prev의 tensor 생성
            time_prev_cond = torch.full((batch,), t_prev, device=self.device, dtype=torch.long)

            x_t, _ = self.ddim_sample(x=x_t, t=time_cond, t_prev=time_prev_cond, cond=cond, eta=eta)

        return x_t

    @torch.no_grad()
    def sample(self, cond_img):
        cond_img = cond_img.to(self.device)
        if self.use_ddim:
            sample_fn = self.ddim_sample_loop
        else:
            sample_fn = self.p_sample_loop
        samples = sample_fn(cond_img)
        return unnormalize_to_zero_to_one(samples)

    def q_sample(self, x_0, t, noise=None):
        """
        Equation 4
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t)x_0, (1-alpha_bar_t)I)
        """

        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "Noise shape != x_0 shape"

        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_0.shape)

        _mean = sqrt_alpha_bar_t * x_0
        _var = 1.0 - alpha_bar_t

        q_x_t = _mean + torch.sqrt(_var) * noise

        return q_x_t

    def p_losses(self, x_0, t, cond, noise=None):
        """
        Equation 15
        E_{x_0, epsilon, t}[||epsilon - epsilon_theta(x_t, t)||^2]
        """
        noise = default_load(noise, lambda: torch.randn_like(x_0))

        x = self.q_sample(x_0=x_0, t=t, noise=noise)
        assert x.shape[1] == 1, f"channel of x is not 1, but {x.shape[1]}"
        model_output = self.arch(x, cond, t)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_0
        elif self.objective == "pred_v":
            # compute v directly
            sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
            sqrt_one_minus_alpha_bar_t = 1-sqrt_alpha_bar_t
            target = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * x_0
        else:
            raise ValueError(f"Invalid objective: {self.objective}")

            # loss = F.mse_loss(model_output, target)
        loss = self.criterion(model_output, target)
        return loss, model_output

    def feed_data(self, batch):
        """Process input batch data for the model"""
        images = batch["image"].to(self.device)
        masks = batch["label"].to(self.device)
        # index = batch["index"]

        # Ensure correct shape (add channel dimension if needed)
        if images.ndim == 3:
            images = images.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        # Normalize to [-1,1]
        images = normalize_to_neg_one_to_one(images)
        masks = normalize_to_neg_one_to_one(masks)

        return images, masks

    def train_step(self, batch):
        """
        Algorithm 2: Training Algorithm
        """
        """Forward pass with normalization to [0, 1]"""
        # feed_data returns (images, masks); rename for clarity
        cond_img, target_mask = self.feed_data(batch)
        if target_mask.ndim == 3:
            target_mask = rearrange(target_mask, "b h w -> b 1 h w")
        if cond_img.ndim == 3:
            cond_img = rearrange(cond_img, "b h w -> b 1 h w")

        timesteps = torch.randint(0, self.timesteps, (target_mask.shape[0],), device=target_mask.device)

        # operate in [-1,1] space directly
        self.optimizer.zero_grad(set_to_none=True)
        loss, model_output = self.p_losses(target_mask, timesteps, cond_img)
        loss.backward()
        self.optimizer.step()
        if self.use_ema:
            self.ema.update()
        return loss, model_output

    @torch.no_grad()
    def validate_step(self, batch):
        if self.use_ema:
            self.ema.apply_shadow()
        cond_img, target_mask = self.feed_data(batch)

        # Generate sample (returns [0,1])
        output = self.sample(cond_img)
        # Align target to [0,1] for loss
        target_mask_01 = unnormalize_to_zero_to_one(target_mask)
        loss = self.criterion(output, target_mask_01)
        if self.use_ema:
            self.ema.restore()
        return loss, output

    @torch.no_grad()
    def test_step(self, data):
        if isinstance(data, torch.Tensor):
            img = data
        elif isinstance(data, dict):
            img, _ = self.feed_data(data)
        if img.ndim == 3:
            img = rearrange(img, "h w c -> 1 h w c")
        output = self.p_sample_loop(cond=img)
        return output


from model.registry import register  # noqa: E402


@register('segdiff')
def _build_segdiff(args):
    in_ch = int(getattr(args, 'num_channels', 3) or 3)
    arch = SegDiffUnet(in_channels=in_ch, out_channels=1, nf=64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SegDiff(
        device=device,
        arch=arch,
        criterion='mse',
        beta_schedule=getattr(args, 'beta_schedule', 'linear'),
        timesteps=int(getattr(args, 'diff_timesteps', 100)),
    )
    return model
