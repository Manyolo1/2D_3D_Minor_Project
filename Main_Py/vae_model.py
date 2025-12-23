import torch
import torch.nn as nn
import torch.nn.functional as F


class Sampler(nn.Module):
    """Reparameterization sampler: z = mu + exp(0.5*logvar) * eps"""

    def forward(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps


class Encoder(nn.Module):
    def __init__(self, in_ch, base_ch, n_latents, conditional=False, num_cats=0):
        super().__init__()
        self.conditional = conditional
        self.num_cats = num_cats

        # Lightweight CNN encoder treating viewpoints as channels
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=5, stride=2, padding=2),  # 224->112
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=5, stride=2, padding=2),  # 112->56
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1),  # 56->28
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=3, stride=2, padding=1),  # 28->14
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 8, base_ch * 8, kernel_size=3, stride=2, padding=1),  # 14->7
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        feat_dim = base_ch * 8 * 7 * 7

        self.fc_mu = nn.Linear(feat_dim, n_latents)
        self.fc_logvar = nn.Linear(feat_dim, n_latents)

        if self.conditional and self.num_cats > 0:
            self.fc_cls = nn.Linear(feat_dim, self.num_cats)

    def forward(self, x):
        h = self.features(x)
        h = self.avgpool(h)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if self.conditional and self.num_cats > 0:
            class_scores = self.fc_cls(h)
            return mu, logvar, class_scores
        else:
            return mu, logvar


class Decoder(nn.Module):
    def __init__(self, num_vps, base_ch, n_latents, tanh=False, conditional=False, num_cats=0):
        super().__init__()
        self.num_vps = num_vps
        self.tanh = tanh
        self.conditional = conditional
        self.num_cats = num_cats

        in_latent = n_latents + (num_cats if conditional and num_cats > 0 else 0)

        # Project latent to a small spatial feature map and upsample to 224x224
        self.fc = nn.Linear(in_latent, base_ch * 8 * 7 * 7)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 8, base_ch * 8, kernel_size=4, stride=2, padding=1),  # 7->14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=4, stride=2, padding=1),  # 14->28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1),  # 28->56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1),      # 56->112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1),          # 112->224
            nn.ReLU(inplace=True),
        )

        # Output both depth and silhouette as separate channel groups
        self.out_conv = nn.Conv2d(base_ch, 2 * self.num_vps, kernel_size=3, padding=1)

    def forward(self, z_or_pair):
        if isinstance(z_or_pair, (list, tuple)):
            z, cond = z_or_pair
            z = torch.cat([z, cond], dim=1)
        else:
            z = z_or_pair

        h = self.fc(z)
        h = h.view(h.size(0), -1, 7, 7)
        h = self.deconv(h)
        out = self.out_conv(h)

        # Split into depth and silhouette groups
        depth = out[:, : self.num_vps, :, :]
        sil = out[:, self.num_vps :, :, :]

        # Optional output activation
        if self.tanh:
            depth = torch.tanh(depth)
            sil = torch.tanh(sil)
        else:
            # Keep raw logits; training uses L1. Optionally clamp for stability.
            depth = depth
            sil = sil

        return depth, sil


class VAE:
    @staticmethod
    def get_encoder(model_params):
        (
            n_input_ch,
            n_ch,
            n_latents,
            tanh,
            single_vp_net,
            conditional,
            num_cats,
            benchmark,
            dropout_net,
        ) = model_params

        in_ch = 1 if single_vp_net else n_input_ch
        return Encoder(in_ch, n_ch, n_latents, conditional=conditional, num_cats=num_cats)

    @staticmethod
    def get_decoder(model_params):
        (
            n_input_ch,
            n_ch,
            n_latents,
            tanh,
            single_vp_net,
            conditional,
            num_cats,
            benchmark,
            dropout_net,
        ) = model_params

        return Decoder(
            num_vps=n_input_ch,
            base_ch=n_ch,
            n_latents=n_latents,
            tanh=bool(tanh),
            conditional=conditional,
            num_cats=num_cats,
        )
