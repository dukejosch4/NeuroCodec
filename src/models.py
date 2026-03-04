"""NeuroCodec model definitions.

All architectures required for the hybrid video latent prediction system:
- SlotAttentionV2: iterative slot attention encoder
- SlotLatentAutoencoderV2: slot-based latent autoencoder (534K params)
- ResidualDecoderV2: 3-layer cross-attention residual decoder (1.43M params)
- DynamicsTransformer: lightweight slot dynamics predictor (438K params)
- BoundaryDetector: MLP for EASY/HARD frame classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttentionV2(nn.Module):
    """Iterative slot attention module.

    Compresses N input tokens into K slots via iterative attention.
    Uses GRU updates + MLP refinement per iteration.
    """

    def __init__(self, n_slots, slot_dim, input_dim, n_iter=5, hidden_dim=128):
        super().__init__()
        self.n_slots = n_slots
        self.n_iter = n_iter
        self.slot_dim = slot_dim
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(input_dim, slot_dim)
        self.to_v = nn.Linear(input_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_out = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, slot_dim)
        )
        self.scale = slot_dim ** -0.5

    def forward(self, x):
        B, N_tok, _ = x.shape
        slots = (
            self.slot_mu.expand(B, self.n_slots, -1)
            + self.slot_log_sigma.exp().expand(B, self.n_slots, -1)
            * torch.randn(B, self.n_slots, self.slot_dim, device=x.device)
        )
        x = self.norm_input(x)
        k = self.to_k(x)
        v = self.to_v(x)
        for _ in range(self.n_iter):
            sp = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            attn = (torch.einsum("bsd,bnd->bsn", q, k) * self.scale).softmax(dim=1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)
            slots = self.gru(
                torch.einsum("bsn,bnd->bsd", attn, v).reshape(-1, self.slot_dim),
                sp.reshape(-1, self.slot_dim),
            ).reshape(B, self.n_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_out(slots))
        return slots


class SlotLatentAutoencoderV2(nn.Module):
    """Slot-based latent autoencoder (534K params).

    Encodes 1024 latent tokens (16d) into 64 slots (128d) via slot attention,
    then decodes back via 2-layer cross-attention.

    Architecture:
        Encoder: LayerNorm -> 2-layer MLP (16->128) -> SlotAttentionV2
        Decoder: 2x cross-attention (slots->tokens) + MLP + output proj
    """

    def __init__(self, n_slots=64, slot_dim=128, input_dim=16, n_tokens=1024, n_iter=5):
        super().__init__()
        self.n_tokens = n_tokens
        self.slot_dim = slot_dim
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, input_dim) * 0.02)
        self.encoder_norm = nn.LayerNorm(input_dim)
        self.encoder_proj = nn.Sequential(
            nn.Linear(input_dim, slot_dim), nn.GELU(), nn.Linear(slot_dim, slot_dim)
        )
        self.slot_attention = SlotAttentionV2(
            n_slots=n_slots, slot_dim=slot_dim, input_dim=slot_dim, n_iter=n_iter
        )
        # Decoder: 2-layer cross-attention from learned queries attending to slots
        self.decoder_pos = nn.Parameter(torch.randn(1, n_tokens, slot_dim) * 0.02)
        self.cross_norm_q1 = nn.LayerNorm(slot_dim)
        self.cross_norm_kv1 = nn.LayerNorm(slot_dim)
        self.cross_q1 = nn.Linear(slot_dim, slot_dim)
        self.cross_k1 = nn.Linear(slot_dim, slot_dim)
        self.cross_v1 = nn.Linear(slot_dim, slot_dim)
        self.decoder_mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.cross_norm_q2 = nn.LayerNorm(slot_dim)
        self.cross_norm_kv2 = nn.LayerNorm(slot_dim)
        self.cross_q2 = nn.Linear(slot_dim, slot_dim)
        self.cross_k2 = nn.Linear(slot_dim, slot_dim)
        self.cross_v2 = nn.Linear(slot_dim, slot_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, input_dim),
        )
        self.scale = slot_dim ** -0.5

    def encode(self, x):
        """Encode tokens to slots. x: [B, 1024, 16] -> [B, 64, 128]"""
        return self.slot_attention(self.encoder_proj(self.encoder_norm(x + self.pos_embed)))

    def decode(self, slots):
        """Decode slots back to tokens. slots: [B, 64, 128] -> [B, 1024, 16]"""
        # Layer 1: cross-attention
        queries = self.decoder_pos.expand(slots.shape[0], -1, -1)
        q = self.cross_q1(self.cross_norm_q1(queries))
        k = self.cross_k1(self.cross_norm_kv1(slots))
        v = self.cross_v1(self.cross_norm_kv1(slots))
        attn = torch.einsum("bnd,bsd->bns", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        x = queries + torch.einsum("bns,bsd->bnd", attn, v)
        x = x + self.decoder_mlp(x)
        # Layer 2: cross-attention
        q2 = self.cross_q2(self.cross_norm_q2(x))
        k2 = self.cross_k2(self.cross_norm_kv2(slots))
        v2 = self.cross_v2(self.cross_norm_kv2(slots))
        attn2 = torch.einsum("bnd,bsd->bns", q2, k2) * self.scale
        attn2 = attn2.softmax(dim=-1)
        x = x + torch.einsum("bns,bsd->bnd", attn2, v2)
        return self.output_proj(x)

    def forward(self, x):
        """Full autoencoder pass. x: [B, 1024, 16] -> (recon, slots)"""
        slots = self.encode(x)
        recon = self.decode(slots)
        return recon, slots


class ResidualCrossAttnLayer(nn.Module):
    """Single cross-attention layer: latent tokens attend to slot features."""

    def __init__(self, d_model: int, slot_d_model: int, n_heads: int):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(slot_d_model)
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(slot_d_model, d_model)
        self.to_v = nn.Linear(slot_d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor, slot_feats: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.to_q(self.norm_q(x)).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        kv_in = self.norm_kv(slot_feats)
        k = self.to_k(kv_in).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(kv_in).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.out_proj(out)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ResidualDecoderV2(nn.Module):
    """3-layer Cross-Attention Residual Decoder.

    Predicts delta_latent from (L_t_tokens, Slots_t, Slots_{t+1}).
    Zero-initialized output projection ensures the model starts as
    delta=0 and learns updates gradually.

    Architecture:
        latent_proj: [16] -> [192]
        slot_proj: [256] -> [192] (concatenated S_t and S_{t+1})
        3x ResidualCrossAttnLayer (d=192, 6 heads)
        out_proj: [192] -> [16] (zero-initialized)

    Parameters: 1.43M
    """

    def __init__(
        self,
        latent_dim: int = 16,
        slot_dim: int = 128,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 3,
    ):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.slot_proj = nn.Sequential(
            nn.Linear(slot_dim * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList(
            [ResidualCrossAttnLayer(d_model, d_model, n_heads) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        latent_tokens: torch.Tensor,
        slots_t: torch.Tensor,
        slots_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Predict residual delta for latent tokens.

        Args:
            latent_tokens: [B, 1024, 16] current frame latent tokens
            slots_t: [B, 64, 128] current frame slots
            slots_t1: [B, 64, 128] next frame slots (predicted or GT)

        Returns:
            delta: [B, 1024, 16] predicted latent update
        """
        x = self.latent_proj(latent_tokens)
        sf = self.slot_proj(torch.cat([slots_t, slots_t1], dim=-1))
        for layer in self.layers:
            x = layer(x, sf)
        return self.out_proj(self.out_norm(x))


class DynamicsTransformer(nn.Module):
    """Lightweight Transformer for slot dynamics prediction.

    Predicts S_{t+1} = f(S_t) in slot space (64 tokens x 128 dim).
    2 layers, 4 heads, 438K parameters.
    """

    def __init__(
        self,
        n_tokens: int = 64,
        token_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next-step slots.

        Args:
            x: [B, 64, 128] current frame slots

        Returns:
            predicted_slots: [B, 64, 128] predicted next frame slots
        """
        h = self.input_proj(x) + self.pos_embed
        h = self.transformer(h)
        return self.output_proj(h)


class ManifoldProjector(nn.Module):
    """Tiny ConvNet that projects off-manifold latents back onto the VAE manifold.

    Learns a residual correction: z_clean = z_pred + P(z_pred).
    Zero-initialized output layer ensures identity at initialization.

    Architecture: 2x Conv2d(3x3) with GELU, ~46K params.
    Latency: ~0.05-0.1ms on A100 (negligible vs 2.5ms ResidualDecoder).
    """

    def __init__(self, channels: int = 16, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
        # Zero-init last conv so model starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project off-manifold latent back onto VAE manifold.

        Args:
            z: [B, 16, 32, 32] predicted latent (spatial format)

        Returns:
            z_clean: [B, 16, 32, 32] manifold-projected latent
        """
        return z + self.net(z)


class BoundaryDetector(nn.Module):
    """MLP boundary detector for EASY/HARD frame classification.

    Classifies frame transitions based on three features:
    - Mean slot-delta norm
    - Cosine similarity between mean slots
    - Slot-delta variance

    Conservative threshold (88% accuracy, 26% recall) ensures
    keyframes are only triggered at true scene boundaries.
    """

    def __init__(self, n_features: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict boundary probability.

        Args:
            features: [B, 3] boundary features

        Returns:
            logits: [B, 1] boundary logits (sigmoid for probability)
        """
        return self.net(features)

    @torch.no_grad()
    def compute_features(
        self, slots_t: torch.Tensor, slots_t1: torch.Tensor
    ) -> torch.Tensor:
        """Compute boundary detection features from slot pairs.

        Args:
            slots_t: [B, 64, 128] current slots
            slots_t1: [B, 64, 128] next slots

        Returns:
            features: [B, 3]
        """
        delta = slots_t1 - slots_t
        mean_norm = delta.norm(dim=-1).mean(dim=-1, keepdim=True)
        cos_sim = F.cosine_similarity(
            slots_t.mean(dim=1), slots_t1.mean(dim=1), dim=-1
        ).unsqueeze(-1)
        delta_var = delta.norm(dim=-1).var(dim=-1, keepdim=True)
        return torch.cat([mean_norm, cos_sim, delta_var], dim=-1)
