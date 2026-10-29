"""Single-transform, early, late and hybrid (transformer) fusion models.

Input layouts:
  single, early          (B, 3, H, W)     one RGB image, or 3 grayscale transforms as channels
  late, hybrid           (B, 3, H, W)     channel i feeds branch i (one grayscale transform each)
  mb_late, mb_hybrid     (B, 3, 3, H, W)  branch i gets a whole viridis RGB image (encoding control)

Attribute names match the checkpoints used in the paper, so those load with load_state_dict.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

MODEL_TYPES = ('single', 'early', 'late', 'hybrid', 'mb_late', 'mb_hybrid')


def efficientnet_features(pretrained, in_channels):
    """EfficientNet-B0 feature extractor; a 1-channel stem gets the RGB stem weights averaged."""
    features = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None).features
    if in_channels == 1:
        rgb = features[0][0]
        gray = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        gray.weight.data = rgb.weight.data.mean(dim=1, keepdim=True)
        features[0][0] = gray
    return features


class SingleModel(nn.Module):
    """EfficientNet-B0 with a 2-class head (single transform and early fusion)."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        net = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        net.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, num_classes))
        self.features, self.avgpool, self.classifier = net.features, net.avgpool, net.classifier

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)).flatten(1))


class BranchFeatures(nn.Module):
    """Three EfficientNet-B0 branches -> list of three (B, 1280) GAP features."""

    def __init__(self, pretrained, rgb_branches):
        super().__init__()
        self.rgb_branches = rgb_branches
        self.feature_extractors = nn.ModuleList(
            [efficientnet_features(pretrained, 3 if rgb_branches else 1) for _ in range(3)])

    def branch_features(self, x):
        inputs = [x[:, i] for i in range(3)] if self.rgb_branches else [x[:, i:i + 1] for i in range(3)]
        return [F.adaptive_avg_pool2d(f(xi), 1).flatten(1) for f, xi in zip(self.feature_extractors, inputs)]


class LateFusionModel(BranchFeatures):
    def __init__(self, num_classes=2, pretrained=True, rgb_branches=False):
        super().__init__(pretrained, rgb_branches)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(1280 * 3, 512), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(512, num_classes))
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(torch.cat(self.branch_features(x), dim=1))


class MultiHeadAttentionWithWeights(nn.Module):
    """Multi-head self-attention that also returns the (B, heads, 3, 3) weights."""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model, self.nhead, self.head_dim = d_model, nhead, d_model // nhead
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.q_linear, self.k_linear, self.v_linear, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = (lin(x).view(b, n, self.nhead, self.head_dim).transpose(1, 2)
                   for lin in (self.q_linear, self.k_linear, self.v_linear))
        weights = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        out = (self.dropout(weights) @ v).transpose(1, 2).reshape(b, n, self.d_model)
        return self.out_proj(out), weights


class PreLNEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithWeights(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        for lin in (self.linear1, self.linear2):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        attn, weights = self.self_attn(self.norm1(x))
        x = x + self.dropout1(attn)
        x = x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x))))))
        return x, weights


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [PreLNEncoderLayer(d_model, nhead, d_model * 4, dropout) for _ in range(num_layers)])

    def forward(self, x):
        weights = []
        for layer in self.layers:
            x, w = layer(x)
            weights.append(w)
        return x, weights


class TransformerFusionModel(BranchFeatures):
    """Hybrid fusion: branch features + channel embeddings -> Pre-LN encoder -> final LayerNorm -> mean."""

    def __init__(self, num_classes=2, d_model=1280, nhead=8, num_layers=3, dropout=0.2,
                 pretrained=True, rgb_branches=False):
        super().__init__(pretrained, rgb_branches)
        self.channel_embeddings = nn.Parameter(torch.empty(3, d_model).normal_(0, 0.02))
        self.transformer = Encoder(d_model, nhead, num_layers, dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 512),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_classes))
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_attention_weights=False):
        tokens = torch.stack(self.branch_features(x), dim=1) + self.channel_embeddings
        encoded, weights = self.transformer(tokens)
        logits = self.classifier(self.final_norm(encoded).mean(dim=1))
        return (logits, weights) if return_attention_weights else logits


def build_model(model_type, pretrained=True):
    if model_type in ('single', 'early'):
        model = SingleModel(pretrained=pretrained)
    elif model_type in ('late', 'mb_late'):
        model = LateFusionModel(pretrained=pretrained, rgb_branches=model_type == 'mb_late')
    elif model_type in ('hybrid', 'mb_hybrid'):
        model = TransformerFusionModel(pretrained=pretrained, rgb_branches=model_type == 'mb_hybrid')
    else:
        raise ValueError(f'unknown model_type {model_type!r}; choose from {MODEL_TYPES}')
    # The paper's models were all trained with this BatchNorm epsilon.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
    return model


if __name__ == '__main__':
    expected = {'single': 4_010_110, 'late': 13_988_534, 'hybrid': 71_719_094}
    for name, n in expected.items():
        got = sum(p.numel() for p in build_model(name, pretrained=False).parameters())
        assert got == n, f'{name}: {got:,} parameters, expected {n:,}'
    logits, attn = build_model('hybrid', pretrained=False).eval()(torch.randn(2, 3, 64, 64), True)
    assert logits.shape == (2, 2) and len(attn) == 3 and attn[0].shape == (2, 8, 3, 3)
    assert build_model('mb_late', pretrained=False)(torch.randn(2, 3, 3, 64, 64)).shape == (2, 2)
    print('models OK: parameter counts match Table 2')
