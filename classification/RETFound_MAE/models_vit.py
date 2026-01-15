# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


import torch
import torch.nn as nn
from functools import partial
import timm

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling and multiple classification heads """

    def __init__(self, num_labels=3, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.num_labels = num_labels

        # --- Handle global pooling setup ---
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

        # --- Multi-head classification heads (one per label) ---
        embed_dim = kwargs['embed_dim']
        self.heads = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_labels)])

        # Optional: initialize heads
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward_features(self, x, attn_mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1, keepdim=True)  # average over patches
            features = self.fc_norm(x)
        else:
            x = self.norm(x)
            features = x[:, 0]  # use CLS token

        return features

    def forward(self, x, attn_mask=None):
        features = self.forward_features(x, attn_mask)
        # Pass features through each head independently
        outputs = [head(features) for head in self.heads]
        outputs = torch.cat(outputs, dim=1).squeeze()  # [B, num_labels]
        return outputs


# Factory function for your model
def RETFound_mae(num_labels=3, **kwargs):
    model = VisionTransformer(
        num_labels=num_labels,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model



class ExponentialMovingAverage:
    """Maintain an exponential moving average of model parameters."""
    def __init__(self, parameters, decay=0.999):
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]

    @torch.no_grad()
    def update(self, parameters):
        for s, p in zip(self.shadow_params, [p for p in parameters if p.requires_grad]):
            s.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, parameters):
        for s, p in zip(self.shadow_params, [p for p in parameters if p.requires_grad]):
            p.data.copy_(s.data)


