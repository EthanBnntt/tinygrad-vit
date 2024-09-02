"""
This file contains the implementation of the ViT model.
"""

from einops import rearrange
import numpy as np
from tinygrad import Tensor, nn

def convert_to_patches(x: Tensor, patch_height: int = 16, patch_width: int = 16) -> Tensor:
    _, __, height, width = x.shape
    assert height % patch_height == 0, "height must be divisible by patch_height"
    assert width % patch_width == 0, "width must be divisible by patch_width"
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=patch_height, pw=patch_width)

class MultiHeadAttention:
    def __init__(self,
                 dim: int = 768,
                 num_heads: int = 12,
                 dropout_p: float = 0.1,
                 bias: bool = True):

        self.dim = dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)

        self.out = nn.Linear(dim, dim, bias=bias)

    def __call__(self, x: Tensor, attn_mask: Tensor = None, is_causal: bool = False) -> Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        q, k, v = [rearrange(t, "b n (h d) -> b h n d", h=self.num_heads) for t in (q, k, v)]
        out = q.scaled_dot_product_attention(k,
                                             v,
                                             attn_mask=attn_mask,
                                             dropout_p=self.dropout_p,
                                             is_causal=is_causal)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out(out)

class MLP:
    def __init__(self,
                 dim: int = 768,
                 hidden_dim: int = 3072,
                 dropout_p: float = 0.1,
                 bias: bool = True,
                 gating: bool = False):

        self.dropout_p = dropout_p

        self.in_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.gating_proj = nn.Linear(dim, hidden_dim, bias=bias) if gating else None
        self.out_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: Tensor):
        h = self.in_proj(x).silu().dropout(self.dropout_p)
        if self.gating_proj is not None:
            h = h * self.gating_proj(x).sigmoid()
        return self.out_proj(h)

class TransformerLayer:
    def __init__(self,
                 dim: int = 768,
                 hidden_dim: int = 3072,
                 num_heads: int = 12,
                 dropout_p: float = 0.1,
                 bias: bool = True,
                 mlp_gating: bool = False):

        self.attn_norm = nn.RMSNorm(dim)
        self.feed_forward_norm = nn.RMSNorm(dim)

        self.multi_head_attention = MultiHeadAttention(dim, num_heads, dropout_p, bias)
        self.feed_forward = MLP(dim, hidden_dim, dropout_p, bias, mlp_gating)

    def __call__(self, x: Tensor):
        h = x + self.multi_head_attention(self.attn_norm(x))
        h = h + self.feed_forward(self.feed_forward_norm(h))

        return h

class Transformer:
    def __init__(self,
                 dim: int = 768,
                 hidden_dim: int = 3072,
                 num_heads: int = 12,
                 dropout_p: float = 0.1,
                 bias: bool = True,
                 num_layers: int = 12,
                 mlp_gating: bool = False):
        self.dim = dim
        self.layers = [
            TransformerLayer(dim, hidden_dim, num_heads, dropout_p, bias, mlp_gating)
            for _ in range(num_layers)
        ]

    def __call__(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

class ViTModel:
    def __init__(self,
                 image_width: int = 256,
                 image_height: int = 256,
                 patch_width: int = 16,
                 patch_height: int = 16,
                 channels: int = 3,
                 embed_dim: int = 768,
                 hidden_dim: int = 3072,
                 output_dim: int = 10,
                 num_heads: int = 12,
                 dropout_p: float = 0.1,
                 bias: bool = True,
                 num_layers: int = 12,
                 mlp_gating: bool = False):

        self.patch_width = patch_width
        self.patch_height = patch_height

        self.patch_dim = patch_width * patch_height * channels
        self.num_patches = (image_width // patch_width) * (image_height // patch_height)

        self.patch_norm = nn.RMSNorm(self.patch_dim)
        self.patch_proj = nn.Linear(self.patch_dim, embed_dim, bias=bias)
        self.cls_embedding = Tensor(np.random.randn(1, 1, embed_dim)).float()
        self.pos_embedding = Tensor(np.random.randn(1, self.num_patches, embed_dim)).float()

        self.transformer = Transformer(
            embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_p=dropout_p,
            bias=bias,
            num_layers=num_layers,
            mlp_gating=mlp_gating
        )
        self.classifier = nn.Linear(embed_dim, output_dim, bias=bias)

    def __call__(self, x: Tensor):
        # Compute the patch embeddings
        patches = self.patch_proj(self.patch_norm(convert_to_patches(x,
                                                     patch_height=self.patch_height,
                                                     patch_width=self.patch_width)))

        # Add the positional embeddings
        patches = patches + self.pos_embedding

        # Concatenate the class token to the patches
        cls_embedding = self.cls_embedding.expand(patches.shape[0], -1, -1)
        patches = patches.cat(cls_embedding, dim=1)

        # Pass the patches through the transformer
        out = self.transformer(patches.float())

        # Return the predictions and the intermediate outputs
        return self.classifier(out[:, 0, :]), out
