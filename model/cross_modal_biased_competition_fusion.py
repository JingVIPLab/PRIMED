import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """Single-direction cross-attention from query tokens to key/value tokens."""

    def __init__(self, query_dim, kv_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: [B, N_q, query_dim].
            key_value: [B, N_kv, kv_dim].
            attn_mask: Optional additive log-bias broadcastable to [B, heads, N_q, N_kv].
            key_padding_mask: Optional bool mask [B, N_kv], where True marks ignored keys.

        Returns:
            Tensor with shape [B, N_q, query_dim].
        """
        B = query.shape[0]
        N_q = query.shape[1]
        N_kv = key_value.shape[1]

        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        Q = Q.reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = torch.clamp(attn, min=-10.0, max=10.0)

        if attn_mask is not None:
            attn = attn + attn_mask

        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, N_kv):
                raise ValueError(
                    f"key_padding_mask must have shape {(B, N_kv)}, "
                    f"got {tuple(key_padding_mask.shape)}"
                )
            key_padding_mask = key_padding_mask.to(device=attn.device, dtype=torch.bool)
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], torch.finfo(attn.dtype).min
            )

        attn = F.softmax(attn, dim=-1)
        self.last_attention_weights = attn[0].detach()
        attn = self.dropout(attn)

        out = attn @ V
        out = out.transpose(1, 2).reshape(B, N_q, -1)
        return self.out_proj(out)
