import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCompressionAttention(nn.Module):
    """Cross-attention from learnable query tokens to feature tokens."""
    def __init__(self, token_dim, feature_dim, num_heads=8, dropout=0.1):
        """Implementation notes for this block."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        """Implementation notes for this block."""
        
        # Query projection (from tokens)
        self.q_proj = nn.Linear(token_dim, token_dim)
        # Key/Value projection (from input features)
        self.k_proj = nn.Linear(feature_dim, token_dim)
        self.v_proj = nn.Linear(feature_dim, token_dim)
        # Output projection
        self.out_proj = nn.Linear(token_dim, token_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key_value):
        """
        Args:
            query: [B, N_q, token_dim] - learnable tokens
            key_value: [B, N_kv, feature_dim] - input features
        Returns:
            output: [B, N_q, token_dim]
        """
        B = query.shape[0]
        
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [B, 1, token_dim]
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)  # [B, 1, feature_dim]
            
        N_q = query.shape[1]
        N_kv = key_value.shape[1]
        
        Q = self.q_proj(query)  # [B, N_q, token_dim]
        K = self.k_proj(key_value)  # [B, N_kv, token_dim]
        V = self.v_proj(key_value)  # [B, N_kv, token_dim]
        
        # Shape: [B, num_heads, N, head_dim]
        Q = Q.reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, num_heads, N_q, N_kv]
        
        # CRITICAL FIX: Clamp attention scores before softmax to prevent overflow in FP16
        # Shape: [-10, 10]
        attn = torch.clamp(attn, min=-10.0, max=10.0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ V  # [B, num_heads, N_q, head_dim]
        out = out.transpose(1, 2).reshape(B, N_q, -1)  # [B, N_q, token_dim]
        
        out = self.out_proj(out)
        
        return out


class TokenCompressor(nn.Module):
    """Compress high-level visual features into learnable global visual tokens."""
    def __init__(self, feature_dim, token_dim, num_tokens=2, num_heads=8, orthogonal_init=True):
        """Implementation notes for this block."""
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        
        self.tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        if orthogonal_init:

            nn.init.orthogonal_(self.tokens)
        else:
            nn.init.normal_(self.tokens, mean=0, std=0.02)
        
        self.feature_proj = nn.Linear(feature_dim, token_dim)
        
        self.compression_attention = TokenCompressionAttention(
            token_dim=token_dim,
            feature_dim=token_dim,
            num_heads=num_heads
        )
        
        # Layer norm (increased eps for FP16 stability)
        self.norm = nn.LayerNorm(token_dim, eps=1e-6)
        
        # MLP for refinement
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(token_dim * 2, token_dim)
        )
        
    def forward(self, input_features):
        """Run the forward pass."""

        original_shape = input_features.shape
        B = original_shape[0]
        
        if len(original_shape) > 2:
            input_features = input_features.reshape(B, -1, original_shape[-1])
        
        projected = self.feature_proj(input_features)  # [B, N, token_dim]
        
        tokens = self.tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_tokens, token_dim]
        
        compressed = self.compression_attention(tokens, projected)  # [B, num_tokens, token_dim]
        
        compressed = compressed + tokens
        
        # MLP refinement with residual
        compressed = compressed + self.mlp(compressed)
        
        # compressed = self.norm(compressed)
        
        return compressed
    
    def compute_orthogonality_loss(self):
        """Compute auxiliary orthogonal regularization for compressed visual tokens."""
        # Shape: [num_tokens, token_dim]
        normalized_tokens = F.normalize(self.tokens, p=2, dim=1)
        
        # Shape: [num_tokens, num_tokens]
        gram_matrix = torch.mm(normalized_tokens, normalized_tokens.t())
        
        identity = torch.eye(self.num_tokens, device=self.tokens.device, dtype=self.tokens.dtype)
        
        orthogonality_loss = torch.tensor(0.0, device=self.tokens.device, dtype=self.tokens.dtype)
        orthogonality_loss = orthogonality_loss + torch.norm(gram_matrix - identity, p='fro') ** 2
        
        return orthogonality_loss
