import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, emb_dim: int, seq_len: int = 8192):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Create sinusoidal position encodings
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        
        pe = torch.zeros(seq_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return self.pe[:x.size(1), :].unsqueeze(0)


class HyenaFilter(nn.Module):
    """
    Implicit long convolution filter
    Parametrized using an MLP (positional encoding -> filter values)
    """
    def __init__(self, d_model, seq_len, order=2, emb_dim=16):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Positional encoding for filter
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len)
        
        # MLP to generate filter coefficients
        self.filter_fn = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.GELU(),
            nn.Linear(64, d_model)
        )
    
    def forward(self, L):
        """Generate filter of length L"""
        # Get positional embeddings
        pos_emb = self.pos_emb.pe[:L, :]  # (L, emb_dim)
        
        # Generate filter coefficients
        h = self.filter_fn(pos_emb)  # (L, d_model)
        
        return h.unsqueeze(0)  # (1, L, d_model)


class HyenaOperator(nn.Module):
    """
    Hyena operator: Data-controlled recurrence with long convolutions
    Order-2 implementation (3 projections: q, k, v-like)
    """
    def __init__(self, d_model, seq_len, order=2, dropout=0.4):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.order = order
        
        # Input projections (order + 1 projections total)
        self.in_proj = nn.Linear(d_model, (order + 1) * d_model)
        
        # Short convolution for local patterns
        self.short_filter = nn.Conv1d(
            d_model, 
            d_model, 
            kernel_size=3, 
            padding=1,
            groups=d_model  # Depthwise convolution
        )
        
        # Long implicit convolution filters
        self.long_filter = HyenaFilter(d_model, seq_len, order)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        B, L, D = x.shape
        
        # Generate order+1 projections
        projections = self.in_proj(x)  # (B, L, (order+1)*D)
        projections = projections.view(B, L, self.order + 1, D)
        
        # Split into individual projections
        x0 = projections[:, :, 0, :]  # First projection
        x1 = projections[:, :, 1, :]  # Second projection
        x2 = projections[:, :, 2, :]  # Third projection (order=2)
        
        # Apply short convolution to x0
        x0_conv = self.short_filter(x0.transpose(1, 2)).transpose(1, 2)
        
        # Get long convolution filter
        h = self.long_filter(L)  # (1, L, D)
        
        # Hyena recurrence (order=2):
        # y = (x0 * h) * x1 * x2
        
        # Step 1: Long conv on x0
        # Using FFT-based convolution for efficiency
        x0_fft = torch.fft.rfft(x0_conv, n=2*L, dim=1)
        h_fft = torch.fft.rfft(h.squeeze(0), n=2*L, dim=0)
        
        # Multiply in frequency domain
        y_fft = x0_fft * h_fft.unsqueeze(0)
        y = torch.fft.irfft(y_fft, n=2*L, dim=1)[:, :L, :]
        
        # Step 2: Element-wise gating with x1
        y = y * x1
        
        # Step 3: Element-wise gating with x2
        y = y * x2
        
        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y


class HyenaBlock(nn.Module):
    """
    Hyena block: Hyena operator + MLP
    Similar structure to Transformer block
    """
    def __init__(self, d_model, seq_len, dim_feedforward, dropout=0.4, drop_path=0.3):
        super().__init__()
        
        self.drop_path = drop_path
        # Hyena operator
        self.hyena = HyenaOperator(d_model, seq_len, order=2, dropout=dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Hyena with residual
        if self.training and torch.rand(1).item() < self.drop_path:
            return x
        x = x + self.hyena(self.norm1(x))
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class RNASeqHyena(nn.Module):
    """
    Hyena model for RNA-Seq classification
    """
    def __init__(
        self,
        input_dim,
        num_classes,
        d_model=192,
        num_layers=4,
        dim_feedforward=768,
        dropout=0.4
    ):
        super().__init__()
        self.d_model = d_model
        
        self.chunk_size = 10
        self.seq_len = input_dim // self.chunk_size  # 4620 // 10 = 462 positions
        self.input_dim = input_dim
        # Input projection
        self.input_projection = nn.Linear(self.chunk_size, d_model)
        
        self.pos_encoder = PositionalEmbedding(d_model, self.seq_len)
        
        # Hyena blocks
        self.layers = nn.ModuleList([
            HyenaBlock(d_model, self.seq_len, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        x: (batch, input_dim)
        """
        batch_size = x.size(0)
        # Add sequence dimension
        x = x.view(batch_size, self.seq_len, self.chunk_size)

        
        # Project to model dimension
        x = self.input_projection(x)  # (batch, 1, d_model)
        x = x + self.pos_encoder(x)
        # Apply Hyena blocks
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling (mean over sequence)
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Normalize
        x = self.norm(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits