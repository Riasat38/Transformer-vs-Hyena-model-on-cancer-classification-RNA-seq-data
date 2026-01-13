import torch
import torch.nn as nn
import numpy as np

class RNASeqTransformer(nn.Module):
    """
    Transformer-based classifier with ADAPTIVE input dimension.
    Automatically adjusts to your data size (300 PCA, 900 variance, or any size).
    """
    def __init__(self, input_dim, num_classes, d_model=None, nhead=8,
                 num_layers=4, dim_feedforward=None, dropout=0.4):
        """
        Args:
            input_dim: Number of input features (genes/PCA components)
                      - Automatically detected from your data
                      - Example: 300 (PCA) or 900 (Variance)
            num_classes: Number of cancer types (5)
            d_model: Embedding dimension (auto-set if None)
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension (auto-set if None)
            dropout: Dropout rate
        """
        super(RNASeqTransformer, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # AUTO-SET d_model based on input_dim if not provided
        if d_model is None:
            # Scale embedding dimension with input
            # Rule: min 64, max 256, proportional to input
            d_model = min(256, max(64, input_dim // 4))

        self.d_model = d_model

        # AUTO-SET dim_feedforward if not provided
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        print(f"✓ Input dimension: {input_dim}")
        print(f"✓ Embedding dimension: {d_model}")
        print(f"✓ Feedforward dimension: {dim_feedforward}")
        print(f"✓ Attention heads: {nhead}")

        # Project input features to d_model dimension
        self.embedding = nn.Linear(1, d_model)

        # Positional encoding (adaptive to input_dim)
        self.pos_encoding = nn.Parameter(
            self._get_positional_encoding(input_dim, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # Adaptive FC layers (scale with d_model)
        fc_hidden = max(128, d_model * 2)
        self.fc1 = nn.Linear(d_model, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.fc3 = nn.Linear(fc_hidden // 2, num_classes)

        self.relu = nn.ReLU()

    def _get_positional_encoding(self, seq_len, d_model):
        """Generate positional encoding (same as before)"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, 1)
               seq_len = 300 (PCA) or 900 (Variance)
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Embed: (batch, seq_len, 1) → (batch, seq_len, d_model)
        x = self.embedding(x)

        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x)  # (batch, d_model, 1)
        x = x.squeeze(-1)  # (batch, d_model)

        # Classification head
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x