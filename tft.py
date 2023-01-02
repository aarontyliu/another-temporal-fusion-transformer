from typing import Optional, Tuple

import torch
from einops.layers.torch import Rearrange
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_features: int, embed_dim: int):
        """Encoder: embeds each feature into a vector of dimension embed_dim.

        Args:
            in_features (int): Number of features in the input tensor.
            embed_dim (int): Embedding dimension.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            Rearrange("b l d -> b d l"),
            nn.Conv1d(
                in_features, in_features * embed_dim, 1, groups=in_features, bias=False
            ),
            Rearrange("b d l -> l b d"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch, dim)
        """
        return self.encoder(x)


class LSTM(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 1) -> None:
        """LSTM encoder.

        Args:
            d_model (int): Dimension of the input.
            num_layers (int, optional): Number of layers. Defaults to 1.
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.lstm_encoder = nn.LSTM(d_model, d_model, num_layers)
        self.lstm_decoder = nn.LSTM(d_model, d_model, num_layers)
        self.gate = GatedLinearUnit(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        past_inputs: torch.Tensor,
        known_future_inputs: torch.Tensor,
        c_h: Optional[torch.Tensor] = None,
        c_c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            past_inputs (torch.Tensor): Input tensor of shape (seq_len, batch, dim)
            known_future_inputs (torch.Tensor): Input tensor of shape (seq_len, batch, dim)
            c_h (Optional[torch.Tensor], optional): Hidden state of shape (num_layers, batch, dim). Defaults to None.
            c_c (Optional[torch.Tensor], optional): Cell state of shape (num_layers, batch, dim). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch, dim)
        """
        n = past_inputs.shape[1]
        c_h = torch.zeros(self.num_layers, n, self.d_model) if c_h is None else c_h
        c_c = torch.zeros(self.num_layers, n, self.d_model) if c_c is None else c_c

        inputs = torch.cat([past_inputs, known_future_inputs], dim=0)
        encoded, (h, c) = self.lstm_encoder(past_inputs, (c_h, c_c))
        decoded, _ = self.lstm_decoder(known_future_inputs, (h, c))
        out = torch.cat([encoded, decoded], dim=0)
        return self.layer_norm(inputs + self.gate(out))


class GatedLinearUnit(nn.Module):
    def __init__(self, num_features: int) -> None:
        """Gated Linear Unit (GLU).

        Args:
            num_features (int): Number of features in the input/output tensor.
        """
        super().__init__()
        self.linear = nn.Linear(num_features, num_features * 2)
        self.gating = nn.GLU(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.gating(x)


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        hidden_features: Optional[int] = None,
        dropout: float = 0.1,
        hidden_context_features: Optional[int] = None,
    ) -> None:
        """Gated Residual Network (GRN). This building block provides a flexible way
            to apply non-linear transformations to the input tensor

        Args:
            in_features (int): Number of input features.
            out_features (int, optional): Number of output features. Defaults to None.
            hidden_features (int, optional): Number of hidden features. Defaults to None.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            hidden_context_features (Optional[int], optional): Number of hidden context
                features. Defaults to None.
        """
        super().__init__()

        out_features = in_features if out_features is None else out_features
        hidden_features = (
            max(in_features, out_features)
            if hidden_features is None
            else hidden_features
        )

        if in_features != out_features:
            self.project = nn.Linear(in_features, out_features)
        if hidden_context_features is not None:
            self.context = nn.Linear(
                hidden_context_features, hidden_features, bias=False
            )
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.elu = nn.ELU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(out_features)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch, dim)
            c (Optional[torch.Tensor], optional): Context tensor of shape (seq_len, batch, dim).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch, dim)
        """
        if self.in_features != self.out_features:
            residual = self.project(x)
        else:
            residual = x
        x = self.fc1(x) + 0 if c is None else self.context(c)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.layer_norm(residual + self.glu(x))


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_features: int,
        out_features: Optional[int] = None,
        dropout: float = 0.1,
        hidden_context_features: Optional[int] = None,
    ) -> None:
        """Variable Selection Network (VSN).
            This building block provides instance-wise variable selection.

        Args:
            embed_dim (int): Embedding dimension of each feature.
            num_features (int): Number of features.
            out_features (Optional[int], optional): Number of output features.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            hidden_context_features (Optional[int], optional): Number of hidden context
                features. Defaults to None.
        """
        super().__init__()
        out_features = embed_dim if out_features is None else out_features
        self.num_features = num_features
        self.flatten_grn = GatedResidualNetwork(
            num_features * embed_dim,
            out_features=num_features,
            dropout=dropout,
            hidden_context_features=hidden_context_features,
        )
        self.grns = nn.ModuleList([])
        for _ in range(num_features):
            self.grns.append(
                GatedResidualNetwork(embed_dim, out_features, dropout=dropout)
            )

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch, dim)
            c (Optional[torch.Tensor], optional): Context tensor of shape
                (seq_len, batch, dim). Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor of
                shape (seq_len, batch, dim) and weights of shape (seq_len, batch, dim)
        """
        weights = self.flatten_grn(x, c)
        weights = torch.softmax(weights, dim=-1)
        xs = x.chunk(self.num_features, dim=-1)
        x = torch.stack([grn(x) for grn, x in zip(self.grns, xs)], dim=-1)
        out = torch.einsum("... e f, ... f -> ... e", x, weights)
        return out, weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ) -> None:
        """InterpretableMultiHeadAttention.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            bias (bool, optional): If True, add bias to input/output projection layers. Defaults to True.
            kdim (Optional[int], optional): Key dimension. Defaults to None.
            vdim (Optional[int], optional): Value dimension. Defaults to None.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.w_k = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.w_v = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.to_head = Rearrange("l n (h e) -> n h l e", h=num_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            q (torch.Tensor): Query tensor of shape (seq_len, batch, dim)
            k (torch.Tensor): Key tensor of shape (seq_len, batch, dim)
            v (torch.Tensor): Value tensor of shape (seq_len, batch, dim)
            attn_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape
                (seq_len, batch, dim). Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor of shape (seq_len, batch, dim)
                and attention weights of shape (seq_len, batch, dim)
        """
        src_len = q.size(0)
        tgt_len = k.size(0)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        q = q * self.scaling

        q, k = map(self.to_head, (q, k))
        attn_output_weights = torch.einsum("n h l e, n h m e -> n h l m", q, k)
        attn_mask = self._generate_attn_mask(
            src_len, tgt_len, attn_output_weights.device
        )
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        attn_output_weights = attn_output_weights.masked_fill(attn_mask, float("-inf"))
        attn_output_weights = self.softmax(attn_output_weights)
        attn_output_weights = self.dropout(attn_output_weights)
        attn_output_weights = attn_output_weights.mean(dim=1)  # Average over heads
        attn_out = torch.einsum("n l m, m n e -> l n e", attn_output_weights, v)
        attn_out = self.out_proj(attn_out)
        return attn_out, attn_output_weights

    @staticmethod
    def _generate_attn_mask(
        src_len: int, tgt_len, device: torch.device
    ) -> torch.Tensor:
        attn_mask = torch.triu(torch.ones(src_len, tgt_len, device=device), diagonal=1)
        return attn_mask.bool()


class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        """Temporal Self-Attention.

        Args:
            d_model (int): Dimension of the input tensor.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.interp_mha = InterpretableMultiHeadAttention(d_model, num_heads, dropout)
        self.gate = GatedLinearUnit(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch, dim).
            **kwargs: Additional arguments for nn.MultiheadAttention forward pass.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch, dim).
        """
        out, weights = self.interp_mha(x, x, x)
        return self.layer_norm(x + self.gate(out)), weights


class TemporalFusionDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        hidden_context_features: Optional[int] = None,
    ) -> None:
        """Temporal Fusion Decoder.

        Args:
            d_model (int): Dimension of the input tensor.
            num_heads (int, optional): Number of attention heads. Defaults to 1.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            hidden_context_features (Optional[int], optional): Number of hidden features in context. Defaults to None.
            causal (bool, optional): Causal attention. Defaults to True.
        """
        super().__init__()
        self.static_enrichment = GatedResidualNetwork(
            d_model,
            dropout=dropout,
            hidden_context_features=hidden_context_features,
        )
        self.temporal_sa = TemporalSelfAttention(d_model, num_heads, dropout)
        self.ffn = GatedResidualNetwork(
            d_model, hidden_features=dim_feedforward, dropout=dropout
        )
        self.gate = GatedLinearUnit(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch, dim).
            c (Optional[torch.Tensor], optional): Context tensor. Defaults to None.
            start_idx (int, optional): Start index for decoding. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Output tensor of shape (seq_len, batch, dim), attention weights.
        """
        residual = x
        x = self.static_enrichment(x, c)
        x, weights = self.temporal_sa(x)
        weights = weights
        x = self.ffn(x)
        return self.layer_norm(residual + self.gate(x)), weights
