"""
Projectors: Bridge between Qwen hidden states and SAM3 prompt embeddings.

Maps <SEG_A>/<SEG_B> token hidden states from LLM space to SAM prompt space.
"""

import torch
import torch.nn as nn


class SegTokenProjector(nn.Module):
    """
    2-layer MLP that projects LLM hidden states into a single SAM prompt embedding.

    Architecture:
        Linear(llm_dim → hidden_dim) → GELU → Linear(hidden_dim → sam_dim)

    Output: (B, sam_dim)
    """

    def __init__(
        self,
        llm_dim: int = 3584,
        sam_dim: int = 256,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, sam_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, llm_dim) from Qwen at <SEG> token positions

        Returns:
            (B, sam_dim) prompt embeddings for SAM
        """
        return self.proj(hidden_states)


class MultiTokenProjector(nn.Module):
    """
    Projects LLM hidden states into N tokens of sam_dim each,
    preserving more information than single-token compression.

    Variant "reshape":
        Direct reshape: (B, 2048) → view(B, 8, 256). Zero learned params.
        Requires llm_dim == num_tokens * sam_dim.

    Variant "learned":
        Linear(llm_dim, llm_dim) → GELU → Linear(llm_dim, N*sam_dim) → reshape.
        Learns an optimal decomposition into N semantic tokens.

    Output: (B, num_tokens, sam_dim)
    """

    def __init__(
        self,
        llm_dim: int = 2048,
        sam_dim: int = 256,
        num_tokens: int = 8,
        variant: str = "reshape",
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.sam_dim = sam_dim
        self.variant = variant

        if variant == "reshape":
            assert llm_dim == num_tokens * sam_dim, (
                f"For reshape variant, llm_dim ({llm_dim}) must equal "
                f"num_tokens * sam_dim ({num_tokens} * {sam_dim} = {num_tokens * sam_dim})"
            )
            self.proj = None
        elif variant == "learned":
            out_dim = num_tokens * sam_dim
            self.proj = nn.Sequential(
                nn.Linear(llm_dim, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, out_dim),
            )
        else:
            raise ValueError(f"Unknown variant: {variant}. Use 'reshape' or 'learned'.")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, llm_dim) from Qwen at <SEG> token positions

        Returns:
            (B, num_tokens, sam_dim) multi-token prompt embeddings
        """
        if self.proj is not None:
            x = self.proj(hidden_states)  # (B, N * sam_dim)
        else:
            x = hidden_states
        return x.view(-1, self.num_tokens, self.sam_dim)


class QFormerProjector(nn.Module):
    """
    Q-Former style projector that cross-attends over multiple Qwen layers.

    Instead of compressing a single 2048-dim vector, stacks the last K layers
    at the SEG token position into a sequence (B, K, 2048), then uses N learnable
    query tokens with cross-attention to extract N meaningful SAM-space tokens.

    Each Qwen layer captures different information (lower=structural, upper=semantic),
    giving the cross-attention real diversity to attend over.

    Input:  (B, K, llm_dim)  — K layer hidden states at SEG position
    Output: (B, num_tokens, sam_dim)
    """

    def __init__(
        self,
        llm_dim: int = 2048,
        sam_dim: int = 256,
        num_tokens: int = 8,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.sam_dim = sam_dim

        # Project LLM dim to SAM dim for K,V
        self.kv_proj = nn.Linear(llm_dim, sam_dim)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, sam_dim) * 0.02)

        # Cross-attention: queries attend to layer stack
        self.cross_attn = nn.MultiheadAttention(
            sam_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(sam_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(sam_dim, sam_dim * 4),
            nn.GELU(),
            nn.Linear(sam_dim * 4, sam_dim),
        )
        self.ffn_norm = nn.LayerNorm(sam_dim)

    def forward(self, layer_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_stack: (B, K, llm_dim) — stacked hidden states from K Qwen layers

        Returns:
            (B, num_tokens, sam_dim) multi-token prompt embeddings
        """
        B = layer_stack.shape[0]
        kv = self.kv_proj(layer_stack)  # (B, K, 256)
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, N, 256)
        out, _ = self.cross_attn(q, kv, kv)  # (B, N, 256)
        out = self.norm(out + q)  # residual + norm
        out = self.ffn_norm(out + self.ffn(out))  # FFN + residual + norm
        return out
