import torch
import torch.nn as nn


class FutureTokenPredictor(nn.Module):
    """Predict structured future-interface tokens from current VLM context."""

    def __init__(
        self,
        vlm_dim: int,
        token_dim: int,
        n_prog: int = 1,
        n_int: int = 1,
        n_obj: int = 1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.n_prog = n_prog
        self.n_int = n_int
        self.n_obj = n_obj

        self.context_proj = nn.Sequential(
            nn.LayerNorm(vlm_dim),
            nn.Linear(vlm_dim, token_dim),
            nn.GELU(),
        )
        self.prog_head = nn.Linear(token_dim, n_prog * token_dim)
        self.int_head = nn.Linear(token_dim, n_int * token_dim)
        self.obj_head = nn.Linear(token_dim, n_obj * token_dim)

    def forward(self, vlm_hidden: torch.Tensor, state: torch.Tensor = None, language_emb: torch.Tensor = None):
        if vlm_hidden.ndim != 3:
            raise ValueError(f"Expected vlm_hidden with shape [B, L, D], got {tuple(vlm_hidden.shape)}")

        pooled_context = vlm_hidden.mean(dim=1)
        context = self.context_proj(pooled_context)
        batch_size = context.shape[0]

        return {
            "z_prog": self.prog_head(context).view(batch_size, self.n_prog, self.token_dim),
            "z_int": self.int_head(context).view(batch_size, self.n_int, self.token_dim),
            "z_obj": self.obj_head(context).view(batch_size, self.n_obj, self.token_dim),
        }
