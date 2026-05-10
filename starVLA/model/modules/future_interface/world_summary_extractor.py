import torch
import torch.nn as nn


class WorldSummaryExtractor(nn.Module):
    """Build teacher future summaries from future latent tokens and optional routing masks."""

    def forward(self, future_latents: torch.Tensor, masks: dict | None = None):
        if future_latents.ndim != 3:
            raise ValueError(f"Expected future_latents with shape [B, N, D], got {tuple(future_latents.shape)}")

        masks = masks or {}
        global_summary = future_latents.mean(dim=1, keepdim=True)

        return {
            "z_prog_star": global_summary,
            "z_int_star": self.masked_pool(future_latents, masks.get("interaction"), fallback=global_summary),
            "z_obj_star": self.masked_pool(future_latents, masks.get("object"), fallback=global_summary),
        }

    @staticmethod
    def masked_pool(future_latents: torch.Tensor, mask: torch.Tensor | None, fallback: torch.Tensor):
        if mask is None:
            return fallback

        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        if mask.ndim != 3:
            raise ValueError(f"Expected mask with shape [B, N] or [B, N, 1], got {tuple(mask.shape)}")
        if mask.shape[:2] != future_latents.shape[:2]:
            raise ValueError(
                f"Mask shape {tuple(mask.shape[:2])} must match latent token shape {tuple(future_latents.shape[:2])}"
            )

        mask = mask.to(device=future_latents.device, dtype=future_latents.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (future_latents * mask).sum(dim=1, keepdim=True) / denom
        has_tokens = mask.sum(dim=1, keepdim=True) > 0
        return torch.where(has_tokens, pooled, fallback)
