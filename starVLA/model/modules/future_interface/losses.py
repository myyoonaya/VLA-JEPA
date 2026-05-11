import torch
import torch.nn.functional as F


def bridge_loss(z_hat: dict, z_star: dict):
    """L2 plus cosine distance for matching predicted future tokens to teacher summaries."""
    key_pairs = (
        ("z_prog", "z_prog_star"),
        ("z_int", "z_int_star"),
        ("z_obj", "z_obj_star"),
    )
    losses = []
    for pred_key, target_key in key_pairs:
        if pred_key not in z_hat or target_key not in z_star:
            continue
        pred = z_hat[pred_key]
        if pred.numel() == 0 or pred.shape[1] == 0:
            continue
        target = z_star[target_key].to(device=pred.device, dtype=pred.dtype)
        if target.numel() == 0 or target.shape[1] == 0:
            continue
        if target.shape[1] == 1 and pred.shape[1] != 1:
            target = target.expand(-1, pred.shape[1], -1)
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch for {pred_key}/{target_key}: {tuple(pred.shape)} vs {tuple(target.shape)}")
        l2 = F.mse_loss(pred, target)
        cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        losses.append(l2 + cosine)

    if not losses:
        raise ValueError("bridge_loss received no matching future token keys")
    return torch.stack(losses).mean()
