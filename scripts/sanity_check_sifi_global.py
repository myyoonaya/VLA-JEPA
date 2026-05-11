import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.VLA_JEPA import VLA_JEPA


def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def collect_trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def build_fake_batch(batch_size, num_views, num_frames, image_size, action_horizon, action_dim, state_dim):
    batch = []
    for _ in range(batch_size):
        images = [
            Image.fromarray(np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8))
            for _ in range(num_views)
        ]
        video = np.random.randint(
            0,
            255,
            (num_views, num_frames, image_size, image_size, 3),
            dtype=np.uint8,
        )
        batch.append(
            {
                "image": images,
                "video": video,
                "lang": "pick up the object and place it into the target container",
                "action": np.random.uniform(-1.0, 1.0, size=(action_horizon, action_dim)).astype(np.float32),
                "state": np.random.uniform(-1.0, 1.0, size=(1, state_dim)).astype(np.float32),
            }
        )
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/config/vlajepa_robot_ft.yaml")
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--vjepa-path", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--freeze-vlm", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sanity check.")

    cfg = OmegaConf.load(args.config)
    cfg.framework.qwenvl.base_vlm = args.qwen_path
    cfg.framework.qwenvl.attn_implementation = "flash_attention_2"
    cfg.framework.vj2_model.base_encoder = args.vjepa_path
    cfg.framework.future_interface.enabled = True
    cfg.framework.future_interface.token_dim = cfg.framework.qwenvl.vl_hidden_dim
    cfg.framework.future_interface.n_prog = 1
    cfg.framework.future_interface.n_int = 0
    cfg.framework.future_interface.n_obj = 0
    cfg.framework.future_interface.losses.lambda_bridge = 0.1

    torch.manual_seed(0)
    np.random.seed(0)

    model = VLA_JEPA(cfg).to("cuda")
    model.train()
    if args.freeze_vlm:
        freeze_module(model.qwen_vl_interface)

    trainable_params = collect_trainable_parameters(model)
    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    batch = build_fake_batch(
        batch_size=args.batch_size,
        num_views=2,
        num_frames=cfg.framework.vj2_model.num_frames,
        image_size=args.image_size,
        action_horizon=cfg.framework.action_model.action_horizon,
        action_dim=cfg.framework.action_model.action_dim,
        state_dim=cfg.framework.action_model.state_dim,
    )

    torch.cuda.reset_peak_memory_stats()
    last_output = None
    last_total_loss = None
    for step in range(args.train_steps):
        optimizer.zero_grad(set_to_none=True)
        output = model.forward(batch)
        total_loss = sum(output.values())
        total_loss.backward()
        optimizer.step()

        last_output = output
        last_total_loss = total_loss
        loss_values = {key: float(value.detach().cpu()) for key, value in output.items()}
        print(f"step={step + 1}/{args.train_steps}", loss_values, "total_loss=", float(total_loss.detach().cpu()))

    print("final_losses:", {key: float(value.detach().cpu()) for key, value in last_output.items()})
    print("final_total_loss:", float(last_total_loss.detach().cpu()))
    print("peak_memory_gb:", round(torch.cuda.max_memory_allocated() / 1024**3, 3))


if __name__ == "__main__":
    main()
