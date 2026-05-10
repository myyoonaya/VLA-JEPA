from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


def test_param_lr_groups_exclude_already_frozen_parameters():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = torch.nn.Linear(2, 2)
            self.frozen = torch.nn.Linear(2, 2)
            for param in self.frozen.parameters():
                param.requires_grad = False

    cfg = OmegaConf.create(
        {
            "trainer": {
                "learning_rate": {"base": 1e-4},
                "freeze_modules": "",
            }
        }
    )
    model = TinyModel()

    param_groups = build_param_lr_groups(model, cfg)
    grouped_params = [param for group in param_groups for param in group["params"]]

    assert grouped_params
    assert all(param.requires_grad for param in grouped_params)
    assert not any(param is frozen for param in grouped_params for frozen in model.frozen.parameters())


def test_vla_jepa_loss_scaling_helpers_keep_raw_metrics_out_of_training_loss():
    from starVLA.model.framework.VLA_JEPA import VLA_JEPA

    model = VLA_JEPA.__new__(VLA_JEPA)
    torch.nn.Module.__init__(model)
    model.config = OmegaConf.create(
        {
            "framework": {
                "future_interface": {
                    "losses": {
                        "alpha_fm": 2.0,
                        "beta_wm": 0.5,
                    }
                }
            }
        }
    )

    losses, metrics = model._build_loss_dict(torch.tensor(3.0), torch.tensor(4.0))

    assert set(losses) == {"action_loss", "wm_loss"}
    assert losses["action_loss"].item() == 6.0
    assert losses["wm_loss"].item() == 2.0
    assert metrics["action_loss_raw"].item() == 3.0
    assert metrics["wm_loss_raw"].item() == 4.0


def test_vla_jepa_freezes_world_teacher_encoder():
    from starVLA.model.framework.VLA_JEPA import VLA_JEPA

    model = VLA_JEPA.__new__(VLA_JEPA)
    torch.nn.Module.__init__(model)
    model.vj_encoder = torch.nn.Linear(2, 2)
    model.vj_encoder.train()

    model._freeze_world_teacher()

    assert not model.vj_encoder.training
    assert all(not param.requires_grad for param in model.vj_encoder.parameters())


def test_future_token_predictor_outputs_structured_token_shapes():
    from starVLA.model.modules.future_interface import FutureTokenPredictor

    predictor = FutureTokenPredictor(vlm_dim=8, token_dim=4, n_prog=1, n_int=2, n_obj=2)
    outputs = predictor(torch.randn(3, 5, 8))

    assert outputs["z_prog"].shape == (3, 1, 4)
    assert outputs["z_int"].shape == (3, 2, 4)
    assert outputs["z_obj"].shape == (3, 2, 4)


def test_world_summary_extractor_global_and_masked_pooling():
    from starVLA.model.modules.future_interface import WorldSummaryExtractor

    extractor = WorldSummaryExtractor()
    future_latents = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    masks = {
        "interaction": torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.float32),
        "object": torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float32),
    }

    outputs = extractor(future_latents, masks=masks)

    assert torch.allclose(outputs["z_prog_star"], future_latents.mean(dim=1, keepdim=True))
    assert torch.allclose(outputs["z_int_star"][0, 0], future_latents[0, :2].mean(dim=0))
    assert torch.allclose(outputs["z_obj_star"][0, 0], future_latents[0, 2])


def test_training_configs_define_disabled_future_interface_defaults():
    for relative_path in ("scripts/config/vlajepa_cotrain.yaml", "scripts/config/vlajepa_robot_ft.yaml"):
        cfg = OmegaConf.load(Path(relative_path))

        assert cfg.framework.future_interface.enabled is False
        assert cfg.framework.future_interface.n_prog == 1
        assert cfg.framework.future_interface.n_int == 1
        assert cfg.framework.future_interface.n_obj == 1
        assert cfg.framework.future_interface.losses.alpha_fm == 1.0
        assert cfg.framework.future_interface.losses.beta_wm == 0.1
        assert cfg.framework.future_interface.losses.lambda_bridge == 0.0
