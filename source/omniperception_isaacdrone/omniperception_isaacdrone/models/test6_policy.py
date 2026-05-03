"""skrl policy/value models and configurable lidar-state feature extractors."""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


# -----------------------------------------------------------------------------
# Config parsing helpers
# -----------------------------------------------------------------------------
def _json_load_maybe_path(model_cfg_path: str | None, model_cfg_json: str | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if model_cfg_path:
        payload = json.loads(Path(model_cfg_path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("model_cfg_path must point to a JSON object")
        merged.update(payload)
    if model_cfg_json:
        payload = json.loads(model_cfg_json)
        if not isinstance(payload, dict):
            raise TypeError("model_cfg_json must decode to a JSON object")
        merged.update(payload)
    return merged


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _ensure_list_of_ints(values: Sequence[Any], *, name: str) -> list[int]:
    out = [int(v) for v in values]
    if len(out) == 0:
        raise ValueError(f"{name} must not be empty")
    return out


def _ensure_list_of_pairs(values: Sequence[Any], *, name: str) -> list[list[int]]:
    out: list[list[int]] = []
    for i, value in enumerate(values):
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(f"{name}[{i}] must be a pair")
        out.append([int(value[0]), int(value[1])])
    if len(out) == 0:
        raise ValueError(f"{name} must not be empty")
    return out


def _norm_name(value: Any, *, default: str = "none") -> str:
    if value is None:
        return default
    name = str(value).strip().lower()
    aliases = {
        "": "none",
        "ln": "layernorm",
        "layer_norm": "layernorm",
        "gn": "groupnorm",
        "group_norm": "groupnorm",
        "bn": "batchnorm2d",
        "batch_norm": "batchnorm2d",
        "in": "instancenorm2d",
        "instance_norm": "instancenorm2d",
    }
    return aliases.get(name, name)


def _activation_name(value: Any, *, default: str = "identity") -> str:
    if value is None:
        return default
    name = str(value).strip().lower()
    aliases = {
        "": "identity",
        "swish": "silu",
        "none": "identity",
    }
    return aliases.get(name, name)


def _make_activation(name: str) -> nn.Module:
    name = _activation_name(name)
    if name == "identity":
        return nn.Identity()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _make_norm_1d(name: str, dim: int) -> nn.Module | None:
    name = _norm_name(name)
    if name == "none":
        return None
    if name == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported 1D norm: {name}")


def _auto_group_norm_groups(num_channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


def _make_conv_norm(name: str, channels: int, groups: str | int) -> nn.Module | None:
    name = _norm_name(name)
    if name == "none":
        return None
    if name == "groupnorm":
        num_groups = _auto_group_norm_groups(channels) if str(groups).lower() == "auto" else int(groups)
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    if name == "batchnorm2d":
        return nn.BatchNorm2d(channels)
    if name == "instancenorm2d":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unsupported conv norm: {name}")


def _normalize_hidden_norms(hidden_dims: Sequence[int], hidden_norms: Sequence[Any] | None) -> list[str]:
    dims = list(hidden_dims)
    if hidden_norms is None:
        return ["none"] * len(dims)
    norms = [_norm_name(v) for v in hidden_norms]
    if len(norms) == len(dims):
        return norms
    if len(norms) == 1 and len(dims) > 1:
        return norms * len(dims)
    raise ValueError("hidden_norms length must match hidden_dims length")


# -----------------------------------------------------------------------------
# Model config dataclasses
# -----------------------------------------------------------------------------
@dataclass
class MLPBranchCfg:
    input_norm: str = "none"
    hidden_dims: list[int] = field(default_factory=list)
    hidden_norms: list[str] = field(default_factory=list)
    activation: str = "identity"

    def normalize(self) -> "MLPBranchCfg":
        self.input_norm = _norm_name(self.input_norm)
        self.hidden_dims = [int(v) for v in self.hidden_dims]
        self.hidden_norms = _normalize_hidden_norms(self.hidden_dims, self.hidden_norms)
        self.activation = _activation_name(self.activation)
        return self


@dataclass
class LidarEncoderCfg:
    type: str = "mlp"
    grid_shape: list[int] | None = None
    output_dim: int = 256
    input_clamp: list[float] = field(default_factory=lambda: [0.0, 1.0])
    input_rescale_to_neg_one_to_one: bool = True
    mlp: MLPBranchCfg = field(
        default_factory=lambda: MLPBranchCfg(
            input_norm="layernorm",
            hidden_dims=[256, 256],
            hidden_norms=["none", "layernorm"],
            activation="tanh",
        )
    )
    conv_channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    kernel_sizes: list[list[int]] = field(default_factory=lambda: [[3, 5], [3, 5], [3, 3]])
    strides: list[list[int]] = field(default_factory=lambda: [[1, 2], [1, 2], [2, 2]])
    conv_norm: str = "groupnorm"
    conv_norm_groups: str | int = "auto"
    conv_activation: str = "silu"
    theta_pad_mode: str = "replicate"
    phi_pad_mode: str = "circular"
    head_pre_norm: str = "layernorm"
    head_post_norm: str = "layernorm"

    def normalize(self) -> "LidarEncoderCfg":
        self.type = str(self.type).strip().lower()
        if self.type not in {"mlp", "cnn"}:
            raise ValueError(f"Unsupported lidar encoder type: {self.type}")
        self.output_dim = int(self.output_dim)
        self.input_clamp = [float(v) for v in self.input_clamp]
        if len(self.input_clamp) != 2:
            raise ValueError("lidar_encoder.input_clamp must contain [min, max]")
        self.mlp.normalize()
        self.conv_channels = _ensure_list_of_ints(self.conv_channels, name="lidar_encoder.conv_channels")
        self.kernel_sizes = _ensure_list_of_pairs(self.kernel_sizes, name="lidar_encoder.kernel_sizes")
        self.strides = _ensure_list_of_pairs(self.strides, name="lidar_encoder.strides")
        if not (len(self.conv_channels) == len(self.kernel_sizes) == len(self.strides)):
            raise ValueError("lidar_encoder conv_channels/kernel_sizes/strides lengths must match")
        self.conv_norm = _norm_name(self.conv_norm)
        self.conv_activation = _activation_name(self.conv_activation)
        self.head_pre_norm = _norm_name(self.head_pre_norm)
        self.head_post_norm = _norm_name(self.head_post_norm)
        self.theta_pad_mode = str(self.theta_pad_mode)
        self.phi_pad_mode = str(self.phi_pad_mode)
        if self.grid_shape is not None:
            if len(self.grid_shape) != 2:
                raise ValueError("lidar_encoder.grid_shape must contain [theta_bins, phi_bins]")
            self.grid_shape = [int(self.grid_shape[0]), int(self.grid_shape[1])]
        return self


@dataclass
class PolicyHeadCfg:
    log_std_init: float = -1.25
    log_std_min: float = -5.0
    log_std_max: float = -0.4
    mean_activation: str = "tanh"

    def normalize(self) -> "PolicyHeadCfg":
        self.log_std_init = float(self.log_std_init)
        self.log_std_min = float(self.log_std_min)
        self.log_std_max = float(self.log_std_max)
        self.mean_activation = _activation_name(self.mean_activation)
        return self


@dataclass
class Test6ModelCfg:
    feature_dim: int = 256
    state_encoder: MLPBranchCfg = field(
        default_factory=lambda: MLPBranchCfg(
            input_norm="layernorm",
            hidden_dims=[128, 128],
            hidden_norms=["none", "layernorm"],
            activation="tanh",
        )
    )
    lidar_encoder: LidarEncoderCfg = field(default_factory=LidarEncoderCfg)
    fusion: MLPBranchCfg = field(
        default_factory=lambda: MLPBranchCfg(
            input_norm="none",
            hidden_dims=[256],
            hidden_norms=["layernorm"],
            activation="tanh",
        )
    )
    policy_head: PolicyHeadCfg = field(default_factory=PolicyHeadCfg)

    def normalize(self) -> "Test6ModelCfg":
        self.feature_dim = int(self.feature_dim)
        self.state_encoder.normalize()
        self.lidar_encoder.normalize()
        self.fusion.normalize()
        self.policy_head.normalize()
        if len(self.fusion.hidden_dims) == 0:
            self.fusion.hidden_dims = [self.feature_dim]
            self.fusion.hidden_norms = ["layernorm"]
        self.feature_dim = int(self.fusion.hidden_dims[-1])
        return self


def model_cfg_to_dict(model_cfg: Test6ModelCfg | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(model_cfg, Test6ModelCfg):
        return asdict(model_cfg)
    return dict(model_cfg)


def build_model_cfg(
    *,
    feat_dim: int = 256,
    lidar_grid_shape: tuple[int, int] | None = None,
) -> Test6ModelCfg:
    cfg = Test6ModelCfg(feature_dim=int(feat_dim))
    cfg.fusion.hidden_dims = [int(feat_dim)]
    cfg.lidar_encoder.type = "cnn" if lidar_grid_shape is not None else "mlp"
    cfg.lidar_encoder.grid_shape = list(lidar_grid_shape) if lidar_grid_shape is not None else None
    return cfg.normalize()


def _mlp_branch_from_dict(default_cfg: MLPBranchCfg, data: Mapping[str, Any]) -> MLPBranchCfg:
    cfg = MLPBranchCfg(
        input_norm=data.get("input_norm", default_cfg.input_norm),
        hidden_dims=list(data.get("hidden_dims", default_cfg.hidden_dims)),
        hidden_norms=list(data.get("hidden_norms", default_cfg.hidden_norms)),
        activation=data.get("activation", default_cfg.activation),
    )
    return cfg.normalize()


def _lidar_encoder_from_dict(default_cfg: LidarEncoderCfg, data: Mapping[str, Any]) -> LidarEncoderCfg:
    cfg = LidarEncoderCfg(
        type=data.get("type", default_cfg.type),
        grid_shape=None if data.get("grid_shape", default_cfg.grid_shape) is None else list(data.get("grid_shape", default_cfg.grid_shape)),
        output_dim=data.get("output_dim", default_cfg.output_dim),
        input_clamp=list(data.get("input_clamp", default_cfg.input_clamp)),
        input_rescale_to_neg_one_to_one=bool(data.get("input_rescale_to_neg_one_to_one", default_cfg.input_rescale_to_neg_one_to_one)),
        mlp=_mlp_branch_from_dict(default_cfg.mlp, dict(data.get("mlp", {}))),
        conv_channels=list(data.get("conv_channels", default_cfg.conv_channels)),
        kernel_sizes=[list(v) for v in data.get("kernel_sizes", default_cfg.kernel_sizes)],
        strides=[list(v) for v in data.get("strides", default_cfg.strides)],
        conv_norm=data.get("conv_norm", default_cfg.conv_norm),
        conv_norm_groups=data.get("conv_norm_groups", default_cfg.conv_norm_groups),
        conv_activation=data.get("conv_activation", default_cfg.conv_activation),
        theta_pad_mode=data.get("theta_pad_mode", default_cfg.theta_pad_mode),
        phi_pad_mode=data.get("phi_pad_mode", default_cfg.phi_pad_mode),
        head_pre_norm=data.get("head_pre_norm", default_cfg.head_pre_norm),
        head_post_norm=data.get("head_post_norm", default_cfg.head_post_norm),
    )
    return cfg.normalize()


def _policy_head_from_dict(default_cfg: PolicyHeadCfg, data: Mapping[str, Any]) -> PolicyHeadCfg:
    cfg = PolicyHeadCfg(
        log_std_init=data.get("log_std_init", default_cfg.log_std_init),
        log_std_min=data.get("log_std_min", default_cfg.log_std_min),
        log_std_max=data.get("log_std_max", default_cfg.log_std_max),
        mean_activation=data.get("mean_activation", default_cfg.mean_activation),
    )
    return cfg.normalize()


def resolve_model_cfg(
    *,
    feat_dim: int = 256,
    lidar_grid_shape: tuple[int, int] | None = None,
    model_cfg_path: str | None = None,
    model_cfg_json: str | None = None,
    base_model_cfg: Mapping[str, Any] | Test6ModelCfg | None = None,
) -> Test6ModelCfg:
    cfg = build_model_cfg(feat_dim=feat_dim, lidar_grid_shape=lidar_grid_shape)
    base_data: dict[str, Any] = {}
    if isinstance(base_model_cfg, Test6ModelCfg):
        base_data = model_cfg_to_dict(base_model_cfg)
    elif isinstance(base_model_cfg, Mapping):
        base_data = dict(base_model_cfg)
    overrides = _json_load_maybe_path(model_cfg_path, model_cfg_json)
    merged = _deep_update(model_cfg_to_dict(cfg), base_data)
    merged = _deep_update(merged, overrides)
    resolved = Test6ModelCfg(
        feature_dim=merged.get("feature_dim", cfg.feature_dim),
        state_encoder=_mlp_branch_from_dict(cfg.state_encoder, dict(merged.get("state_encoder", {}))),
        lidar_encoder=_lidar_encoder_from_dict(cfg.lidar_encoder, dict(merged.get("lidar_encoder", {}))),
        fusion=_mlp_branch_from_dict(cfg.fusion, dict(merged.get("fusion", {}))),
        policy_head=_policy_head_from_dict(cfg.policy_head, dict(merged.get("policy_head", {}))),
    ).normalize()
    if lidar_grid_shape is not None and resolved.lidar_encoder.grid_shape is None:
        resolved.lidar_encoder.grid_shape = list(lidar_grid_shape)
    if resolved.lidar_encoder.type == "cnn" and resolved.lidar_encoder.grid_shape is None:
        raise ValueError("CNN lidar encoder requires lidar_encoder.grid_shape")
    return resolved


# -----------------------------------------------------------------------------
# Config snapshot helpers
# -----------------------------------------------------------------------------
def _parse_snapshot_sections(config_path: Path) -> dict[str, str]:
    current: str | None = None
    sections: dict[str, list[str]] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def load_model_cfg_from_snapshot(config_path: Path) -> Test6ModelCfg | None:
    if not config_path.exists():
        return None
    sections = _parse_snapshot_sections(config_path)
    raw = sections.get("model_cfg", "")
    if not raw:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"model_cfg section in {config_path} is not a JSON object")
    return resolve_model_cfg(base_model_cfg=payload)


def find_config_snapshot_for_checkpoint(checkpoint_path: Path) -> Path | None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    for parent in [checkpoint_path.parent, *checkpoint_path.parents]:
        candidate = parent / "config" / "config.txt"
        if candidate.exists():
            return candidate
    return None


# -----------------------------------------------------------------------------
# Module initialization helpers
# -----------------------------------------------------------------------------
def init_hidden(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.InstanceNorm2d)):
        if getattr(m, "weight", None) is not None:
            nn.init.constant_(m.weight, 1.0)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0.0)


def init_policy_head(m: nn.Linear) -> None:
    nn.init.orthogonal_(m.weight, gain=0.01)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.0)


def init_value_head(m: nn.Linear) -> None:
    nn.init.orthogonal_(m.weight, gain=1.0)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.0)


def gaussian_mixin_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"clip_actions": True}
    if "clip_mean_actions" in inspect.signature(GaussianMixin.__init__).parameters:
        kwargs["clip_mean_actions"] = True
    return kwargs


# -----------------------------------------------------------------------------
# Neural network building blocks
# -----------------------------------------------------------------------------
class MLPBranch(nn.Module):
    def __init__(self, input_dim: int, cfg: MLPBranchCfg):
        super().__init__()
        cfg = cfg.normalize()
        self.input_norm = _make_norm_1d(cfg.input_norm, int(input_dim))
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim, hidden_norm in zip(cfg.hidden_dims, cfg.hidden_norms):
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            norm = _make_norm_1d(hidden_norm, int(hidden_dim))
            if norm is not None:
                layers.append(norm)
            layers.append(_make_activation(cfg.activation))
            prev_dim = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.output_dim = int(prev_dim)
        self.apply(init_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm is not None:
            x = self.input_norm(x)
        return self.net(x)


class LidarConvBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        norm_name: str,
        norm_groups: str | int,
        activation: str,
        theta_pad_mode: str,
        phi_pad_mode: str,
    ):
        super().__init__()
        self.pad_h = int(kernel_size[0]) // 2
        self.pad_w = int(kernel_size[1]) // 2
        self.theta_pad_mode = str(theta_pad_mode)
        self.phi_pad_mode = str(phi_pad_mode)
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=tuple(int(v) for v in kernel_size),
            stride=tuple(int(v) for v in stride),
            padding=0,
            bias=False,
        )
        self.norm = _make_conv_norm(norm_name, int(out_channels), norm_groups)
        self.act = _make_activation(activation)
        self.apply(init_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode=self.phi_pad_mode)
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode=self.theta_pad_mode)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x)


class LidarCNNEncoder(nn.Module):
    def __init__(self, cfg: LidarEncoderCfg):
        super().__init__()
        cfg = cfg.normalize()
        if cfg.grid_shape is None:
            raise ValueError("LidarCNNEncoder requires grid_shape")
        self.theta_bins, self.phi_bins = int(cfg.grid_shape[0]), int(cfg.grid_shape[1])
        self.clamp_min, self.clamp_max = float(cfg.input_clamp[0]), float(cfg.input_clamp[1])
        self.input_rescale_to_neg_one_to_one = bool(cfg.input_rescale_to_neg_one_to_one)
        channels = [1, *cfg.conv_channels]
        blocks = []
        for in_channels, out_channels, kernel_size, stride in zip(
            channels[:-1],
            channels[1:],
            cfg.kernel_sizes,
            cfg.strides,
        ):
            blocks.append(
                LidarConvBlock(
                    in_channels=int(in_channels),
                    out_channels=int(out_channels),
                    kernel_size=(int(kernel_size[0]), int(kernel_size[1])),
                    stride=(int(stride[0]), int(stride[1])),
                    norm_name=cfg.conv_norm,
                    norm_groups=cfg.conv_norm_groups,
                    activation=cfg.conv_activation,
                    theta_pad_mode=cfg.theta_pad_mode,
                    phi_pad_mode=cfg.phi_pad_mode,
                )
            )
        self.backbone = nn.Sequential(*blocks)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.theta_bins, self.phi_bins, dtype=torch.float32)
            conv_out = self.backbone(dummy)
        flat_dim = int(np.prod(conv_out.shape[1:]))
        head_layers: list[nn.Module] = [nn.Flatten()]
        pre_norm = _make_norm_1d(cfg.head_pre_norm, flat_dim)
        if pre_norm is not None:
            head_layers.append(pre_norm)
        head_layers.append(nn.Linear(flat_dim, int(cfg.output_dim)))
        head_layers.append(_make_activation(cfg.conv_activation))
        post_norm = _make_norm_1d(cfg.head_post_norm, int(cfg.output_dim))
        if post_norm is not None:
            head_layers.append(post_norm)
        self.head = nn.Sequential(*head_layers)
        self.output_dim = int(cfg.output_dim)
        self.apply(init_hidden)

    def forward(self, lidar_flat: torch.Tensor) -> torch.Tensor:
        lidar_grid = torch.clamp(lidar_flat, self.clamp_min, self.clamp_max).reshape(
            -1, 1, self.theta_bins, self.phi_bins
        )
        if self.input_rescale_to_neg_one_to_one:
            lidar_grid = lidar_grid * 2.0 - 1.0
        return self.head(self.backbone(lidar_grid))


class StructuredFeatureExtractor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        lidar_dim: int,
        feat_dim: int = 256,
        lidar_grid_shape: tuple[int, int] | None = None,
        model_cfg: Test6ModelCfg | Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)
        self.model_cfg = resolve_model_cfg(
            feat_dim=int(feat_dim),
            lidar_grid_shape=lidar_grid_shape,
            base_model_cfg=model_cfg,
        )
        self.state_net = MLPBranch(self.state_dim, self.model_cfg.state_encoder)
        if self.lidar_dim > 0:
            if self.model_cfg.lidar_encoder.type == "cnn":
                if self.model_cfg.lidar_encoder.grid_shape is None:
                    raise ValueError("CNN lidar encoder requires grid_shape")
                expected = int(np.prod(self.model_cfg.lidar_encoder.grid_shape))
                if expected != self.lidar_dim:
                    raise ValueError(
                        f"lidar grid_shape product {expected} does not match lidar_dim={self.lidar_dim}"
                    )
                self.lidar_encoder = LidarCNNEncoder(self.model_cfg.lidar_encoder)
            else:
                self.lidar_encoder = MLPBranch(self.lidar_dim, self.model_cfg.lidar_encoder.mlp)
            fuse_input_dim = int(self.state_net.output_dim + self.lidar_encoder.output_dim)
        else:
            self.lidar_encoder = None
            fuse_input_dim = int(self.state_net.output_dim)
        self.fuse_net = MLPBranch(fuse_input_dim, self.model_cfg.fusion)

    @property
    def output_dim(self) -> int:
        return int(self.fuse_net.output_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state = self.state_net(torch.clamp(obs[:, : self.state_dim], -1.0, 1.0))
        if self.lidar_dim <= 0 or self.lidar_encoder is None:
            return self.fuse_net(state)
        lidar = obs[:, self.state_dim : self.state_dim + self.lidar_dim]
        lidar_feat = self.lidar_encoder(lidar)
        return self.fuse_net(torch.cat([state, lidar_feat], dim=-1))


# -----------------------------------------------------------------------------
# skrl models
# -----------------------------------------------------------------------------
class Policy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        state_dim,
        lidar_dim,
        feat_dim: int = 256,
        lidar_grid_shape: tuple[int, int] | None = None,
        model_cfg: Test6ModelCfg | Mapping[str, Any] | None = None,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **gaussian_mixin_kwargs())
        if self.num_observations != int(state_dim) + int(lidar_dim):
            raise RuntimeError("obs dim mismatch")
        self.fe = StructuredFeatureExtractor(
            state_dim,
            lidar_dim,
            feat_dim=feat_dim,
            lidar_grid_shape=lidar_grid_shape,
            model_cfg=model_cfg,
        )
        self.mean = nn.Linear(self.fe.output_dim, self.num_actions)
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), float(self.fe.model_cfg.policy_head.log_std_init))
        )
        self.mean_activation = _make_activation(self.fe.model_cfg.policy_head.mean_activation)
        init_policy_head(self.mean)

    def compute(self, inputs, role):
        mean = self.mean_activation(self.mean(self.fe(inputs["states"])))
        log_std = torch.clamp(
            self.log_std_parameter,
            min=float(self.fe.model_cfg.policy_head.log_std_min),
            max=float(self.fe.model_cfg.policy_head.log_std_max),
        ).expand_as(mean)
        return mean, log_std, {}

    @torch.no_grad()
    def play_act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean, log_std, _ = self.compute({"states": obs}, role="policy")
        if deterministic:
            action = mean
        else:
            std = torch.exp(log_std)
            action = mean + std * torch.randn_like(std)
        return torch.clamp(action, -1.0, 1.0)


class Value(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        state_dim,
        lidar_dim,
        feat_dim: int = 256,
        lidar_grid_shape: tuple[int, int] | None = None,
        model_cfg: Test6ModelCfg | Mapping[str, Any] | None = None,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        if self.num_observations != int(state_dim) + int(lidar_dim):
            raise RuntimeError("obs dim mismatch")
        self.fe = StructuredFeatureExtractor(
            state_dim,
            lidar_dim,
            feat_dim=feat_dim,
            lidar_grid_shape=lidar_grid_shape,
            model_cfg=model_cfg,
        )
        self.value = nn.Linear(self.fe.output_dim, 1)
        init_value_head(self.value)

    def compute(self, inputs, role):
        return self.value(self.fe(inputs["states"])), {}
