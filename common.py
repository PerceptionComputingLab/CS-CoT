from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n" for row in rows), encoding="utf-8")


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("IDs must be pseudonymous single path components using letters, digits, _, -, .")
    return value


def resolve(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root)/path


def cuda_device(value="cuda:0"):
    import torch
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError("Model inference requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a CUDA-enabled PyTorch build and an accessible NVIDIA GPU")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable")
    return torch.device("cuda", index)


def inference_dtype(device, precision="auto"):
    import torch
    types = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    with torch.cuda.device(device):
        supports_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if precision == "auto":
        return torch.bfloat16 if supports_bf16 else torch.float16
    if precision not in types:
        raise ValueError("precision must be auto/float16/bfloat16/float32")
    if precision == "bfloat16" and not supports_bf16:
        raise ValueError("The selected CUDA device does not support bfloat16")
    return types[precision]


def to_numpy(value):
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.cpu().numpy()
    return np.asarray(value)


def to_python(value):
    import torch
    if isinstance(value, dict):
        return {key: to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return to_numpy(value).tolist()
    return value.item() if isinstance(value, np.generic) else value


def create_external_adapter(config):

    spec = config.get("factory")
    if not spec or ":" not in spec:
        raise ValueError("python backend requires factory='your_adapter_module:create_model'")
    module, name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    checkpoint = config.get("checkpoint")
    checkpoint = str(Path(checkpoint).expanduser()) if checkpoint else None
    return factory(checkpoint=checkpoint, device=str(cuda_device(config.get("device", "cuda:0"))),
                   **config.get("adapter_options", {}))


@dataclass
class Grid:
    hu: np.ndarray
    affine: np.ndarray

    def __post_init__(self):
        self.hu = np.asarray(self.hu)
        self.affine = np.asarray(self.affine, dtype=float)
        if self.hu.ndim != 3 or self.affine.shape != (4, 4) or min(self.hu.shape) < 2:
            raise ValueError("Need a 3-D ZYX array (all dimensions >=2) and 4x4 affine")
        if not np.isfinite(self.hu).all() or not np.isfinite(self.affine).all():
            raise ValueError("Nonfinite volume or affine")
        if not np.allclose(self.affine[3], [0, 0, 0, 1]):
            raise ValueError("Invalid affine homogeneous row")
        self.spacing_xyz = np.linalg.norm(self.affine[:3, :3], axis=0)
        if np.any(self.spacing_xyz <= 0):
            raise ValueError("Nonpositive voxel spacing")
        unit = self.affine[:3, :3]/self.spacing_xyz
        if not np.allclose(unit.T@unit, np.eye(3), atol=1e-5):
            raise ValueError("Sheared/tilted grid: convert with a validated DICOM converter before using this loader")

    def world_to_zyx(self, points):
        points = np.asarray(points, dtype=float)
        inverse = np.linalg.inv(self.affine)
        return (points@inverse[:3, :3].T+inverse[:3, 3])[..., ::-1]

    def zyx_to_world(self, points):
        xyz = np.asarray(points, dtype=float)[..., ::-1]
        return xyz@self.affine[:3, :3].T+self.affine[:3, 3]


def load_grid(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return Grid(data["hu"].astype(np.float32), data["affine"])
    import nibabel as nib
    image = nib.load(str(path))
    return Grid(image.get_fdata(dtype=np.float32).transpose(2, 1, 0), image.affine)


def save_grid(path, grid):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, hu=grid.hu.astype(np.float32), affine=grid.affine)


def normalize_hu(hu, window=(-200, 800)):
    low, high = window
    if high <= low:
        raise ValueError("HU window must have low < high")
    return (np.clip(np.asarray(hu, dtype=np.float32), low, high)-low)/(high-low)


def resample_array(array, source_affine, target_grid, order=1, cval=0):

    reverse = np.eye(4)[[2, 1, 0, 3]]
    transform = reverse@np.linalg.inv(source_affine)@target_grid.affine@reverse
    return ndimage.affine_transform(np.asarray(array), transform[:3, :3], transform[:3, 3],
                                    output_shape=target_grid.hu.shape, order=order, mode="constant",
                                    cval=cval, prefilter=False)


def load_mask_npz(path, grid):
    with np.load(path, allow_pickle=False) as data:
        if "affine" not in data or not np.allclose(data["affine"], grid.affine, atol=1e-5):
            raise ValueError("Masks need the same affine as the preprocessed CCTA")
        masks = {name: data[name].astype(bool) for name in data.files if name != "affine"}
    if not masks or any(mask.shape != grid.hu.shape for mask in masks.values()):
        raise ValueError("Mask shape mismatch or no masks")
    return masks


def round_values(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [round_values(v) for v in value]
    if isinstance(value, dict):
        return {key: round_values(v) for key, v in value.items()}
    return value
