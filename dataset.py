from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import cuda_device, load_grid, load_mask_npz, normalize_hu, read_json, read_jsonl, resolve


def _rows(index, split=None):
    index = Path(index).resolve()
    all_rows = read_jsonl(index)
    seen, patients = set(), {}
    for row in all_rows:
        if row["case_id"] in seen:
            raise ValueError("Duplicate case_id in dataset index")
        seen.add(row["case_id"])
        if row["patient_id"] in patients and patients[row["patient_id"]] != row["split"]:
            raise ValueError("Patient leakage across dataset splits")
        patients[row["patient_id"]] = row["split"]
    return index.parent, [r for r in all_rows if split is None or r["split"] == split]


def _targets(case, folder):
    annotation = case.get("annotations", {})
    return {"targets": read_json(resolve(folder, annotation["targets"])) if annotation.get("targets") else None,
            "target_report": resolve(folder, annotation["report"]).read_text(encoding="utf-8") if annotation.get("report") else None}


class ClinicalVolumeDataset(Dataset):

    def __init__(self, index, split=None, hu_window=(-200, 800), include_targets=True):
        self.root, self.records = _rows(index, split)
        self.hu_window, self.include_targets = tuple(hu_window), include_targets

    def __len__(self):
        return len(self.records)

    def __getitem__(self, position):
        row = self.records[position]
        case_path = resolve(self.root, row["case_path"])
        case, folder = read_json(case_path), case_path.parent
        if any(case[k] != row[k] for k in ("case_id", "patient_id", "split")):
            raise ValueError("Dataset index and case metadata disagree")
        grid = load_grid(resolve(folder, case["volume_path"]))
        image = torch.from_numpy(np.ascontiguousarray(normalize_hu(grid.hu, self.hu_window)))[None]
        sample = {"case_id": case["case_id"], "patient_id": case["patient_id"], "split": case["split"],
                  "image": image, "affine": torch.tensor(grid.affine, dtype=torch.float64),
                  "spacing_xyz_mm": torch.tensor(grid.spacing_xyz, dtype=torch.float32),
                  "patient": read_json(resolve(folder, case["clinical_path"])),
                  "case_path": str(case_path.resolve()),
                  "segmentation_target": None, "targets": None, "target_report": None}
        if self.include_targets:
            sample.update(_targets(case, folder))
            if case.get("annotations", {}).get("segmentation"):
                masks = load_mask_npz(resolve(folder, case["annotations"]["segmentation"]), grid)
                target = np.logical_or.reduce(list(masks.values())).astype(np.int64)
                sample["segmentation_target"] = torch.from_numpy(target)[None]
        return sample


def collate_volumes(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    shapes = [tuple(sample["image"].shape[1:]) for sample in samples]
    maximum = tuple(max(s[axis] for s in shapes) for axis in range(3))
    batch_size = len(samples)
    images = torch.zeros((batch_size, 1, *maximum), dtype=torch.float32)
    valid = torch.zeros((batch_size, 1, *maximum), dtype=torch.bool)
    has_label = any(s["segmentation_target"] is not None for s in samples)
    labels = torch.full((batch_size, 1, *maximum), -1, dtype=torch.long) if has_label else None
    for i, (sample, shape) in enumerate(zip(samples, shapes)):
        slices = (i, slice(None), *(slice(0, n) for n in shape))
        images[slices], valid[slices] = sample["image"], True
        if labels is not None and sample["segmentation_target"] is not None:
            labels[slices] = sample["segmentation_target"]
    return {"image": images, "valid_mask": valid, "original_shapes_zyx": shapes,
            "affine": torch.stack([s["affine"] for s in samples]),
            "spacing_xyz_mm": torch.stack([s["spacing_xyz_mm"] for s in samples]),
            "segmentation_target": labels, "segmentation_target_valid_mask": labels != -1 if labels is not None else None,
            **{key: [s[key] for s in samples] for key in ("case_id", "patient_id", "split", "patient", "targets", "target_report", "case_path")}}


class EvidenceDataset(Dataset):

    def __init__(self, index, split=None, include_targets=True):
        self.root, self.records = _rows(index, split)
        self.include_targets = include_targets

    def __len__(self):
        return len(self.records)

    def __getitem__(self, position):
        row = self.records[position]
        document = read_json(resolve(self.root, row["evidence_path"]))
        if any(document[k] != row[k] for k in ("case_id", "patient_id", "split")):
            raise ValueError("Evidence index identity mismatch")
        sample = {"case_id": row["case_id"], "patient_id": row["patient_id"], "split": row["split"],
                  "evidence": document["evidence"],
                  "report": resolve(self.root, row["report_path"]).read_text(encoding="utf-8"),
                  "target_report": None, "targets": None}
        if self.include_targets:
            case_path = resolve(self.root, row["source_case_path"])
            sample.update(_targets(read_json(case_path), case_path.parent))
        return sample


def collate_evidence(samples):

    return {key: [sample[key] for sample in samples] for key in samples[0]}


def move_batch_to_device(batch, device="cuda:0"):
    device = cuda_device(device)

    def transfer(value):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {key: transfer(item) for key, item in value.items()}
        if isinstance(value, list):
            return [transfer(item) for item in value]
        if isinstance(value, tuple):
            return tuple(transfer(item) for item in value)
        return value

    return transfer(batch)


def make_volume_dataloader(index, batch_size=1, split=None, shuffle=False, num_workers=4,
                           pin_memory=True, include_targets=True, prefetch_factor=1):
    workers = {"prefetch_factor": prefetch_factor, "multiprocessing_context": "spawn"} if num_workers > 0 else {}
    return DataLoader(ClinicalVolumeDataset(index, split, include_targets=include_targets), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers, collate_fn=collate_volumes,
                      pin_memory=pin_memory, persistent_workers=num_workers > 0, **workers)


def make_evidence_dataloader(index, batch_size=1, split=None, shuffle=False, num_workers=4,
                             include_targets=True, pin_memory=True, prefetch_factor=1):
    workers = {"prefetch_factor": prefetch_factor, "multiprocessing_context": "spawn"} if num_workers > 0 else {}
    return DataLoader(EvidenceDataset(index, split, include_targets), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers, collate_fn=collate_evidence,
                      pin_memory=pin_memory, persistent_workers=num_workers > 0, **workers)
