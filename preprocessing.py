from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from .common import (Grid, load_grid, read_json, read_jsonl, resample_array, resolve,
                     safe_id, save_grid, write_json, write_jsonl)


RISK_FIELDS = ("age_years", "sex", "smoking_status", "hypertension", "diabetes_mellitus",
               "hyperlipidemia", "family_history", "body_mass_index", "symptom_profile",
               "relevant_clinical_indicators")


def clinical_profile(raw):

    out = {key: raw.get(key) for key in RISK_FIELDS}
    for key in ("hypertension", "diabetes_mellitus", "hyperlipidemia", "family_history"):
        if out[key] is not None and type(out[key]) is not bool:
            raise ValueError(f"{key} must be JSON true/false/null")
    for key in ("age_years", "body_mass_index"):
        value = out[key]
        if value is not None and (type(value) not in (int, float) or not np.isfinite(value) or value < 0):
            raise ValueError(f"Invalid {key}")
    if out["age_years"] is not None and out["age_years"] > 120:
        raise ValueError("Age out of range")
    if out["sex"] not in (None, "male", "female", "other", "unknown"):
        raise ValueError("sex must be male/female/other/unknown/null")
    if out["smoking_status"] not in (None, "never", "former", "current", "unknown"):
        raise ValueError("Unknown smoking_status")
    for key in ("symptom_profile", "relevant_clinical_indicators"):
        if out[key] is not None and not isinstance(out[key], dict):
            raise ValueError(f"{key} must be an object or null")
    indicators = out["relevant_clinical_indicators"]
    if indicators and isinstance(indicators.get("troponin"), dict):
        item = indicators["troponin"]
        value, unit = item.get("value"), item.get("unit")
        if value is not None:
            factors = {"ng/mL": 1.0, "ng/L": .001}
            if unit not in factors or not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError("Troponin needs a nonnegative value and unit ng/mL or ng/L")
            indicators = dict(indicators)
            indicators["troponin_level_ng_per_ml"] = float(value)*factors[unit]
            out["relevant_clinical_indicators"] = indicators
    return out


def load_dicom_series(folder, series_uid=None):

    import pydicom
    from pydicom.errors import InvalidDicomError
    groups = {}
    for path in Path(folder).rglob("*"):
        if not path.is_file():
            continue
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True)
        except InvalidDicomError:
            continue
        if getattr(header, "Modality", None) != "CT":
            continue
        uid = str(header.SeriesInstanceUID)
        groups.setdefault(uid, []).append((path, header))
    if series_uid is None:
        if len(groups) != 1:
            raise ValueError("DICOM input must contain exactly one CT series or specify series_uid")
        selected = next(iter(groups.values()))
    else:
        if series_uid not in groups:
            raise ValueError("Requested CT series_uid not found")
        selected = groups[series_uid]
    if len(selected) < 2:
        raise ValueError("CT volume needs at least two slices")
    first = selected[0][1]
    orientation = np.array(first.ImageOrientationPatient, dtype=float)
    column_direction, row_direction = orientation[:3], orientation[3:]
    normal = np.cross(column_direction, row_direction)
    pixel_spacing = np.array(first.PixelSpacing, dtype=float)
    if not np.isclose(np.linalg.norm(normal), 1, atol=1e-4) or np.any(pixel_spacing <= 0):
        raise ValueError("Invalid CT orientation or spacing")
    for _, ds in selected:
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError("Enhanced/multiframe CT needs validated conversion to HU NIfTI first")
        if not np.allclose(ds.ImageOrientationPatient, orientation, atol=1e-4) or not np.allclose(ds.PixelSpacing, pixel_spacing):
            raise ValueError("Mixed slice orientations or pixel spacings")
        if (ds.Rows, ds.Columns) != (first.Rows, first.Columns):
            raise ValueError("Mixed slice dimensions")
    selected.sort(key=lambda pair: np.array(pair[1].ImagePositionPatient, dtype=float)@normal)
    positions = np.array([ds.ImagePositionPatient for _, ds in selected], dtype=float)
    steps = np.diff(positions, axis=0)
    dz = float(np.median(steps@normal))
    if dz <= 0 or not np.allclose(steps, normal*dz, atol=max(.02, dz*.01)):
        raise ValueError("Duplicate, missing, irregularly spaced or gantry-tilted CT slices")
    slices = []
    for path, _ in selected:
        ds = pydicom.dcmread(path)
        if not hasattr(ds, "RescaleSlope") or not hasattr(ds, "RescaleIntercept"):
            raise ValueError("CT slice missing HU calibration RescaleSlope/RescaleIntercept")
        if hasattr(ds, "ModalityLUTSequence"):
            raise ValueError("Modality LUT CT needs validated HU conversion before this loader")
        slices.append(ds.pixel_array.astype(np.float32)*float(ds.RescaleSlope)+float(ds.RescaleIntercept))
    lps_affine = np.eye(4)
    lps_affine[:3, 0] = column_direction*pixel_spacing[1]
    lps_affine[:3, 1] = row_direction*pixel_spacing[0]
    lps_affine[:3, 2] = normal*dz
    lps_affine[:3, 3] = positions[0]
    ras_affine = np.diag([-1, -1, 1, 1])@lps_affine
    return Grid(np.stack(slices), ras_affine)


def isotropic_grid(source, spacing_mm=.6):
    if spacing_mm <= 0:
        raise ValueError("Target spacing must be positive")
    corners = np.array(list(itertools.product(*[(0, n-1) for n in source.hu.shape])))
    world = source.zyx_to_world(corners)
    origin, end = world.min(axis=0), world.max(axis=0)
    shape_xyz = np.ceil((end-origin)/spacing_mm-1e-8).astype(int)+1
    affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
    affine[:3, 3] = origin
    target = Grid(np.zeros(tuple(shape_xyz[::-1]), dtype=np.float32), affine)
    if target.hu.shape == source.hu.shape and np.allclose(source.affine, affine, atol=1e-6):
        return Grid(source.hu.copy(), affine)
    hu = resample_array(source.hu, source.affine, target, order=1, cval=-1024)
    return Grid(hu.astype(np.float32), affine)


def aligned_mask_export(source_path, output_path, source_grid, target_grid):
    source_path = Path(source_path)
    if source_path.suffix == ".npz":
        with np.load(source_path, allow_pickle=False) as data:
            affine = data["affine"] if "affine" in data else source_grid.affine
            arrays = {name: data[name] for name in data.files if name != "affine"}
            if "affine" not in data and any(mask.shape != source_grid.hu.shape for mask in arrays.values()):
                raise ValueError("Mask without affine must have the exact original CCTA shape")
    else:
        import nibabel as nib
        image = nib.load(str(source_path))
        affine = image.affine
        arrays = {"mask": np.asanyarray(image.dataobj).transpose(2, 1, 0)}
    if not arrays:
        raise ValueError("Empty mask export")
    converted = {}
    for name, mask in arrays.items():
        safe_id(name)
        if mask.ndim != 3 or not np.isfinite(mask).all():
            raise ValueError("Masks must be finite 3D arrays")
        if not np.all(np.isin(np.unique(mask), [0, 1])):
            raise ValueError("Use a binary segmentation or separate named binary branch arrays")
        converted[name] = resample_array(mask.astype(np.uint8), affine, target_grid, order=0).astype(bool)
    np.savez_compressed(output_path, **converted, affine=target_grid.affine)


def preprocess_case(row, source_root, output_root, spacing_mm=.6):
    case_id, patient_id = safe_id(row["case_id"]), safe_id(row["patient_id"])
    folder = Path(output_root)/case_id
    if (folder/"case.json").exists():
        raise FileExistsError(f"Already processed: {case_id}; choose a new output directory")
    if bool(row.get("ccta_path")) == bool(row.get("dicom_dir")):
        raise ValueError("Supply exactly one ccta_path (HU NPZ/NIfTI) or dicom_dir")
    source = load_grid(resolve(source_root, row["ccta_path"])) if row.get("ccta_path") else load_dicom_series(
        resolve(source_root, row["dicom_dir"]), row.get("series_uid"))
    grid = isotropic_grid(source, spacing_mm)
    profile = clinical_profile(read_json(resolve(source_root, row["clinical_path"])))
    folder.mkdir(parents=True, exist_ok=True)
    save_grid(folder/"ccta.npz", grid)
    write_json(folder/"clinical.json", profile)
    exports = {}
    source_exports = row.get("exports", {})
    for name in ("segmentation", "branch_masks"):
        if source_exports.get(name):
            aligned_mask_export(resolve(source_root, source_exports[name]), folder/(name+".npz"), source, grid)
            exports[name] = name+".npz"
    for name in ("branches", "lesions"):
        if source_exports.get(name):
            content = read_json(resolve(source_root, source_exports[name]))
            if name == "branches":
                coordinate_system = content.get("coordinate_system", "RAS")
                if coordinate_system not in ("RAS", "LPS"):
                    raise ValueError("Branches must use world RAS/LPS millimetres, not voxel indices")
                for branch in content["branches"]:
                    safe_id(branch["branch_name"])
                    points = np.asarray(branch["centerline_world_mm"], dtype=float)
                    if coordinate_system == "LPS":
                        points = points*np.array([-1, -1, 1])
                    branch["centerline_world_mm"] = points.tolist()
                content["coordinate_system"] = "RAS"
            write_json(folder/(name+".json"), content)
            exports[name] = name+".json"
    annotations = {}
    for key in ("report", "targets"):
        path = row.get("annotations", {}).get(key)
        if path:
            if key == "report":
                (folder/"target_report.txt").write_text(resolve(source_root, path).read_text(encoding="utf-8"), encoding="utf-8")
                annotations[key] = "target_report.txt"
            else:
                write_json(folder/"targets.json", read_json(resolve(source_root, path)))
                annotations[key] = "targets.json"
    mask_path = row.get("annotations", {}).get("segmentation")
    if mask_path:
        aligned_mask_export(resolve(source_root, mask_path), folder/"target_segmentation.npz", source, grid)
        annotations["segmentation"] = "target_segmentation.npz"
    case = {"case_id": case_id, "patient_id": patient_id, "split": row.get("split", "inference"),
            "volume_path": "ccta.npz",
            "clinical_path": "clinical.json", "exports": exports, "annotations": annotations,
            "user_prompt": row.get("user_prompt", "Summarize diagnostic findings, risk-related evidence, and additional information needed."),
            "acquisition": row.get("acquisition", {}),
            "preprocessing": {"shape_zyx": list(grid.hu.shape), "spacing_xyz_mm": grid.spacing_xyz.tolist(),
                              "coordinate_system": "RAS", "image_unit": "HU", "normalized_HU_saved": False,
                              "missing_risk_fields": [k for k, v in profile.items() if v is None]}}
    write_json(folder/"case.json", case)
    return {"case_id": case_id, "patient_id": patient_id, "split": case["split"], "case_path": f"{case_id}/case.json"}


def assign_patient_splits(rows, ratios=(.7, .15, .15), seed=56):
    if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError("Need three positive split ratios summing to one")
    patients = np.array(sorted({row["patient_id"] for row in rows}))
    if len(patients) < 3:
        raise ValueError("At least three patients for patient-wise train/val/test splitting")
    np.random.default_rng(seed).shuffle(patients)
    exact = len(patients)*np.array(ratios)
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact-counts), kind="stable")[:len(patients)-counts.sum()]:
        counts[i] += 1
    for i in range(3):
        if counts[i] == 0:
            counts[np.argmax(counts)] -= 1
            counts[i] += 1
    assignment, start = {}, 0
    for split, count in zip(("train", "val", "test"), counts):
        assignment.update({str(p): split for p in patients[start:start+count]})
        start += count
    return [{**row, "split": assignment[row["patient_id"]]} for row in rows]


def preprocess_batch(manifest, output, spacing_mm=.6, split_ratios=None, seed=56):
    manifest, output = Path(manifest), Path(output)
    if (output/"index.jsonl").exists():
        raise FileExistsError("Preprocessing output index already exists; use a new output directory")
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError("Empty raw cohort manifest")
    if split_ratios:
        if any(row.get("split") not in (None, "inference") for row in rows):
            raise ValueError("Do not override explicitly assigned splits")
        rows = assign_patient_splits(rows, split_ratios, seed)
    identities, patient_splits = set(), {}
    for row in rows:
        safe_id(row["case_id"])
        safe_id(row["patient_id"])
        if row["case_id"] in identities:
            raise ValueError("Duplicate case_id")
        identities.add(row["case_id"])
        split = row.get("split", "inference")
        if split not in ("train", "val", "test", "inference", "reference"):
            raise ValueError("Invalid split")
        if row["patient_id"] in patient_splits and patient_splits[row["patient_id"]] != split:
            raise ValueError("The same patient's visits cannot cross splits")
        patient_splits[row["patient_id"]] = split
    processed, failed = [], []
    for row in rows:
        try:
            processed.append(preprocess_case(row, manifest.parent, output, spacing_mm))
            print(f"preprocessed {row['case_id']}", flush=True)
        except Exception as exc:
            failed.append({"case_id": row["case_id"], "error_type": type(exc).__name__, "error": str(exc)})
    write_jsonl(output/"index.jsonl", processed)
    write_jsonl(output/"errors.jsonl", failed)
    write_json(output/"preprocessing_summary.json", {"succeeded": len(processed), "failed": len(failed),
               "target_spacing_mm": spacing_mm, "split_seed": seed if split_ratios else None})
    return processed, failed
