from __future__ import annotations

import numpy as np
from scipy import ndimage

from .common import Grid


def resample_centerline(points, step_mm=0.6):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Centerline requires at least two finite world [x,y,z] points")
    if step_mm <= 0:
        raise ValueError("Sampling step must be positive")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Zero-length centerline")
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    target = np.linspace(0, distance[-1], max(2, int(np.ceil(distance[-1] / step_mm)) + 1))
    return np.column_stack([np.interp(target, distance, points[:, i]) for i in range(3)]), target


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def parallel_transport_frames(points):
    points = np.asarray(points, dtype=float)
    tangent = _unit(np.gradient(points, axis=0))
    normal = np.empty_like(tangent)
    seed_axis = np.eye(3)[np.argmin(np.abs(tangent[0]))]
    normal[0] = _unit(np.cross(tangent[0], seed_axis))
    for i in range(1, len(points)):
        axis = np.cross(tangent[i - 1], tangent[i])
        sine = np.linalg.norm(axis)
        cosine = np.clip(tangent[i - 1] @ tangent[i], -1, 1)
        if sine < 1e-10:
            candidate = normal[i - 1]
        else:
            axis /= sine
            old = normal[i - 1]
            candidate = old * cosine + np.cross(axis, old) * sine + axis * (axis @ old) * (1 - cosine)
        normal[i] = _unit(candidate - (candidate @ tangent[i]) * tangent[i])
    return tangent, normal, _unit(np.cross(tangent, normal))


def reconstruct_cpr(grid: Grid, points, step_mm=0.6, pixel_mm=0.4, size=25):
    if size < 3 or size % 2 == 0 or pixel_mm <= 0:
        raise ValueError("CPR size must be odd and >=3; pixel size must be positive")
    line, distances = resample_centerline(points, step_mm)
    _, normal, binormal = parallel_transport_frames(line)
    offsets = (np.arange(size) - size // 2) * pixel_mm
    u, v = np.meshgrid(offsets, offsets, indexing="ij")
    world = line[:, None, None, :] + u[None, :, :, None] * normal[:, None, None, :] + v[None, :, :, None] * binormal[:, None, None, :]
    coords = grid.world_to_zyx(world).reshape(-1, 3).T
    cpr = ndimage.map_coordinates(grid.hu.astype(np.float32), coords, order=1, mode="constant", cval=-1024)
    return cpr.reshape(len(line), size, size), line, distances


def diameter_profile(grid, branch_mask, line):
    distance = ndimage.distance_transform_edt(branch_mask, sampling=grid.spacing_xyz[::-1])
    coords = grid.world_to_zyx(line)
    if np.any(coords < 0) or np.any(coords > np.array(grid.hu.shape) - 1):
        raise ValueError("Centerline outside volume")
    return 2 * ndimage.map_coordinates(distance, coords.T, order=1, mode="constant", cval=0)


def anatomical_features(grid, mask, branch, branches):
    line, distances = resample_centerline(branch["centerline_world_mm"])
    diameters = diameter_profile(grid, mask, line)
    tangent, _, _ = parallel_transport_frames(line)
    curvature = np.linalg.norm(np.gradient(tangent, distances, axis=0), axis=1)
    return {
        "branch_name": branch["branch_name"],
        "branch_length_mm": float(distances[-1]),
        "mean_lumen_diameter_mm": float(diameters.mean()),
        "max_lumen_diameter_mm": float(diameters.max()),
        "start_coordinate": line[0].tolist(),
        "end_coordinate": line[-1].tolist(),
        "curvature_per_mm": float(curvature.mean()),
        "parent_branch_name": branch.get("parent_branch_name"),
        "child_branch_count": sum(b.get("parent_branch_name") == branch["branch_name"] for b in branches),
        "centerline_outside_mask_fraction": float(np.mean(diameters == 0)),
    }, line, distances, diameters


def perivascular_features(grid, branch_mask, whole_lumen, mean_diameter_mm, branch_name,
                         fat_min=-190, fat_max=-30, wall_distance_factor=1.0, connectivity=26):
    if connectivity not in (6, 18, 26) or wall_distance_factor <= 0:
        raise ValueError("Invalid perivascular ROI/connectivity settings")

    wall_distance = ndimage.distance_transform_edt(~branch_mask, sampling=grid.spacing_xyz[::-1])
    shell = (wall_distance > 0) & (wall_distance <= wall_distance_factor * mean_diameter_mm) & ~whole_lumen
    fat = shell & (grid.hu >= fat_min) & (grid.hu <= fat_max)
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    _, components = ndimage.label(fat, structure=structure)
    count = int(fat.sum())
    return {
        "branch_name": branch_name,
        "fat_attenuation_index_HU": float(grid.hu[fat].mean()) if count else None,
        "connected_component_count": int(components),
        "fat_voxel_count": count,
        "fat_volume_mm3": float(count * np.prod(grid.spacing_xyz)),
        "roi_wall_distance_mm": float(wall_distance_factor * mean_diameter_mm),
        "connectivity": connectivity,
    }
