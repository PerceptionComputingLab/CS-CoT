from __future__ import annotations

import numpy as np

from .common import round_values
from .geometry import anatomical_features, perivascular_features


def build_evidence(grid, branches, masks, lesions_by_branch, patient, config=None):
    config = config or {}
    whole_lumen = np.logical_or.reduce(list(masks.values()))
    evidence = {"A": [], "D": [], "F": [], "P": patient}
    for branch in branches:
        name = branch["branch_name"]
        anatomy, line, distance, diameter = anatomical_features(grid, masks[name], branch, branches)
        evidence["A"].append(anatomy)
        lesions = lesions_by_branch[name]
        if lesions is None:
            disease = {"branch_name": name, "lesion_count": None, "lesions": None}
        else:
            output_lesions = []
            for lesion in lesions:
                item = dict(lesion)
                selected = (distance >= item["start_mm"]) & (distance <= item["end_mm"])
                if not selected.any():
                    selected[np.argmin(abs(distance-(item["start_mm"]+item["end_mm"])/2))] = True
                item["minimal_lumen_diameter_mm"] = float(diameter[selected].min())
                relative = (item["start_mm"]+item["end_mm"])/(2*distance[-1])
                item["lesion_position_proximal_distal"] = "proximal" if relative < 1/3 else "mid" if relative < 2/3 else "distal"
                outer, reference = item.pop("outer_vessel_diameter_mm", None), item.pop("reference_outer_vessel_diameter_mm", None)
                item["remodeling_index"] = outer/reference if outer is not None and reference and reference > 0 else None
                output_lesions.append(item)
            disease = {"branch_name": name, "lesion_count": len(lesions),
                       "stenosis_severity_percent": [p.get("stenosis_severity_percent") for p in lesions],
                       "plaque_composition": [p.get("plaque_composition") for p in lesions],
                       "minimal_lumen_diameter_mm": float(diameter.min()), "lesions": output_lesions}
        evidence["D"].append(disease)
        evidence["F"].append(perivascular_features(grid, masks[name], whole_lumen, anatomy["mean_lumen_diameter_mm"], name,
                              config.get("fat_hu_min", -190), config.get("fat_hu_max", -30),
                              config.get("wall_distance_factor", 1), config.get("connectivity", 26)))
    return round_values(evidence)
