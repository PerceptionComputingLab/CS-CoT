import os
from pathlib import Path

from .common import load_grid, read_json, resolve, safe_id, write_json, write_jsonl
from .dataset import _rows
from .evidence import build_evidence
from .geometry import reconstruct_cpr
from .models.bioclinicalbert_pipeline import BioClinicalBertPipeline
from .models.branch_labeling_pipeline import BranchLabelingPipeline
from .models.coronary_segmentation_pipeline import CoronarySegmentationPipeline
from .models.qwen_pipeline import QwenPipeline
from .models.stenosis_analysis_pipeline import StenosisAnalysisPipeline
from .prompts import STAGE_NAMES, final_messages, stage_messages
from .retrieval import HistoricalCaseRetriever


class CSCotPipeline:
    def __init__(self, config):


        models = {name: {"device": config.get("device", "cuda:0"),
                         "precision": config.get("precision", "auto"), **settings}
                  for name, settings in config["models"].items()}
        self.config = {**config, "models": models}
        self.segmentation = CoronarySegmentationPipeline(models["coronary_segmentation"])
        self.branch_labeling = BranchLabelingPipeline(models["branch_labeling"])
        self.stenosis = StenosisAnalysisPipeline(models["stenosis_analysis"])
        self.qwen = QwenPipeline(models["qwen"])
        self.retriever = None
        retrieval = config.get("retrieval", {})
        if retrieval.get("index_path"):
            encoder = BioClinicalBertPipeline(models["bioclinicalbert"])
            self.retriever = HistoricalCaseRetriever(encoder, retrieval.get("alpha", .2), retrieval.get("top_k", 5))
            self.retriever.load(retrieval["index_path"])

    def predict(self, case_path, user_prompt=None):

        case_path = Path(case_path).resolve()
        case, root = read_json(case_path), case_path.parent
        grid = load_grid(resolve(root, case["volume_path"]))
        patient = read_json(resolve(root, case["clinical_path"]))
        mask = self.segmentation.predict(grid, case, root)
        branches, branch_masks = self.branch_labeling.predict(grid, mask, case, root)
        settings, cprs = self.config.get("cpr", {}), {}
        for branch in branches:
            name = safe_id(branch["branch_name"])
            hu, line, distance = reconstruct_cpr(grid, branch["centerline_world_mm"],
                settings.get("step_mm", .6), settings.get("pixel_mm", .4), settings.get("size", 25))
            cprs[name] = {"hu": hu, "centerline_world_mm": line, "distance_mm": distance,
                          "pixel_mm": settings.get("pixel_mm", .4)}
        lesions = self.stenosis.predict(cprs, branches, grid, case, root)
        evidence = build_evidence(grid, branches, branch_masks, lesions, patient, self.config.get("features"))
        evidence["Rsim"] = self.retriever.retrieve(case, evidence) if self.retriever else []
        stages, previous = [], None
        for stage in range(1, 5):
            messages = stage_messages(stage, evidence, previous)
            previous = self.qwen.generate(messages)
            stages.append({"stage": stage, "name": STAGE_NAMES[stage - 1], "messages": messages, "text": previous})
        focus = user_prompt if user_prompt is not None else case["user_prompt"]
        text = self.qwen.generate(final_messages(previous, focus))
        return {"case_id": case["case_id"], "patient_id": case["patient_id"], "split": case["split"],
                "evidence": evidence, "stages": stages, "text": text}

    def predict_batch(self, index_path, output, user_prompt=None):

        index_path, output = Path(index_path).resolve(), Path(output).resolve()
        if (output / "index.jsonl").exists():
            raise FileExistsError("Output index already exists")
        root, rows = _rows(index_path)
        successful, failed = [], []
        for row in rows:
            try:
                case_path = resolve(root, row["case_path"]).resolve()
                case = read_json(case_path)
                if any(case[key] != row[key] for key in ("case_id", "patient_id", "split")):
                    raise ValueError("Case/index identity mismatch")
                result = self.predict(case_path, user_prompt)
                name = safe_id(row["case_id"])
                folder = output / name
                metadata = {key: result[key] for key in ("case_id", "patient_id", "split")}
                write_json(folder / "evidence.json", {**metadata, "evidence": result["evidence"]})
                write_json(folder / "reasoning.json", {**metadata, "stages": result["stages"], "text": result["text"]})
                (folder / "report.txt").write_text(result["text"], encoding="utf-8")
                source_path = Path(os.path.relpath(case_path, output)).as_posix()
                successful.append({**metadata, "evidence_path": f"{name}/evidence.json",
                    "report_path": f"{name}/report.txt", "source_case_path": source_path})
            except Exception as exc:
                failed.append({"case_id": row["case_id"], "error": str(exc)})
        write_jsonl(output / "index.jsonl", successful)
        write_jsonl(output / "errors.jsonl", failed)
        return successful, failed
