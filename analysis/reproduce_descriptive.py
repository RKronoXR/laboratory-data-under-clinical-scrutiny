#!/usr/bin/env python3
"""Recreate the main clinical tables from saved regional predictions using CPU only.

Uses only the Python standard library. Scores are reference-conditioned localisation
summaries, not conventional diagnostic-classification scores for a new patient.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("nnU-Net", "YOLO")
STRATEGIES = ("clinical_only", "laboratory_pretraining", "knowledge_distillation", "pretraining_and_distillation")
METRICS = ("sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "accuracy", "f1", "mcc", "dice_positive", "dice_tp", "auroc", "auprc_ap", "tp", "fn", "fp", "tn", "pixel_dice_micro")
COUNTS = ("reference_positive", "reference_pixels", "predicted_pixels", "intersection_pixels", "pixel_tp", "pixel_fp", "pixel_tn", "pixel_fn", "detected", "correct_overlap")


def divide(a: float, b: float) -> float | None:
    return a / b if b else None


def average(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(values) if values else None


def sample_sd(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.stdev(values) if len(values) > 1 else None


def localisation_curves(rows):
    """Exact tie-aware ROC area and step average precision from saved scores."""
    pairs = [(r["overlap_score"] if r["reference_positive"] else r["score"], r["reference_positive"]) for r in rows]
    positives = sum(y for _, y in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None, None
    pairs.sort(key=lambda x: x[0], reverse=True)
    tp = fp = 0
    previous_tpr = previous_fpr = 0.0
    area = ap = 0.0
    for score, group in itertools.groupby(pairs, key=lambda x: x[0]):
        labels = [y for _, y in group]
        new_tp = sum(labels)
        tp += new_tp
        fp += len(labels) - new_tp
        tpr, fpr = tp / positives, fp / negatives
        area += (fpr - previous_fpr) * (tpr + previous_tpr) / 2
        ap += (new_tp / positives) * (tp / (tp + fp))
        previous_tpr, previous_fpr = tpr, fpr
    return area, ap


def read_predictions(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 88560:
        raise ValueError(f"Expected 88,560 records (120 fits × two thresholds × 369 regions); found {len(rows)}")
    required = {"architecture", "strategy", "seed", "fold", "region", "bitewing", "threshold_mode", "threshold", "score", "overlap_score", "dice_positive", *COUNTS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(rows[0] if rows else {}))}")
    groups = defaultdict(list)
    regions = {}
    seen = set()
    for r in rows:
        for name in COUNTS:
            r[name] = int(r[name])
            if r[name] < 0:
                raise ValueError(f"Negative count in {name}")
        r["seed"], r["fold"] = int(r["seed"]), int(r["fold"])
        for name in ("threshold", "score", "overlap_score", "dice_positive"):
            r[name] = float(r[name]) if r[name] else None
        if r["architecture"] not in ARCHITECTURES or r["strategy"] not in STRATEGIES:
            raise ValueError("Unrecognised architecture or strategy")
        if r["threshold_mode"] not in ("fixed", "youden") or r["seed"] not in (17, 42, 2026) or r["fold"] not in range(5):
            raise ValueError("Invalid threshold, seed or fold key")
        if r["threshold_mode"] == "fixed" and r["threshold"] != 0.5:
            raise ValueError("The fixed operating threshold must be 0.5")
        positive = r["reference_positive"]
        if positive not in (0, 1) or positive != int(r["reference_pixels"] > 0):
            raise ValueError("Reference-presence flag disagrees with reference pixels")
        if r["intersection_pixels"] > min(r["reference_pixels"], r["predicted_pixels"]):
            raise ValueError("Intersection exceeds reference or predicted area")
        if r["pixel_tp"] != r["intersection_pixels"] or r["pixel_fp"] != r["predicted_pixels"] - r["intersection_pixels"] or r["pixel_fn"] != r["reference_pixels"] - r["intersection_pixels"]:
            raise ValueError("Pixel counts are internally inconsistent")
        correct = r["intersection_pixels"] > 0 if positive else r["predicted_pixels"] == 0
        if r["correct_overlap"] != int(correct) or r["detected"] != int(r["intersection_pixels"] > 0):
            raise ValueError("Stored detection does not satisfy the lesion-overlap definition")
        if r["score"] is None or not math.isfinite(r["score"]) or not 0 <= r["score"] <= 1:
            raise ValueError("Invalid maximum prediction score")
        if positive and (r["overlap_score"] is None or not math.isfinite(r["overlap_score"]) or not 0 <= r["overlap_score"] <= 1):
            raise ValueError("Missing or invalid reference-lesion score")
        if positive:
            expected = 2 * r["intersection_pixels"] / (r["reference_pixels"] + r["predicted_pixels"])
            if r["dice_positive"] is None or abs(r["dice_positive"] - expected) > 1e-12:
                raise ValueError("Stored positive-region Dice disagrees with pixel counts")
        key = (r["architecture"], r["strategy"], r["threshold_mode"], r["seed"], r["fold"])
        full_key = key + (r["region"],)
        if full_key in seen:
            raise ValueError("Duplicate model/threshold/region record")
        seen.add(full_key)
        ref = (r["bitewing"], positive, r["reference_pixels"])
        if r["region"] in regions and regions[r["region"]] != ref:
            raise ValueError("Reference identity changed across repeated assessments")
        regions[r["region"]] = ref
        groups[key].append(r)
    if len(regions) != 369 or len({v[0] for v in regions.values()}) != 25 or sum(v[1] for v in regions.values()) != 184:
        raise ValueError("Expected 369 paired regions: 184 positive, 185 negative in 25 bitewings")
    expected_keys = set(itertools.product(ARCHITECTURES, STRATEGIES, ("fixed", "youden"), (17, 42, 2026), range(5)))
    if set(groups) != expected_keys or any(len(rr) != 369 for rr in groups.values()):
        raise ValueError("Incomplete strategy/seed/fold/threshold coverage")
    return rows, groups


def one_model(rows):
    tp = sum(r["reference_positive"] and r["intersection_pixels"] > 0 for r in rows)
    fn = sum(r["reference_positive"] and r["intersection_pixels"] == 0 for r in rows)
    fp = sum(not r["reference_positive"] and r["predicted_pixels"] > 0 for r in rows)
    tn = sum(not r["reference_positive"] and r["predicted_pixels"] == 0 for r in rows)
    if tp + fn != 184 or tn + fp != 185:
        raise ValueError("A fitted model has a different reference-positive/negative population")
    sensitivity, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    positive = [r for r in rows if r["reference_positive"]]
    hits = [r for r in positive if r["intersection_pixels"] > 0]
    pixel_tp, pixel_fp, pixel_fn = (sum(r[f"pixel_{name}"] for r in rows) for name in ("tp", "fp", "fn"))
    auc, ap = localisation_curves(rows)
    thresholds = {r["threshold"] for r in rows}
    if len(thresholds) != 1:
        raise ValueError("A fitted model used more than one threshold for one operating point")
    return dict(tp=tp, fn=fn, fp=fp, tn=tn,
                sensitivity=sensitivity, specificity=specificity, ppv=divide(tp, tp + fp), npv=divide(tn, tn + fn),
                balanced_accuracy=(sensitivity + specificity) / 2, accuracy=(tp + tn) / 369,
                f1=divide(2 * tp, 2 * tp + fp + fn), mcc=(tp * tn - fp * fn) / math.sqrt(denominator) if denominator else None,
                dice_positive=average([r["dice_positive"] for r in positive]), dice_tp=average([r["dice_positive"] for r in hits]),
                pixel_dice_micro=divide(2 * pixel_tp, 2 * pixel_tp + pixel_fp + pixel_fn), auroc=auc, auprc_ap=ap,
                threshold=next(iter(thresholds)))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reproduce(input_path, output_directory, verify_reference=False):
    rows, groups = read_predictions(input_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    model_rows = []
    for key, rr in groups.items():
        model_rows.append(dict(zip(("architecture", "strategy", "threshold_mode", "seed", "fold"), key), **one_model(rr)))
    seed_rows = []
    summaries = []
    for arch, strategy, mode in itertools.product(ARCHITECTURES, STRATEGIES, ("fixed", "youden")):
        models = [r for r in model_rows if (r["architecture"], r["strategy"], r["threshold_mode"]) == (arch, strategy, mode)]
        seeds = []
        for seed in (17, 42, 2026):
            mm = [r for r in models if r["seed"] == seed]
            ss = dict(architecture=arch, strategy=strategy, threshold_mode=mode, seed=seed)
            for name in (*METRICS, "threshold"):
                ss[name] = average([r[name] for r in mm])
            seed_rows.append(ss)
            seeds.append(ss)
        summary = dict(architecture=arch, strategy=strategy, threshold_mode=mode)
        for name in (*METRICS, "threshold"):
            summary[name + "_mean"] = average([r[name] for r in seeds])
            summary[name + "_sd"] = sample_sd([r[name] for r in seeds])
            summary[name + "_valid_models"] = sum(r[name] is not None for r in models)
        summary["threshold_min"] = min(r["threshold"] for r in models)
        summary["threshold_max"] = max(r["threshold"] for r in models)
        summaries.append(summary)
    save_csv(output_directory / "model_metrics.csv", model_rows)
    save_csv(output_directory / "seed_metrics.csv", seed_rows)
    save_csv(output_directory / "descriptive.csv", summaries)
    by = {(r["architecture"], r["strategy"], r["threshold_mode"]): r for r in summaries}

    def table(metrics, mode="fixed"):
        result = []
        for arch, strategy in itertools.product(ARCHITECTURES, STRATEGIES):
            s = by[(arch, strategy, mode)]
            row = dict(architecture=arch, strategy=strategy)
            for name in metrics:
                mean, sd = s[name + "_mean"], s[name + "_sd"]
                row[name] = f"{mean:.3f} ({sd:.3f})" if mean is not None and sd is not None else "Undefined"
            result.append(row)
        return result
    save_csv(output_directory / "clinical_detection.csv", table(("sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "mcc")))
    save_csv(output_directory / "clinical_segmentation.csv", table(("accuracy", "f1", "dice_positive", "dice_tp", "pixel_dice_micro")))
    local = table(("auroc", "auprc_ap"))
    tuned = table(("sensitivity", "specificity"), "youden")
    for row, rr in zip(local, tuned):
        row["youden_sensitivity"], row["youden_specificity"] = rr["sensitivity"], rr["specificity"]
    save_csv(output_directory / "clinical_localisation.csv", local)
    save_csv(output_directory / "clinical_confusion_counts.csv", table(("tp", "fn", "fp", "tn")))
    comparison = []
    if verify_reference:
        expected = REPO / "data" / "reference" / "expected_main_descriptive.csv"
        with expected.open(encoding="utf-8", newline="") as f:
            for ref in csv.DictReader(f):
                key = (ref["architecture"], ref["strategy"], ref["threshold_mode"])
                for metric in METRICS:
                    for statistic in ("mean", "sd"):
                        column = f"{metric}_{statistic}"
                        actual, wanted = by[key][column], float(ref[column])
                        difference = abs(actual - wanted)
                        comparison.append(dict(architecture=key[0], strategy=key[1], threshold_mode=key[2], metric=metric, statistic=statistic, actual=actual, expected=wanted, absolute_difference=difference))
                        if difference > 1e-10:
                            raise ValueError(f"Published summary mismatch: {key} {column}: {actual} vs {wanted}")
        save_csv(output_directory / "published_descriptive_comparison.csv", comparison)
    report = dict(status="PASS", input_file=input_path.name, input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  records=len(rows), unique_regions=369, positive_regions=184, negative_regions=185, bitewings=25,
                  fitted_models=120, operating_points=2, seed_summaries=len(seed_rows), published_values_checked=len(comparison),
                  maximum_published_difference=max((r["absolute_difference"] for r in comparison), default=None),
                  aggregation="Per fitted model; mean over five folds within each seed; mean and sample SD across three seeds. Undefined individual metrics omitted within seed, never replaced by zero.",
                  localisation_scores="Maximum prediction within the reference lesion on positive regions; maximum anywhere in the evaluated region on negatives. Reference-conditioned localisation summaries.")
    (output_directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO / "data/reference/main_region_predictions.csv.gz")
    parser.add_argument("--out", type=Path, default=REPO / "results/descriptive")
    parser.add_argument("--verify-reference", action="store_true", help="Compare all 17 main descriptive metrics at both operating points with the supplied reference summaries")
    args = parser.parse_args()
    reproduce(args.input, args.out, args.verify_reference)
