#!/usr/bin/env python3
"""Portable development benchmark preparation, training, and held-out assessment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = REPO / "configs" / "benchmark"
sys.path.insert(0, str(HERE / "_vendor"))
SEEDS = (17, 42, 2026)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def groups():
    return {g["name"]: g for g in load(CONFIG / "groups.json")}


def selected(args):
    chosen = args.group or ("all" if args.command in ("plan", "check", "assess", "report") else None)
    if chosen is None:
        raise ValueError("Select --group NAME or --group all explicitly before training.")
    return [r for r in load(CONFIG / "model_recipes.json")
            if (chosen == "all" or r["group"] == chosen)
            and (args.seed is None or r["seed"] == args.seed)
            and (args.fold is None or r["fold"] == args.fold)]


def frame(path):
    import pandas as pd
    return pd.read_csv(path, dtype={"sample_id": str, "bitewing_id": str, "group_id": str}, keep_default_na=False)


def save_frame(path, df):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def roots(args):
    work = args.work_dir.expanduser().resolve()
    return work, (args.data_root.expanduser().resolve() if args.data_root else work / "data")


def safe_extract(archive, target):
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            name = item.filename.replace("\\", "/")
            if name.startswith("/") or ":" in name or ".." in Path(name).parts:
                raise ValueError("Unsafe archive member: " + name)
            dest = (target / name).resolve()
            if not dest.is_relative_to(target.resolve()):
                raise ValueError("Archive member leaves the extraction directory.")
        z.extractall(target)


def locate_source(raw, kind):
    if kind == "laboratory":
        found = [p.parent.parent for p in raw.rglob("2_Proximal_Sides")
                 if p.is_dir() and (p.parent.parent / "4_Radiograph_Annotations" / "2_Proximal_Sides_Annotations").is_dir()]
    else:
        found = [p.parent for p in raw.rglob("Proximal_Side_Regions.csv")
                 if (p.parent / "bitewings").is_dir() and (p.parent / "masks").is_dir()]
    found = sorted(set(found))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {kind} dataset with the required images, masks and region files; found {len(found)}.")
    return found[0]


def verify_updated_archive(path):
    """Accept renamed or repacked containers only when every frozen file matches."""
    sys.path.insert(0, str(REPO / "src"))
    from carieslab.datasets import _discover, _fingerprint_records, _stream_sha256
    manifest = REPO / "configs/main/data_fingerprints_updated.json"
    if not manifest.is_file():
        raise FileNotFoundError("Frozen updated-annotation file fingerprints are required.")
    required = _fingerprint_records(manifest)
    with zipfile.ZipFile(path) as archive:
        files, by_prefix, prefixes = _discover(archive)
        lookup = {}
        for domain, prefix in prefixes.items():
            for info, relative in by_prefix[prefix]:
                lookup[(domain, "/".join(relative))] = info
        for item in required:
            info = lookup.get((item["domain"], item["path"]))
            if info is None:
                raise FileNotFoundError("Updated dataset file is absent: " + item["domain"] + "/" + item["path"])
            with archive.open(info) as source:
                actual = _stream_sha256(source)
            if actual != item["sha256"]:
                raise ValueError("Updated dataset file differs from its frozen version: " + item["domain"] + "/" + item["path"])
    return sha(manifest)


def prepare(args):
    import numpy as np
    import pandas as pd
    from PIL import Image, __version__ as pillow_version
    from proxcaries.data import prepare as original_prepare
    from proxcaries.data.prepare import prepare_acta_direct, _prepare_bitewing_dataset
    from proxcaries.data.transforms import load_radiograph_uint8, load_binary_mask, letterbox_square
    work, data = roots(args)
    if pillow_version != "10.2.0":
        raise RuntimeError(f"Benchmark preprocessing requires Pillow 10.2.0; found {pillow_version}. Install the documented preprocessing environment before preparing data.")
    if not args.dataset_zip or not args.dataset_zip.is_file():
        raise FileNotFoundError("Provide --dataset-zip pointing to the released study image/mask archive.")
    archive_hash = sha(args.dataset_zip)
    annotation_fingerprint = verify_updated_archive(args.dataset_zip) if args.annotation_version == "updated" else None
    receipt = data / "prepared.json"
    if receipt.exists():
        saved = load(receipt)
        if saved.get("annotation_version", "recorded") != args.annotation_version:
            raise RuntimeError("This work directory uses another annotation version. Use a separate --work-dir.")
        if saved["dataset_zip_sha256"] != archive_hash and not (args.annotation_version == "updated" and saved.get("annotation_fingerprint") == annotation_fingerprint):
            raise RuntimeError("The source archive changed. Use a separate --work-dir.")
        return check(args)
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    marker = raw / ".archive_sha256"
    if not marker.exists():
        safe_extract(args.dataset_zip, raw)
        marker.write_text(archive_hash, encoding="utf-8")
    elif marker.read_text().strip() != archive_hash:
        raise RuntimeError("An existing extraction belongs to a different archive.")
    lab = locate_source(raw, "laboratory")
    clinical = locate_source(raw, "clinical")
    clinical_membership = frame(CONFIG / "clinical_crops.csv").set_index("sample_id").fold.astype(int).to_dict()
    laboratory_membership = {}
    for f, split in enumerate(load(CONFIG / "plans/laboratory_splits_final.json")):
        laboratory_membership.update({sid: f for sid in split["val"]})
    def frozen_folds(df, n_splits=5, seed=2026):
        lookup = laboratory_membership if set(df.source) == {"acta_direct"} else clinical_membership
        if set(df.sample_id) != set(lookup):
            raise ValueError("Source case identities differ from the frozen split membership.")
        out = df.copy(); out["fold"] = out.sample_id.map(lookup).astype(int)
        if out.groupby("group_id").fold.nunique().max() != 1:
            raise ValueError("The frozen assignment separates regions from the same source image or tooth.")
        return out
    original_prepare.assign_grouped_folds = frozen_folds
    # Only location discovery is adapted; the original pixel transformations are retained.
    labdf = prepare_acta_direct(lab.parent, data / "processed", 384, 5, 2026, source_directory=lab)
    clindf = _prepare_bitewing_dataset("dcbr", clinical, "bitewings", "masks", data / "processed", 384, 5, 2026, annotation_version=args.annotation_version)
    expected = frame(CONFIG / "clinical_crops.csv").set_index("sample_id")
    actual = clindf.set_index("sample_id")
    if set(actual.index) != set(expected.index):
        raise ValueError("Prepared clinical region identities differ from the frozen experiment membership.")
    keys = ["x1", "y1", "x2", "y2", "lb_resized_width", "lb_resized_height", "lb_pad_left", "lb_pad_top"]
    if args.annotation_version == "recorded":
        keys.insert(0, "lesion_present")
    for key in keys:
        if not np.array_equal(actual.loc[expected.index, key].astype(float), expected[key].astype(float)):
            raise ValueError("Prepared clinical regions disagree with the frozen " + key)
    clindf["fold"] = clindf.sample_id.map(expected.fold.astype(int)).astype(int)
    label_audit = pd.DataFrame(dict(sample_id=expected.index,
                                   recorded_label=expected.lesion_present.astype(int).values,
                                   prepared_label=actual.loc[expected.index, "lesion_present"].astype(int).values))
    label_audit["changed"] = label_audit.recorded_label != label_audit.prepared_label
    save_frame(data / "annotation_label_audit.csv", label_audit)
    # Reuse the exact published split rather than rerunning a planner or split search.
    labfolds = {}
    for fold, split in enumerate(load(CONFIG / "plans/laboratory_splits_final.json")):
        labfolds.update({key: fold for key in split["val"]})
    if set(labfolds) != set(labdf.sample_id):
        raise ValueError("Laboratory region identities differ from the frozen native split.")
    labdf["fold"] = labdf.sample_id.map(labfolds).astype(int)
    if len(labdf) != 2270 or int(labdf.lesion_present.sum()) != 850:
        raise ValueError("Laboratory input inventory differs from the recorded experiment.")
    save_frame(data / "processed/acta_direct/manifest.csv", labdf)
    save_frame(data / "processed/dcbr/manifest.csv", clindf)
    imagefolds = {}
    for split in load(CONFIG / "clinical_image_splits.json"):
        imagefolds.update({key.removeprefix("dcbr_bw_"): int(split["fold"]) for key in split["test_groups"]})
    full = []
    for bid in sorted(imagefolds):
        image = next((p for p in (clinical / "bitewings").iterdir() if p.is_file() and p.stem.isdigit() and int(p.stem) == int(bid)), None)
        mask = next((p for p in (clinical / "masks").iterdir() if p.is_file() and p.stem.isdigit() and int(p.stem) == int(bid)), None)
        if image is None or mask is None:
            raise FileNotFoundError("Missing clinical full image/mask: " + bid)
        img, target = load_radiograph_uint8(image), load_binary_mask(mask)
        if img.shape != target.shape:
            raise ValueError("Full image and lesion mask have different dimensions: " + bid)
        x, meta = letterbox_square(img, 384)
        y, _ = letterbox_square(target, 384, True)
        ip, mp = data / f"full/images/{bid}.png", data / f"full/masks/{bid}.png"
        ip.parent.mkdir(parents=True, exist_ok=True); mp.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(x).save(ip, compress_level=3); Image.fromarray(y).save(mp, compress_level=3)
        full.append(dict(sample_id="bw" + bid, bitewing_id=bid, group_id="dcbr_bw_" + bid, patient_id="", fold=imagefolds[bid],
                         image_path=str(ip), mask_path=str(mp), native_image_path=str(image), native_mask_path=str(mask),
                         native_width=img.shape[1], native_height=img.shape[0], width=img.shape[1], height=img.shape[0],
                         lb_width=meta.resized_width, lb_height=meta.resized_height, lb_left=meta.pad_left, lb_top=meta.pad_top))
    fullframe = pd.DataFrame(full)
    oldregions = frame(CONFIG / "clinical_evaluation_regions.csv")
    current = clindf.set_index("sample_id")
    if not set(oldregions.sample_id).issubset(current.index):
        raise ValueError("The frozen development evaluation regions are unavailable.")
    for key in ("image_path", "mask_path"):
        oldregions[key] = oldregions.sample_id.map(current[key])
    oldregions["reference_mask_path"] = oldregions["mask_path"]
    oldregions["crop_mask_path"] = oldregions["mask_path"]
    oldregions["crop_image_path"] = oldregions["image_path"]
    save_frame(data / "full/bitewings.csv", fullframe)
    save_frame(data / "full/surfaces.csv", oldregions)
    # Compatibility directory used only by the unchanged MONAI training kernel.
    save_frame(work / "output/data/bitewings.csv", fullframe)
    save_frame(work / "output/data/surfaces.csv", oldregions)
    updated_masks = {}
    if args.annotation_version == "updated":
        for folder in [data / "processed/dcbr/masks", data / "full/masks"]:
            updated_masks.update({p.relative_to(data).as_posix(): sha(p) for p in folder.glob("*.png")})
    checked = verify_pngs(data, args.annotation_version, updated_masks)
    save(receipt, dict(dataset_zip_sha256=archive_hash, pngs_verified=checked,
                       preprocessing_software={"Pillow":pillow_version},
                       annotation_version=args.annotation_version, annotation_fingerprint=annotation_fingerprint,
                       updated_masks_sha256=updated_masks, changed_region_labels=int(label_audit.changed.sum()),
                       clinical_positive_regions=int(clindf.lesion_present.sum()),
                       context_exclusion_rules="Original case-085 sound-region exclusions retained; positivity expectations are derived from the selected annotation version.",
                       laboratory_regions=len(labdf), clinical_regions=len(clindf),
                       frozen_evaluation_regions=len(oldregions), full_images=len(fullframe)))
    return check(args)


def verify_pngs(data, annotation_version="recorded", updated_masks=None):
    n = 0
    with (CONFIG / "prepared_png_sha256.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            base = data / ("processed/dcbr" if row["dataset"] == "clinical_crops" else "full")
            path = base / row["kind"] / row["filename"]
            if annotation_version == "updated" and row["kind"] == "masks":
                expected = (updated_masks or {}).get(path.relative_to(data).as_posix())
                if expected is None or not path.is_file() or sha(path) != expected:
                    raise RuntimeError("An updated prepared mask changed: " + path.name)
                n += 1
                continue
            if not path.is_file() or sha(path) != row["sha256"]:
                raise RuntimeError(f"Prepared pixel file differs from the frozen study input: {path.name}. Use the exact released source archive.")
            n += 1
    return n


def check(args):
    work, data = roots(args)
    if not (data / "prepared.json").exists():
        raise FileNotFoundError("Run prepare before checking or training the benchmark.")
    receipt = load(data / "prepared.json")
    if receipt.get("annotation_version", "recorded") != args.annotation_version:
        raise RuntimeError("Annotation version differs from this work directory. Select the recorded version explicitly.")
    verified = verify_pngs(data, args.annotation_version, receipt.get("updated_masks_sha256"))
    versions = {}
    for package in ["numpy", "pandas", "Pillow", "scikit-learn", "torch", "ultralytics", "monai", "nnunetv2"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = dict(status="data-verified", verified_pngs=verified, preparation=load(data / "prepared.json"), versions=versions)
    save(work / "checks/data_check.json", result)
    shown = dict(result, preparation={key:value for key,value in result["preparation"].items() if key != "updated_masks_sha256"})
    print(json.dumps(shown, indent=2))
    return result


def native_bundle(work, df, plans, dataset_name, kind):
    from crops100.io import atomic_json
    root = work / "native" / kind
    raw, pre = root / "raw" / dataset_name, root / "preprocessed" / dataset_name
    raw.mkdir(parents=True, exist_ok=True); pre.mkdir(parents=True, exist_ok=True)
    for row in df.itertuples():
        for source, dest in [(row.image_path, raw / "imagesTr" / (row.sample_id + "_0000.png")),
                             (row.mask_path, raw / "labelsTr" / (row.sample_id + ".png"))]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                try:
                    dest.symlink_to(Path(source).resolve())
                except OSError:
                    shutil.copy2(source, dest)
    plans = dict(plans); plans["dataset_name"] = dataset_name
    dataset = dict(channel_names={"0": "grayscale"}, labels={"background": 0, "caries": 1}, numTraining=len(df), file_ending=".png")
    atomic_json(raw / "dataset.json", dataset); atomic_json(pre / "dataset.json", dataset)
    atomic_json(pre / "nnUNetPlans.json", plans)
    splits = [{"train": df.loc[df.fold.astype(int) != f, "sample_id"].tolist(), "val": df.loc[df.fold.astype(int) == f, "sample_id"].tolist()} for f in range(5)]
    atomic_json(pre / "splits_final.json", splits)
    source_csv = root / "source_manifest.csv"; save_frame(source_csv, df)
    return dict(raw_root=str(root / "raw"), preprocessed_root=str(root / "preprocessed"), raw_directory=str(raw), preprocessed_directory=str(pre),
                plans_file=str(pre / "nnUNetPlans.json"), dataset_json_file=str(raw / "dataset.json"),
                manifest_path=str(source_csv), full_frame_path=str(source_csv), configuration="2d", plans_name="nnUNetPlans",
                dataset_name=dataset_name, validation_ids_by_fold={str(f): splits[f]["val"] for f in range(5)})


def install_native_extension():
    import nnunetv2
    destination = Path(nnunetv2.__file__).resolve().parent / "training/nnUNetTrainer/variants/proxcaries"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "__init__.py").touch()
    source = HERE / "_vendor/nnunet_ext/nnUNetTrainerCariesEarlyStop.py"
    shutil.copy2(source, destination / source.name)


def train(args):
    recipes = selected(args)
    if not args.worker:
        # A separate interpreter preserves native framework path globals and RNG
        # isolation across architectures, datasets, seeds and folds.
        for item in recipes:
            cmd = [sys.executable, str(HERE / "reproduce.py"), "train", "--group", item["group"],
                   "--seed", str(item["seed"]), "--fold", str(item["fold"]),
                   "--work-dir", str(roots(args)[0]), "--annotation-version", args.annotation_version, "--worker"]
            if args.data_root:
                cmd.extend(["--data-root", str(roots(args)[1])])
            subprocess.run(cmd, check=True)
        return
    if len(recipes) != 1:
        raise ValueError("Each training worker must select exactly one group, seed and fold.")
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark model training requires the documented CUDA environment. CPU data checks are available with prepare and check.")
    work, data = roots(args)
    if not (data / "prepared.json").is_file():
        raise FileNotFoundError("Run benchmark prepare first.")
    prepared = load(data / "prepared.json")
    if prepared.get("annotation_version", "recorded") != args.annotation_version:
        raise RuntimeError("Train with the same --annotation-version used by prepare. Annotation versions cannot share a training work directory.")
    mapping = groups()
    for rec in selected(args):
        group = mapping[rec["group"]]; seed, fold = rec["seed"], rec["fold"]
        directory = work / "models" / rec["group"] / f"seed_{seed}" / f"fold_{fold}"
        record_path = directory / "benchmark_model.json"
        if record_path.exists():
            saved = load(record_path)
            if saved.get("annotation_version", "recorded") != args.annotation_version or saved.get("data_signature") != signature(prepared):
                raise RuntimeError("Completed model belongs to different annotations or prepared inputs. Use a separate --work-dir.")
            if sha(saved["checkpoint"]) != saved["checkpoint_sha256"]:
                raise RuntimeError("Completed checkpoint has changed: " + str(directory))
            print("Already complete:", rec["group"], seed, fold)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        common_spec = dict(directory=str(directory), run_root=str(work / "activity"), stop_file=str(work / "STOP"),
                           signature=signature(dict(recipe=rec, data=load(data / "prepared.json"))), seed=seed, fold=fold)
        if group["dataset"] == "acta_direct":
            from proxcaries.training.common import RunSpec
            cfg = load(CONFIG / "laboratory_config.json")
            cfg["data"].update(work_dir="native_laboratory", runs_dir="models")
            df = frame(data / "processed/acta_direct/manifest.csv")
            spec = RunSpec(group["architecture"], "acta_direct", seed, fold, directory)
            if group["architecture"] == "monai_segresnet":
                from proxcaries.training.monai_runner import run_monai
                run_monai(df, spec, cfg); checkpoint = directory / "best.pt"
            elif group["architecture"] == "yolo26_sem":
                from proxcaries.training.yolo_runner import run_yolo
                run_yolo(df, spec, cfg); checkpoint = directory / "ultralytics/weights/best.pt"
            else:
                from proxcaries.training.nnunet_runner import run_nnunet
                bundle = native_bundle(work, df, load(CONFIG / "plans/laboratory_nnUNetPlans.json"), "Dataset601_ACTADIRECT", "laboratory")
                shutil.copy2(CONFIG / "plans/laboratory_dataset_fingerprint.json",Path(bundle["preprocessed_directory"])/"dataset_fingerprint.json")
                native = work / "native_laboratory/nnunet"
                native.mkdir(parents=True, exist_ok=True)
                for key in ("raw", "preprocessed"):
                    target = native / key
                    if not target.exists():
                        target.symlink_to(Path(bundle[key + "_root"]), target_is_directory=True)
                env = os.environ.copy(); env.update(nnUNet_raw=bundle["raw_root"], nnUNet_preprocessed=bundle["preprocessed_root"], nnUNet_results=str(native / "results"))
                install_native_extension()
                subprocess.run(["nnUNetv2_preprocess", "-d", "601", "-plans_name", "nnUNetPlans", "-c", "2d"], env=env, check=True)
                run_nnunet(df, spec, cfg, work)
                checkpoint = native / f"results/seed_{seed}/Dataset601_ACTADIRECT/nnUNetTrainerCariesEarlyStop__nnUNetPlans__2d/fold_{fold}/checkpoint_best.pth"
            paths = {r.sample_id: str(directory / "probabilities" / (r.sample_id + ".npy")) for r in df[df.fold.astype(int) == fold].itertuples()}
            probability_grid = "crop"
        elif group["input"] == "crops":
            from crops100.backends import train_fit
            df = frame(data / "processed/dcbr/manifest.csv")
            crop_project = work / "crop_models"
            crop_project.mkdir(exist_ok=True)
            # The native kernel resolves its local output configuration explicitly.
            eff = load(CONFIG / "clinical_crops_config.json")
            save(crop_project / "output/effective_config.json", eff)
            save_frame(crop_project / "output/data/manifest.csv", df)
            protocol = dict(manifest=str(crop_project / "output/data/manifest.csv"), signature=signature(dict(data=load(data / "prepared.json"), config=eff)))
            result = train_fit(crop_project, dict(architecture=group["architecture"], seed=seed, fold=fold), protocol)
            checkpoint = Path(result["checkpoint"])
            if result.get("probability_paths"):
                paths = result["probability_paths"]
            else:
                from crops100.backends import load_probability
                paths = {}
                for sid, path in result["native_npz_paths"].items():
                    p = directory / "probabilities" / (sid + ".npy"); p.parent.mkdir(exist_ok=True)
                    np.save(p, load_probability(path)); paths[sid] = str(p)
            probability_grid = "crop"
        else:
            images = frame(data / "full/bitewings.csv"); regions = frame(data / "full/surfaces.csv")
            tune = images[images.fold.astype(int) == fold]
            if group["architecture"] == "yolo26_sem":
                from dcbr_compare.training import train_candidate
                from dcbr_compare.evaluation import load_network, probability_maps
                cfg = load(CONFIG / "clinical_yolo_config.json")
                cfg["patience"] = rec["patience"]; cfg["training_args"]["patience"] = rec["patience"]
                result = train_candidate(dict(common_spec, condition="full", epochs=rec["epochs"]), images[images.fold.astype(int) != fold], tune,
                                         regions[regions.fold.astype(int) == fold], cfg)
                net = load_network(result["checkpoint"], cfg); paths = {}
                for row, probability in probability_maps(net, tune, cfg):
                    p = directory / "assessment_probabilities" / (row["bitewing_id"] + ".npy"); p.parent.mkdir(exist_ok=True)
                    np.save(p, probability); paths[row["bitewing_id"]] = str(p)
            elif group["architecture"] == "nnunetv2":
                from dcbr_nnunet.backend import train_and_predict
                from dcbr_compare.evaluation import restore_probability
                cfg = load(CONFIG / "clinical_nnunet_config.json")
                bundle = native_bundle(work, images, load(CONFIG / "plans/clinical_full_nnUNetPlans.json"), "Dataset702_DCBRFullBW", "clinical_full")
                result = train_and_predict(common_spec, bundle, cfg); paths = {}
                for row in tune.to_dict("records"):
                    probability = np.load(result["probability_paths"][row["bitewing_id"]], allow_pickle=False)
                    p = directory / "assessment_probabilities" / (row["bitewing_id"] + ".npy"); p.parent.mkdir(exist_ok=True)
                    np.save(p, restore_probability(probability, row)); paths[row["bitewing_id"]] = str(p)
            else:
                from dcbr_monai_full.training import train as fit_monai, evaluate as eval_monai
                cfg = load(CONFIG / "clinical_segresnet_config.json")
                save(work / "monai_full/output/effective_config.json", cfg)
                save(work / "monai_full/output/protocol.json", dict(signature=signature(dict(config=cfg, data=load(data / "prepared.json")))))
                result = fit_monai(work, seed, fold); eval_monai(work, seed, fold, result)
                paths = {r.bitewing_id: str(work / f"monai_full/output/runs/seed_{seed}/fold_{fold}/full/evaluation/probabilities/{r.bitewing_id}.npy") for r in tune.itertuples()}
            checkpoint = Path(result["checkpoint"]); probability_grid = "native_full"
        if not checkpoint.is_file() or any(not Path(p).is_file() for p in paths.values()):
            raise RuntimeError("Training did not produce every required checkpoint and held-out probability map.")
        save(record_path, dict(group=rec["group"], seed=seed, fold=fold, recipe=rec,
                              annotation_version=args.annotation_version,
                              checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                              probability_paths=paths, probability_grid=probability_grid,
                              probability_sha256={key:sha(path) for key,path in paths.items()},
                              transpose=bool(group["development_probability_transpose"]), data_signature=signature(load(data / "prepared.json"))))
        print("Complete:", rec["group"], seed, fold)


def model_rows(record, data, threshold=.5, paired=False):
    import numpy as np
    from PIL import Image
    from dcbr_compare.baseline import project_full_to_crop
    from dcbr_compare.metrics import surface_record
    group = groups()[record["group"]]
    if group["dataset"] == "acta_direct":
        df = frame(data / "processed/acta_direct/manifest.csv")
    elif group["input"] == "full":
        df = frame(data / "full/surfaces.csv")
    else:
        df = frame(data / "processed/dcbr/manifest.csv")
        if paired:
            membership = set(frame(data / "full/surfaces.csv").sample_id)
            df = df[df.sample_id.isin(membership)]
    df = df[df.fold.astype(int) == record["fold"]]
    loaded = {}; rows = []
    for row in df.to_dict("records"):
        key = row["bitewing_id"] if group["input"] == "full" else row["sample_id"]
        if key not in loaded:
            p = np.load(record["probability_paths"][key], allow_pickle=False)
            loaded[key] = p.T if record.get("transpose") else p
        probability = project_full_to_crop(loaded[key], row) if group["input"] == "full" else loaded[key]
        with Image.open(row["mask_path"]) as im:
            reference = np.asarray(im) > 0
        metric = surface_record(reference, probability, threshold)
        left = int(float(row.get("lb_pad_left", row.get("lb_left", 0))))
        top = int(float(row.get("lb_pad_top", row.get("lb_top", 0))))
        width = int(float(row.get("lb_resized_width", row.get("lb_width", reference.shape[1]))))
        height = int(float(row.get("lb_resized_height", row.get("lb_height", reference.shape[0]))))
        valid = np.zeros(reference.shape, dtype=bool)
        valid[top:top+height,left:left+width] = True
        predicted = probability >= threshold
        metric.update(pixel_tp=int((reference & predicted & valid).sum()),pixel_fp=int((~reference & predicted & valid).sum()),
                      pixel_tn=int((~reference & ~predicted & valid).sum()),pixel_fn=int((reference & ~predicted & valid).sum()),
                      pixel_scope="letterbox_content_only")
        metric.update(group=record["group"], seed=record["seed"], fold=record["fold"], sample_id=row["sample_id"],
                      bitewing_id=row["bitewing_id"], group_id=row["group_id"], threshold_name="fixed" if threshold == .5 else "youden")
        rows.append(metric)
    return rows


def metric_summary(rows, threshold_name):
    from dcbr_compare.metrics import summarize_records
    s = summarize_records(rows, threshold_name)
    tp,fp,tn,fn = [sum(int(r["pixel_"+key]) for r in rows) for key in ["tp","fp","tn","fn"]]
    s.update(pixel_tp=tp,pixel_fp=fp,pixel_tn=tn,pixel_fn=fn)
    s["pixel_dice_micro"] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    return s


def assess(args):
    import pandas as pd
    from dcbr_compare.metrics import choose_youden
    work, data = roots(args)
    if not (data / "prepared.json").exists():
        raise FileNotFoundError("Run benchmark prepare before assessment.")
    records = [load(p) for p in sorted((work / "models").glob("*/seed_*/fold_*/benchmark_model.json"))]
    prepared = load(data / "prepared.json")
    if prepared.get("annotation_version", "recorded") != args.annotation_version:
        raise RuntimeError("Assess with the same --annotation-version used for preparation and training.")
    requested = {(r["group"], r["seed"], r["fold"]) for r in selected(args)}
    records = [r for r in records if (r["group"], r["seed"], r["fold"]) in requested]
    if not records:
        raise FileNotFoundError("No completed benchmark models match this selection. Run train first.")
    for rec in records:
        if rec.get("annotation_version", "recorded") != args.annotation_version or rec.get("data_signature") != signature(prepared):
            raise RuntimeError("A selected model was trained with different annotations or prepared inputs.")
        if sha(rec["checkpoint"]) != rec["checkpoint_sha256"]:
            raise RuntimeError("A selected checkpoint changed after training.")
        for key,path in rec["probability_paths"].items():
            if not Path(path).is_file() or (rec.get("probability_sha256") and sha(path) != rec["probability_sha256"].get(key)):
                raise RuntimeError("A held-out probability map changed after training.")
    all_seed = []; per_model = []; thresholds = []
    for group_name in sorted({r["group"] for r in records}):
        group = groups()[group_name]
        domains = ["native"] if group["dataset"] == "acta_direct" else (["native", "paired"] if group["input"] == "crops" else ["paired"])
        for domain in domains:
            for seed in sorted({r["seed"] for r in records if r["group"] == group_name}):
                rs = [r for r in records if r["group"] == group_name and r["seed"] == seed]
                fixed = {r["fold"]: model_rows(r, data, .5, domain == "paired") for r in rs}
                calibration = ({r["fold"]: model_rows(r,data,.5,False) for r in rs}
                               if domain == "paired" and group["input"] == "crops" else fixed)
                operating = {"fixed": {f: x for f, x in fixed.items()}}
                if set(fixed) == set(range(5)):
                    tuned = {}
                    for r in rs:
                        fold = r["fold"]
                        peers = [x for f, rows in calibration.items() if f != fold for x in rows]
                        threshold = choose_youden(peers)
                        rows = model_rows(r, data, threshold, domain == "paired")
                        for row in rows:
                            row["threshold_name"] = "youden"
                        tuned[fold] = rows
                        thresholds.append(dict(group=group_name, domain=domain, seed=seed, fold=fold, threshold=threshold))
                    operating["youden"] = tuned
                for name, byfold in operating.items():
                    pooled = [x for rows in byfold.values() for x in rows]
                    for f, rows in byfold.items():
                        per_model.append(dict(group=group_name, domain=domain, seed=seed, fold=f, threshold_name=name, **metric_summary(rows, name)))
                    if set(byfold) == set(range(5)):
                        all_seed.append(dict(group=group_name, domain=domain, seed=seed, threshold_name=name, **metric_summary(pooled, name)))
                    out = work / "assessment" / group_name / domain / f"seed_{seed}_{name}.csv"
                    save_frame(out, pd.DataFrame(pooled))
    save_frame(work / "assessment/per_model_metrics.csv", pd.DataFrame(per_model))
    save_frame(work / "assessment/per_seed_metrics.csv", pd.DataFrame(all_seed) if all_seed else pd.DataFrame(columns=["group","domain","seed","threshold_name"]))
    save_frame(work / "assessment/thresholds.csv", pd.DataFrame(thresholds))
    return report(args)


def report(args):
    import numpy as np
    import pandas as pd
    work, data = roots(args)
    p = work / "assessment/per_seed_metrics.csv"
    if not p.exists():
        raise FileNotFoundError("Run assess before report.")
    df = frame(p); rows = []
    if df.empty:
        print("No complete five-fold seed summary is available yet. Per-model metrics remain in assessment/per_model_metrics.csv.")
        return []
    keys = ["group", "domain", "threshold_name"]
    for key, sub in df.groupby(keys):
        for metric in [c for c in df.columns if c not in keys + ["seed"]]:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            rows.append(dict(zip(keys, key), metric=metric, mean=float(vals.mean()) if vals.notna().any() else None,
                             sd=float(vals.std(ddof=1)) if vals.notna().sum() > 1 else None, n_seeds=int(vals.notna().sum())))
    save_frame(work / "reports/benchmark_summary.csv", pd.DataFrame(rows))
    coverage = load(CONFIG / "additional_report_coverage.json")
    additional = []
    for group, thresholds in coverage.items():
        for name, pairs in thresholds.items():
            for seed in sorted({s for s,f in pairs}):
                required = {f for s,f in pairs if s == seed}
                path = work / "assessment" / group / "native" / f"seed_{seed}_{name}.csv"
                if not path.exists():
                    continue
                saved = frame(path)
                saved = saved[saved.fold.astype(int).isin(required)]
                if set(saved.fold.astype(int)) != required:
                    continue
                values = metric_summary(saved.to_dict("records"),name)
                additional.append(dict(group=group,seed=seed,threshold_name=name,required_folds="/".join(str(f) for f in sorted(required)),**values))
    if additional:
        save_frame(work / "reports/additional_crop_coverage.csv",pd.DataFrame(additional))
    print("Benchmark summary:", work / "reports/benchmark_summary.csv")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "prepare", "check", "train", "assess", "report", "reproduce"])
    parser.add_argument("--dataset-zip", type=Path)
    parser.add_argument("--work-dir", type=Path, default=REPO / "runs/benchmark")
    parser.add_argument("--data-root", type=Path, help="Override the prepared-data directory, normally WORK/data.")
    parser.add_argument("--annotation-version", choices=["recorded", "updated"], default="recorded", help="Use strict recorded inputs or the explicitly versioned updated clinical masks.")
    parser.add_argument("--group", choices=["all", *groups()])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--dry-run", action="store_true", help="Show selected configuration and commands without loading ML libraries or starting work.")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.dry_run or args.command == "plan":
        print(json.dumps(dict(command=args.command, work_dir=str(roots(args)[0]), data_root=str(roots(args)[1]),
                              annotation_version=args.annotation_version,
                              dataset_zip=str(args.dataset_zip) if args.dataset_zip else None,
                              runs=selected(args) if args.command not in ("prepare", "reproduce") else [],
                              expected_groups=9, expected_models=135), indent=2))
        return 0
    try:
        if args.command == "reproduce":
            from reproduce_tables import run
            run(REPO, roots(args)[0])
        else:
            globals()[args.command](args)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(2, "Benchmark error: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
