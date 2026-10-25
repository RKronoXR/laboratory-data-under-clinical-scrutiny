# Architecture comparison

This pipeline covers the nine architecture and input combinations in the development benchmark: nnU-Net, YOLO and SegResNet applied to laboratory proximal-side region crops, clinical proximal-side region crops and full clinical bitewings. Each group has three seeds and five folds. Model-specific settings, native nnU-Net plans, grouped splits and evaluation-region membership are supplied in `configs/benchmark`.

Run commands from the repository root. No private machine paths are needed.

## Reproduce the manuscript tables on CPU

```bash
python pipelines/benchmark/reproduce.py reproduce --work-dir runs/benchmark_tables
```

This uses only the Python standard library. It rebuilds Table 5.4 and Tables B21–B24 in `runs/benchmark_tables/reproduced_tables`. The benchmark means and sample SDs are calculated from the supplied seed summaries and checked against 1,980 stored summary values. The additional crop reports are recalculated from 11,638 regional prediction records. Partial fold coverage is retained and is never presented as a three-seed SD. This operation reproduces numerical tables from recorded outputs; it does not retrain models or create new predictions.

To generate Figures 5.4 and 5.5 with the three original input panels, install the optional plotting dependencies and run:

```bash
python tools/plot_benchmark.py
```

The tool uses Matplotlib 3.11.2 and no ML framework. It saves editable SVGs and 600-dpi PNGs in `results/figures`, using the actual three-seed means and sample SDs from the supplied recorded summaries.

## Prepare data

Use the study archive assembled by the main repository data instructions. Source folder names are discovered from their contents. Radiographs remain local and are not included in this repository.

Preprocessing requires Pillow **10.2.0**, as captured for the recorded experiments. The command checks this before doing work. Later Pillow releases can produce different resized or encoded PNG bytes and are not substituted silently.

The current released clinical masks are explicitly versioned as `updated`:

```bash
python pipelines/benchmark/reproduce.py prepare --dataset-zip data/Datasets.zip --work-dir runs/benchmark_updated --annotation-version updated
python pipelines/benchmark/reproduce.py check --work-dir runs/benchmark_updated --annotation-version updated
```

Updated mode verifies every source file against the supplied updated-data fingerprint manifest. It retains the original image processing, anatomical regions, context-exclusion rules, grouped folds and training recipes. Region positivity is derived from the selected masks, and `data/annotation_label_audit.csv` records differences from the recorded benchmark labels. The source masks themselves are never edited. New model results with updated annotations need not equal the manuscript’s recorded results.

The default `recorded` mode requires the frozen image and mask pixels used for the recorded benchmark. It checks all 2,844 captured prepared PNG hashes and stops on an input mismatch. Updated masks are not silently accepted as the recorded annotations. Use a separate work directory for each annotation version; checkpoints and probability maps cannot be reused across versions.

## Plan and train

Install the repository’s documented CUDA training environment first, using Linux or WSL2 for the native training kernels. GPU training uses the selected visible CUDA device; set `CUDA_VISIBLE_DEVICES` before launching if necessary. Training kernels require the captured framework versions. A complete benchmark has 135 fits and is computationally substantial.

Preview one fit without importing ML frameworks:

```bash
python pipelines/benchmark/reproduce.py train --group clinical-crops-yolo --seed 17 --fold 0 --work-dir runs/benchmark_updated --annotation-version updated --dry-run
```

Run that fit, or explicitly select the full benchmark:

```bash
python pipelines/benchmark/reproduce.py train --group clinical-crops-yolo --seed 17 --fold 0 --work-dir runs/benchmark_updated --annotation-version updated
python pipelines/benchmark/reproduce.py train --group all --work-dir runs/benchmark_updated --annotation-version updated
```

Group names are `laboratory-yolo`, `laboratory-nnunet`, `laboratory-segresnet`, `clinical-crops-yolo`, `clinical-crops-nnunet`, `clinical-crops-segresnet`, `clinical-bitewings-yolo`, `clinical-bitewings-nnunet` and `clinical-bitewings-segresnet`.

Each model has its own seed/fold directory and runs in a fresh worker interpreter. The native nnU-Net plans are reused; no new architecture planner or hyperparameter search is run. The laboratory nnU-Net trainer extension is installed into the active training environment when needed. Existing completed fits are reused only after checking their input version and checkpoint hash. Native checkpoints retain the state required by their recorded training implementations.

## Assess held-out development images

```bash
python pipelines/benchmark/reproduce.py assess --group all --work-dir runs/benchmark_updated --annotation-version updated
python pipelines/benchmark/reproduce.py report --work-dir runs/benchmark_updated --annotation-version updated
```

`assess` evaluates only each model’s held-out development fold. It writes regional decisions, per-model metrics, pooled out-of-fold seed metrics and thresholds. Fixed 0.5 results can be produced for available folds; development-selected Youden results require all five folds of a seed. Summaries require complete seed coverage and retain unavailable values. The recorded paired clinical region membership is used for crop-versus-bitewing comparisons, with a separate native-crop report domain.

The main training-strategy pipeline provides the separate clinical assessment reported in the manuscript.

Pixel summaries use the recorded letterbox-content domain. Detection and Dice summaries retain their regional evaluation grids. The native crop thresholds are also used for the paired crop report; a separate threshold is not refitted on the paired subset.

## Additional crop reports

The crop variants in Tables B21–B24 reuse the same crop training kernels. Their report scope is supplied separately in `configs/benchmark/additional_report_coverage.json`. The archived partial SegResNet report includes only its recorded model coverage. Completing more fits does not retroactively change that report. CPU table reproduction preserves that snapshot explicitly.

## Outputs and verification

Preparation creates local `data/processed`, `data/full` and annotation audit files. Model records are under `models/GROUP/seed_N/fold_N`, with checkpoint and probability hashes. New held-out metrics are under `assessment`, and summaries are under `reports`. Table-only reproduction uses a separate `reproduced_tables` directory.

Recorded data preparation and numerical table reproduction can be checked without a GPU. Full CUDA training must be validated in the training environment; a CPU check does not establish that GPU training has been executed.
