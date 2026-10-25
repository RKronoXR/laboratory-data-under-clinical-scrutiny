# Main experiment backend

Run commands from the repository root. The top-level `run.py main` command forwards to this backend; the direct entry point is `pipelines/main/reproduce.py`.

```bash
python pipelines/main/reproduce.py plan
python pipelines/main/reproduce.py check --annotation-version updated
python pipelines/main/reproduce.py prepare --annotation-version updated
python pipelines/main/reproduce.py train --annotation-version updated
python pipelines/main/reproduce.py assess --annotation-version updated
python pipelines/main/reproduce.py status --annotation-version updated
```

The default input is `data/Datasets.zip`; use `--data-zip` to select another assembled archive. The default output is `runs/main`. Training requires the pinned GPU environment. Preparation uses image-processing dependencies and does not train or initialise CUDA. `check --runtime` additionally checks installed training versions.

## Annotation versions

`--annotation-version recorded` is the strict default. It requires the scientific input files used for the archived experiments. Enclosing folder aliases and ZIP metadata may differ, but input image, mask and coordinate-file checksums must match.

`--annotation-version updated` explicitly selects the revised Refined DCBR masks supplied with this release. The images, anatomical regions, original splits and model resolutions stay unchanged. These masks differ from the archived training labels; resulting model weights and performance can differ, with no assumed size or direction of the change.

In updated mode, a small lesion that disappears on a downsampled training mask is retained in its native evaluation mask. `resolution_audit.csv` records these regions. The code neither deletes those regions nor increases the image resolution to conceal the loss. The supplied context-exclusion flags remain active. Disagreements between the earlier review-presence labels and the revised native masks are recorded in the same audit; the revised masks define lesion presence. All other input checks remain active. Each run freezes its annotation-version identity; switching versions requires a different output directory.

## Training graph

The main menu contains exactly eight endpoint strategies: four nnU-Net strategies and four YOLO strategies. Clinical-only nnU-Net is the same trained model that supplies the clinical teacher predictions to YOLO. Laboratory pre-training models are shared between the corresponding routes; the final clinical student models are independently fitted.

For the complete three-seed, five-fold matrix, the graph contains 120 clinical endpoint models and 30 additional shared laboratory pre-training fits: 150 unique training stages in total. `--seed 17 --fold 0` selects a one-cell reproduction; keep the same subset arguments for every command in that run. Subset outputs are labelled in `RUN_MANIFEST.json` and are not the complete manuscript matrix.

The archived clinical nnU-Net plan and split files are included under `configs/main`. No replacement folds or new plans are generated. The clinical native recipe preserves the audited runtime's 12 augmentation workers and compiled training; inference is uncompiled. A numerical native-training equivalence check runs before the later nnU-Net stages.

## Assessment and outputs

`train` uses only development annotations and freezes the selected model and threshold hashes. `assess` refuses to proceed until all requested strategies are frozen. It evaluates the separate clinical data at 0.5 and at the frozen, model-specific development Youden thresholds.

Outputs include:

- `RUN_MANIFEST.json`: data, code, settings, annotation version and requested cells.
- `DEVELOPMENT_FROZEN.json`: selected weights and threshold checksums.
- `reports/per_model_metrics.csv`: fitted-model metrics with full strategy names.
- `reports/surface_results.csv`: proximal-region outcomes, both operating thresholds, and identifiers needed for paired analysis.
- `reports/regional_predictions.csv.gz`: the canonical clinical assessment export used by the repository descriptive and statistical analysis, with consistent anonymous region and bitewing identifiers.
- `reports/strategy_mapping.json`: full strategy names and internal source labels.
- `yolo/NO_HPO` and `nnunet/CONTROLLED_V3_NO_HPO`: checkpoint, calibration, native-input and prediction receipts from the numerical engines.

The separate repository statistical analysis reproduces the manuscript's planned baseline comparisons. The archived broad comparison-report menus are not called by this backend.

## Source provenance

The numerical engines came from the source archive listed in `UPSTREAM_MANIFEST.json`. Original file hashes and bundled hashes are both recorded for each explicitly documented updated-annotation admission patch. Native training, losses, augmentations, inference, thresholds and metrics are otherwise retained from the audited implementations. Packaging checks and real-data preparation are distinct from a complete GPU rerun; see the release validation report for what was actually executed.
