# Training workflows

The three public workflow families are separate so that the main results, the development architecture comparison and the additional experiments cannot be confused.

## Main clinical strategies

| Public strategy name | Training route | Teacher during clinical training |
|---|---|---|
| `nnunet_clinical` | Clinical bitewings | None |
| `nnunet_laboratory_pretraining` | Laboratory crops, then clinical bitewings | None |
| `nnunet_laboratory_distillation` | Clinical baseline, then clinical training with laboratory guidance | Laboratory YOLO |
| `nnunet_pretraining_and_distillation` | Laboratory crops, then clinical training with laboratory guidance | Laboratory YOLO |
| `yolo_clinical` | Clinical bitewings | None |
| `yolo_laboratory_pretraining` | Laboratory crops, then clinical bitewings | None |
| `yolo_clinical_distillation` | Clinical training with teacher guidance | Clinical nnU-Net |
| `yolo_pretraining_and_distillation` | Laboratory crops, then clinical training with teacher guidance | Clinical nnU-Net |

The orchestrator automatically constructs dependencies, including the clinical nnU-Net teacher and laboratory YOLO teacher, rather than expecting old trained models to exist. It reuses shared stages within the same reproduction run. Teachers remain frozen during distillation. Each public strategy is assessed separately for seeds 17, 42 and 2026 and folds 0–4.

Inspect the plan without loading GPU frameworks:

```bash
python run.py main plan
```

Prepare and train using the updated annotation release:

```bash
python run.py main prepare --data-zip data/Datasets.zip --annotation-version updated
python run.py main train --data-zip data/Datasets.zip --annotation-version updated
```

For the original annotation snapshot, use `--annotation-version recorded` with the original matching data. Each mode verifies its own scientific input fingerprints. A renamed outer folder or repacked ZIP is accepted when the scientific file bytes match; changed annotation content is not silently accepted as the same experiment.

Default outputs are under `runs/main/`. A run may be resumed using its matching data, source and protocol identity. To start a different experiment, choose a new `--output` directory. To inspect progress:

```bash
python run.py main status
```

## Fixed plans and model selection

`configs/main/clinical_plans.json` contains the recorded native nnU-Net plan. `clinical_splits.json` contains the actual clinical fold assignments. `native_recipe.json` captures the selected native training recipe. The configuration files specify the optimisation, augmentation, initialisation, checkpoint-selection and distillation settings.

Preprocessing uses the supplied plan; it does not call a fresh automatic planner to choose a different network. The wrappers provide portable paths while keeping the numerical implementations and their documented version-specific preprocessing changes traceable in the source inventories.

Clinical nnU-Net checkpoints use EMA foreground Dice. Clinical YOLO selection prioritises regional localisation ROC area, then localisation average precision, then Dice-Positive. The architecture-comparison criteria are separately configured. Study-derived weights are transferred only at the stages that call for them; external pretrained weights are not substituted.

## Architecture comparison and additional experiments

The architecture comparison has nine named groups, with one explicit recipe per seed–fold fit under `configs/benchmark/`. Run `python run.py benchmark plan` to see all groups and their recorded settings. The same entry point prepares inputs, trains, assesses held-out development images and writes summaries.

The additional experiment wrapper preserves the routes and implementation versions described in Appendix B. Its commands are listed by `python run.py additional --help`. The additional crop variants reuse the appropriate benchmark models, with their own report scope and coverage manifest under `configs/benchmark/additional_report_coverage.json`.

## Environment and new-run records

Install `requirements/training.txt` in a separate environment and use the appropriate PyTorch/CUDA build for the GPU. Run `python run.py doctor --training` before training and `python tools/capture_environment.py` to retain the actual environment. The model backends also record run identities, selected checkpoint hashes and thresholds.

The recorded hardware and package versions identify the source experiment. They do not guarantee bit-identical training on a different GPU, driver or numerical stack. Use the supplied seeds, splits and settings and retain the generated audit files with any new results.
