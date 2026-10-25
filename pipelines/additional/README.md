# Additional nnU-Net pipelines

This entry point covers the two additional native nnU-Net routes and the six earlier full-frame nnU-Net implementations reported separately in Appendix B. Additional clinical-crop architecture variants are handled by the architecture-comparison backend.

Complete the matching main training run first. Then run:

```bash
python pipelines/additional/reproduce.py plan
python pipelines/additional/reproduce.py check --annotation-version updated
python pipelines/additional/reproduce.py prepare --annotation-version updated
python pipelines/additional/reproduce.py train --annotation-version updated
python pipelines/additional/reproduce.py assess --annotation-version updated
python pipelines/additional/reproduce.py status --annotation-version updated
```

Defaults are `data/Datasets.zip`, `runs/main` for shared source models, and `runs/additional` for these results. `--main-output` selects a different main run. The annotation version must match the source run. Preserve the same `--seed` and `--fold` subset arguments if using a limited reproduction.

## Profiles

- `--profile native`: clinical training followed by laboratory training; or laboratory-teacher-guided clinical training followed by laboratory training. The main clinical baseline and laboratory-guided model supply their starting weights. There are two additional fits per seed/fold.
- `--profile earlier`: six archived full-frame nnU-Net routes. These retain their own optimisation, supervised loss, augmentation, input grid and clinical validation selection. They are not substituted for native nnU-Net training. Their laboratory pre-training checkpoint is shared once per seed, as in the archived implementation.
- `--profile all`: runs the two profiles in separate processes to keep their numerical implementations isolated.

For all three seeds and five folds, these profiles produce 30 and 90 endpoint models respectively. Earlier full-frame training uses 93 unique stages because the three laboratory pre-training fits are shared across folds. These results are exported separately and are not pooled with the eight main strategies.

## Output and protection

Each profile has a separate folder under `runs/additional`, with frozen data/configuration identity, selected model and threshold hashes, development and assessment results, and friendly-name `reports/per_model_metrics.csv` and `reports/surface_results.csv` exports. Assessment requires a completed development freeze. Existing outputs belonging to another configuration are rejected rather than overwritten.

`UPSTREAM_MANIFEST.json` records the earlier numerical source files and the explicit updated-mask admission patch. The shared main backend contains the exact native implementation and frozen clinical plans/splits. No alternative input resolution, replacement folds, automatic hyperparameter search, or unreported endpoint groups are selected by this public menu.
