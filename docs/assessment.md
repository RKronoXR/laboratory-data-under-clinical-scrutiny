# Assessment and metric calculation

## Generate predictions

After training and development selection have completed, run:

```bash
python run.py main assess --data-zip data/Datasets.zip --annotation-version updated
```

This uses the selected weights and frozen development thresholds. It does not refit models or calibrate thresholds on the separate clinical assessment annotations. Native images, masks and anatomical region coordinates are preserved. Probabilities are restored to the image dimensions before regional assessment.

For architecture-comparison development assessment:

```bash
python run.py benchmark assess --annotation-version updated
python run.py benchmark report --annotation-version updated
```

The model workflows write their per-model metrics and regional prediction records under their run folders. These raw records can be inspected independently of the tables.

The main workflow also writes `runs/main/reports/regional_predictions.csv.gz` in the common analysis format. To analyse a new full run, use:

```bash
python run.py reproduce --input runs/main/reports/regional_predictions.csv.gz --out results/new_run_tables --no-reference-check
python run.py statistics --input runs/main/reports/regional_predictions.csv.gz --out results/new_run_statistics --no-reference-check
```

These commands retain the same outcome and methods while skipping comparison against the original numerical results. The full summaries require the same eight strategies, three seeds, five folds and clinical assessment regions as the study.

## Regional detection outcomes

| Reference region | Prediction | Outcome |
|---|---|---|
| Lesion present | Prediction overlaps the reference lesion | True positive |
| Lesion present | No lesion overlap, including a prediction elsewhere in the region | False negative |
| No lesion | A predicted lesion is present | False positive |
| No lesion | No predicted lesion | True negative |

Regional detection accuracy is the proportion of correct outcomes. A misplaced prediction in a positive region must not be counted as correct merely because the region has a positive prediction somewhere. The R analysis validates this overlap rule directly against reference, prediction and intersection pixel counts.

The full metric code is in the model workflow modules and in the independent standard-library implementation `analysis/reproduce_descriptive.py`. It calculates detection confusion counts, sensitivity, specificity, predictive values, accuracy, balanced accuracy, F1 and Matthews correlation coefficient.

## Segmentation and localisation

- **Dice-Positive:** mean Dice across all reference-positive regions, including zero overlap.
- **Dice-TP:** mean Dice conditional on lesion-overlapping detections. It is undefined when none occur.
- **Pixel micro-Dice:** the Dice calculation from pixel counts summed over the evaluated regions.
- **Localisation ROC area and average precision:** scores use the maximum predicted probability inside the known reference lesion on positive regions, and anywhere in the evaluated region on negatives. Their interpretation is localisation agreement, not conventional patient-level classification.

The primary operating threshold is 0.5. Secondary model-specific Youden thresholds are selected during development and retained for separate clinical assessment. The architecture-comparison calibration and aggregation are implemented separately in its own workflow.

## Recompute tables from saved regional predictions

```bash
python run.py reproduce
```

The included reference file has 88,560 regional records: eight strategies, 15 fits per strategy, 369 regions and two operating points. These are repeated model assessments of the same 369 regions, not independent new patients or lesions.

Main strategy summaries are calculated per fit, averaged over the five folds within each seed, and then summarised across the three seeds. Between-seed SD measures training variability. Undefined individual Dice-TP values are omitted from the within-seed mean and are not changed to zero; valid-model counts are written alongside the summaries.

Architecture-comparison development summaries pool the held-out fold predictions within each seed before calculating the seed-level metric. The saved benchmark reproduction uses this same procedure and retains the separate native/paired report domains.

## R statistics

`analysis/detection_accuracy.R` uses only the fixed-threshold regional correct/incorrect outcome. It fits one binomial multilevel logistic regression per architecture, preserves paired assessments and evaluates three contrasts with the architecture-specific clinical-only baseline. It checks convergence, singularity and the Hessian, writes the final random-effects structure, and compares the six odds ratios, confidence bounds and nominal P values with the supplied reference values.

```bash
Rscript analysis/install_packages.R
python run.py statistics
```

See [the analysis documentation](../analysis/README.md) for the full procedure, package versions and output dictionary.
