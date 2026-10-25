# Reproduce the reported clinical results

The files in this directory reproduce the main clinical detection, segmentation and statistical results from the saved regional predictions. This route needs no radiographs, trained weights or GPU. It checks the results of the saved models; use the repository's training and assessment workflows to fit models or generate predictions again.

Run the commands from the repository root.

## Quick start with Python

Python 3.10 or newer is sufficient. This script uses only the Python standard library.

```bash
python analysis/reproduce_descriptive.py --verify-reference
```

The default input is `data/reference/main_region_predictions.csv.gz`. Outputs are written to `results/descriptive/`:

| File | Contents |
|---|---|
| `clinical_detection.csv` | Sensitivity, specificity, predictive values, balanced accuracy and Matthews correlation coefficient at threshold 0.5 |
| `clinical_segmentation.csv` | Detection accuracy, F1, Dice-Positive, Dice-TP and pixel micro-Dice at threshold 0.5 |
| `clinical_localisation.csv` | Localisation ROC area and average precision, plus sensitivity and specificity at the saved Youden thresholds |
| `clinical_confusion_counts.csv` | Mean true-positive, false-negative, false-positive and true-negative regional counts |
| `model_metrics.csv` | Metrics for each fitted model and operating point |
| `seed_metrics.csv` | Fold averages within each seed |
| `descriptive.csv` | All metric means, sample SDs, valid-model counts and threshold ranges |
| `published_descriptive_comparison.csv` | Comparison with all 544 supplied reference means and SDs |
| `validation.json` | Coverage checks, input checksum and maximum discrepancy |

For a different destination:

```bash
python analysis/reproduce_descriptive.py --input data/reference/main_region_predictions.csv.gz --out results/descriptive --verify-reference
```

## Statistical analysis in R

The study used R 4.5.1. The exact analysis package versions are listed in `package_versions.csv`; the central packages are **lme4 2.0-6** and **jsonlite 2.0.0**.

Install the recorded versions once:

```bash
Rscript analysis/install_packages.R
```

The installer uses CRAN and its source archive where necessary. Source installation requires an R compilation toolchain: for example, the standard compiler dependencies on Linux or the matching Rtools version on Windows. Package binaries are not copied into this repository.

Run the six planned comparisons:

```bash
Rscript analysis/detection_accuracy.R
```

Or specify the input and output explicitly:

```bash
Rscript analysis/detection_accuracy.R data/reference/main_region_predictions.csv.gz results/statistics
```

The analysis uses the fixed-threshold rows only. It writes the six contrasts to `results/statistics/contrasts.csv`, together with fitted RDS models, descriptive accuracy values, convergence diagnostics, package versions and a comparison with the supplied reference results. Numerical optimisation can differ slightly across platforms; the reference comparison uses a tolerance of 0.0001 for odds ratios, confidence bounds and P values.

For predictions produced by a new complete training/assessment run, use its canonical regional output and explicitly disable comparison with the saved manuscript values:

```bash
python analysis/reproduce_descriptive.py --input runs/main/reports/regional_predictions.csv.gz --out results/new_descriptive
Rscript analysis/detection_accuracy.R runs/main/reports/regional_predictions.csv.gz results/new_statistics --no-reference-check
```

This keeps the same endpoint, model formulas, planned contrasts and convergence checks. It skips only the comparison with previously reported numerical values. The analysis requires the full eight-strategy, three-seed, five-fold assessment; an incomplete smoke run is not a substitute for that input.

If packages are already installed in a separate library, the optional `CARIES_R_LIB` environment variable can point to that directory. Normal R libraries work without setting this variable.

## What is compared

Each architecture has four strategies:

| Strategy key | Meaning |
|---|---|
| `clinical_only` | Training on clinical bitewings without laboratory pre-training or teacher guidance |
| `laboratory_pretraining` | Laboratory pre-training followed by clinical training |
| `knowledge_distillation` | Clinical training guided by the specified teacher |
| `pretraining_and_distillation` | Laboratory pre-training followed by clinical training with teacher guidance |

The teacher for nnU-Net was the laboratory-trained YOLO model. The teacher for YOLO was the clinical nnU-Net model. The exact training sequences and weight transfers belong to the training configurations; these short keys do not replace those definitions.

For each architecture, pre-training, distillation and their combination are compared **directly with that architecture's clinical-only baseline**. There are three contrasts per architecture and six in total. No between-architecture test or additional sensitivity, specificity or Dice hypothesis test is performed here.

## Outcome and dependence between predictions

The outcome is a correct regional detection assessment. In a reference-positive region, a correct assessment requires overlap between the predicted lesion and the reference lesion. A prediction elsewhere in that positive region counts as an error. In a reference-negative region, a correct assessment requires no predicted lesion.

All 369 regions are assessed by 15 fitted models per strategy: five folds and three seeds. The R analysis therefore uses 44,280 binary assessments from 120 fitted models. These are repeated assessments of 369 regions in 25 bitewings, **not 44,280 independent lesions or patients**.

One binomial multilevel logistic model is fitted per architecture. The strategy effect is referenced to clinical-only training. The initial random-intercept terms are bitewing, region, region-by-strategy, matched seed–fold run and fitted model. A variance component is removed only when estimated below `1e-8`; the same model is then refitted. The optimisation procedure, convergence checks and Hessian checks are preserved in the R script.

The reported odds ratios are conditional odds ratios with pointwise 95% Wald confidence intervals. P values are two-sided, nominal and exploratory, without multiplicity adjustment. Thresholds are not recalibrated by this analysis.

## Aggregation and segmentation

Descriptive metrics are first calculated for each fitted model, averaged across the five folds within each seed, then summarised by the mean and sample SD across the three seeds. Undefined individual metrics are omitted within the corresponding seed, never replaced with zero. The output records each metric's valid-model count.

Dice-Positive includes every reference-positive region, including missed lesions. Dice-TP includes only positive regions with lesion overlap. Pixel micro-Dice pools the pixel counts within the evaluated proximal-side regions for each fitted model. The script retains the original region pairing and native-grid pixel counts.

Localisation ROC area and average precision use the maximum prediction within the reference lesion on positive regions and the maximum prediction anywhere in the evaluated region on negative regions. They are **reference-conditioned localisation summaries**, not ordinary diagnostic-classification scores available for an unseen patient.

## Reference files and validation

The bundled input contains both fixed and saved model-specific Youden operating points. Both share the same anonymous region and bitewing keys. The input dictionary and provenance are in `data/reference/README.md` and `data/reference/provenance.json`.

The Python workflow independently recomputes the metrics from regional decisions and pixel counts. It verifies 369 regions per model, 184 positive and 185 negative regions, complete seed/fold coverage, consistent reference labels and lesion-overlap decisions. The reference means and SDs correspond to the manuscript's main clinical tables. The R workflow verifies the six published accuracy contrasts and flags invalid final fits.
