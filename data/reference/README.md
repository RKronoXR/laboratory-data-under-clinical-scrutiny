# Reference clinical predictions

`main_region_predictions.csv.gz` contains saved model assessments for the **eight main training strategies** evaluated on ACTA-Bw25-RefStd. It contains no radiograph pixels, lesion-mask images, patient metadata, source filenames or machine paths.

The reference dataset is described at [ACTA-Bw25-RefStd](https://doi.org/10.48338/VU01-UHGI3K). Obtain the images and reference masks from the dataset provider when reproducing image preparation, training or assessment. The compact file here is sufficient for checking the reported main clinical descriptive results and the six detection-accuracy statistical contrasts.

## Coverage

- 2 student architectures × 4 main strategies × 3 seeds × 5 folds = **120 fitted models**.
- Every fitted model assessed the same **369 proximal-side regions** in **25 bitewings**: 184 reference-positive and 185 reference-negative regions.
- Two saved operating points per model: fixed threshold 0.5 and a development-selected Youden threshold.
- **88,560 rows** in total; the R analysis uses the 44,280 fixed-threshold rows.

`region` and `bitewing` are anonymous evaluation-unit keys. They are stable across all architectures, strategies, seeds, folds and operating points. They do not identify individual patients. Replacing the original keys preserves their sorted grouping order, pairing and all numerical outcomes.

## Data dictionary

| Column | Definition |
|---|---|
| `architecture` | `nnU-Net` or `YOLO` |
| `strategy` | `clinical_only`, `laboratory_pretraining`, `knowledge_distillation` or `pretraining_and_distillation` |
| `seed`, `fold` | Training seed (17, 42, 2026) and validation-fold index (0–4) identifying the fitted model |
| `region`, `bitewing` | Stable anonymous region and image keys |
| `threshold_mode` | `fixed` or `youden`; thresholds are saved rather than reselected here |
| `threshold` | Actual probability cutoff for that fitted model and operating point |
| `reference_positive` | 1 if the reference lesion mask contains lesion pixels; otherwise 0 |
| `reference_pixels` | Number of reference-positive pixels in the evaluated region |
| `predicted_pixels` | Number of retained predicted-positive pixels in the region |
| `intersection_pixels` | Number of pixels shared by the prediction and reference lesion |
| `pixel_tp`, `pixel_fp`, `pixel_tn`, `pixel_fn` | Native-grid pixel confusion counts within the evaluated region |
| `score` | Maximum predicted lesion probability anywhere within the region |
| `overlap_score` | Maximum predicted probability within the reference lesion; empty for reference-negative regions |
| `dice_positive` | Region Dice for reference-positive regions; empty for negative regions |
| `detected` | 1 when prediction and reference lesion overlap; otherwise 0 |
| `correct_overlap` | Correct lesion-localising assessment: positive regions require overlap; negative regions require an empty prediction |

An empty value for `overlap_score` or `dice_positive` in a reference-negative region is intentional. It is not an imputed zero. A positive region with no overlapping detection has Dice 0 and an incorrect regional detection assessment.

## Expected results

| File | Purpose |
|---|---|
| `expected_main_descriptive.csv` | Means and sample SDs for 17 main metrics, eight strategies and both operating points |
| `expected_accuracy_descriptive.csv` | The eight fixed-threshold mean accuracy values and sample SDs |
| `expected_published_summary.csv` | All six accuracy odds ratios, confidence bounds, nominal P values and descriptive differences |
| `provenance.json` | Input checksum, coverage and anonymous per-fit source-record checksums |

The records were taken from the saved per-model regional assessments used for the manuscript. They were filtered to the main eight strategies and two reported operating points. Numeric counts, scores, decisions and thresholds were retained without rounding. Administrative metadata and local paths were removed. Training and benchmark source files are described separately in the repository's experiment documentation.

Use `python analysis/reproduce_descriptive.py --verify-reference` to verify the descriptive results and `Rscript analysis/detection_accuracy.R` to refit the statistical models. These reference files support result verification and do not substitute for the images, annotations, splits or trained weights needed for the image-based workflows.
