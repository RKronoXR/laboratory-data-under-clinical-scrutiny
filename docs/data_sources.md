# Data sources and exact input assembly

Radiographs are obtained from their dataset providers. The code repository does not redistribute them. The authored annotation complement is supplied separately in [`data/study_annotations.zip`](../data/study_annotations.zip).

## Source datasets

| Dataset | Source record | Files used by the study workflows |
|---|---|---|
| ACTA-DIRECT v2 | [VU YODA dataset record](https://doi.org/10.48338/VU01-H5ALYJ) | 2,270 proximal-side radiographs and their matching lesion masks from the selected annotation set |
| DCBR proximal caries lesions dataset | [Mendeley Data version 1](https://doi.org/10.17632/4fbdxs7s7w.1) | 100 clinical bitewing radiographs, combined with the study's refined lesion masks and proximal-side region coordinates |
| ACTA-Bw25-RefStd | [VU YODA dataset record](https://doi.org/10.48338/VU01-UHGI3K) | 25 clinical bitewings, full-image reference lesion masks and the coordinates of 369 proximal-side regions |

The original DCBR lesion boxes do not substitute for the study's refined segmentation masks. These masks and the anatomical region coordinates are explicit input data. They are not inferred again from a model during data assembly.

ACTA-DIRECT v2 contains several annotation families. The assembler selects files by their required SHA256 values rather than choosing a family by its folder name. Keep the original downloaded files intact and retain the source dataset's attribution and licence information.

## Included authored annotations

`data/study_annotations.zip` contains the intentionally updated Refined DCBR annotation release, identified as **`refined-dcbr-updated`**:

- 100 full-image binary lesion masks.
- `Proximal_Side_Regions.csv`: 1,322 anatomical proximal-side regions.
- `Surface_Review_085.csv`: the 14 explicitly reviewed regions of bitewing 085, including the target-label decisions needed when neighbouring lesion pixels enter another region's box.
- An annotation README, full CC BY 4.0 licence and author attribution.

The bundle contains no radiographs. Its authored masks and coordinates are licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The code's AGPL 3.0 licence is separate. Credit Ricardo Eugenio Gonzalez Valenzuela, Swasti Mishra, Pascal Mettes and Erwin Berkhout, and cite the original DCBR dataset.

The data reader consumes the review CSV together with the full-image mask and ROI coordinates. An arbitrary cropper that assigns a positive label whenever any lesion pixel enters a rectangle does not implement these reviewed target labels.

## Assemble the updated input archive

Put the downloaded ZIP files or extracted dataset directories inside `data/downloads/`. Their outer folder names may differ from the names used by the original project. Then run from the repository root:

```bash
python run.py datasets assemble --inputs data/downloads --out data/Datasets.zip --annotation-version updated
```

The repository command also includes the supplied annotation-only bundle. It matches required files across all inputs by SHA256 and creates a canonical archive. Radiographs and masks are copied byte-for-byte; there is no pixel decoding, scaling or re-encoding during assembly. The image-processing steps used by the experiments run later during preparation.

To prepare the main experiments with this updated annotation release:

```bash
python run.py main prepare --annotation-version updated --data-zip data/Datasets.zip
```

The fingerprint profile is [`configs/main/data_fingerprints_updated.json`](../configs/main/data_fingerprints_updated.json). It specifies 4,793 required image, mask and CSV files across the three datasets. The assembled archive has these canonical folders:

```text
Datasets/
|-- ACTA-DIRECT dataset v2/
|-- DCBR proximal caries lesions dataset/
`-- ACTA-Bw25-RefStd/
```

If exact required files are missing, assembly stops and reports their filenames and SHA256 values. An existing destination is left unchanged and no partial dataset archive is committed. Unsafe ZIP paths, symlinks, encrypted members and path collisions are rejected.

## Recorded and updated annotations have different purposes

[`configs/main/data_fingerprints.json`](../configs/main/data_fingerprints.json) identifies the recorded input files associated with the study's saved results. The supplied updated annotation bundle has its own fingerprint profile and is not declared byte-identical to that recorded input.

Use `--annotation-version updated` explicitly to train and assess experiments with the updated release. Their performance must be calculated from those runs; it is not assumed to equal the saved manuscript values. The pipelines retain the frozen image grouping and training/assessment separation rather than silently creating a different split.

Use the included [reference regional predictions](../data/reference/README.md) to reproduce the reported main clinical tables and statistical comparisons without images or model training:

```bash
python analysis/reproduce_descriptive.py --verify-reference
Rscript analysis/detection_accuracy.R
```

These two routes are complementary: the reference records verify the saved results, while image-based training and assessment measure a complete run with the explicitly selected annotation version.

## Utility API

The data-discovery and assembly module uses only Python's standard library:

```python
from carieslab.datasets import detect_raw_names, inspect_archive, assemble_archives

names = detect_raw_names("data/Datasets.zip")
inventory = inspect_archive("data/Datasets.zip", include_members=False)
report = assemble_archives(
    ["data/downloads", "data/study_annotations.zip"],
    "data/Datasets.zip",
    fingerprints="configs/main/data_fingerprints_updated.json",
)
```

`detect_raw_names` returns the actual dataset folder components under keys `acta_direct`, `dcbr` and `bw25`. It recognises internal layouts and known folder aliases, including wrapper directories, and rejects ambiguous copies. `inspect_archive` reports file membership and CSV row counts without extracting image pixels. `assemble_archives` accepts downloaded ZIPs, loose files or directories; byte-identical duplicate sources are allowed. If files are absent, `MissingDatasetFilesError.missing_files` contains the complete missing-file inventory.
