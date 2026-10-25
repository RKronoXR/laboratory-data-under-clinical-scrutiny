# Data preparation

Download the radiographs and original reference annotations from the three original data repositories listed in [Data sources](data_sources.md). Put the downloaded archives in `data/downloads/`. Keep their original files intact; nested folder names do not need to match this repository.

The study's additional Refined DCBR annotations are supplied in `data/study_annotations.zip`. This bundle contains lesion masks, anatomical region coordinates and region-review decisions, without radiographs. Its annotation licence and credits are separate from the code licence.

Assemble the current annotation release:

```bash
python run.py datasets assemble
python run.py datasets inspect data/Datasets.zip
```

The assembler searches the downloaded ZIPs and the study annotation bundle by file checksum and writes a canonical `data/Datasets.zip`. It copies file bytes without rescaling, re-encoding or editing images. Missing files are reported by dataset, relative name and expected checksum. An incomplete archive is never accepted as a completed assembly.

To inspect an already assembled study archive, use the second command directly. Folder names are recognised by their internal structure, so names such as `ACTA-DIRECT v2` and `Refined DCBR` are accepted.

## Annotation versions

The included Refined DCBR masks are an updated annotation release. The radiographs, anatomical region coordinates, training folds and clinical assessment reference are unchanged. New training with the updated masks can give results different from those in the manuscript; the magnitude of that difference has not been measured. The saved regional predictions under `data/reference/` retain the original model results and reproduce the reported tables and statistics.

Two explicit input modes are available:

| Mode | Input inventory | Purpose |
|---|---|---|
| `recorded` | Original image, mask and coordinate checksums | Repeat the original experiment with its original annotations |
| `updated` | Updated Refined DCBR mask checksums; otherwise the same scientific inputs | Train and assess using the included annotation revision |

The model workflows default to `recorded`. Use `--annotation-version updated` explicitly with the current mask bundle. The assembler defaults to the updated release. Do not reuse an output directory between annotation versions: model and threshold identities include the input version.

The updated workflow records annotation-dependent label changes and any very small native lesions that disappear on the unchanged training grid. It keeps the original image dimensions, resampling method, anatomical regions and split membership. Evaluation still uses the original-resolution masks. The data audit is written with the prepared run.

## Data files and generated folders

- `data/downloads/`: locally downloaded source datasets; ignored by Git.
- `data/study_annotations.zip`: the study's additional annotation files, without radiographs.
- `configs/main/data_fingerprints.json`: original scientific input inventory.
- `configs/main/data_fingerprints_updated.json`: updated annotation inventory.
- `data/Datasets.zip`: locally assembled canonical input archive; ignored by Git.
- `runs/main/` and `runs/benchmark/`: independently prepared data, models, audit files and assessments; ignored by Git.

The exact native nnU-Net plans and clinical split assignments are included under `configs/`. Preparation reuses these plans and splits rather than running a new experiment planner.
