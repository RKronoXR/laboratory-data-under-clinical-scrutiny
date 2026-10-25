# Working with this source release

Keep the original numerical reference data unchanged when modifying training code. New experiments should use separate output folders and retain their annotation version, input checksums, model-selection records and assessment thresholds.

Before proposing a source change, run the relevant workflow and the CPU tests:

```bash
python -m unittest discover -s tests -v
```

The release integrity manifest records the exact source, configuration, annotation supplement and reference-data files shipped together. After an intentional edit, rebuild it and rerun the checks:

```bash
python tools/update_integrity.py
python run.py verify
python -m unittest discover -s tests -v
```

A successful checksum check establishes which files were used. It does not establish that a changed training method reproduces the original scientific results. Report new model results separately and retain the stated detection definitions and aggregation procedure.

Keep original dataset attribution, the study attribution and applicable licence notices with reused material. The Python and R result-reproduction commands are independent of GPU training and should remain runnable without images.
