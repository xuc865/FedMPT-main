# ZSL / GZSL label splits (optional)

Place filtered JSON annotations here **only** when running with `--zsl zsl` or `--zsl gzsl` on COCO or NUS-WIDE.

Expected files (from the FedTPG release):

- `train_48_filtered.json`, `test_17_filtered.json`, `test_65_filtered.json` (COCO)
- `train_81_filtered.json`, `test_81_filtered.json`, `test_1006_filtered.json` (NUS-WIDE)

Standard federated MLR experiments (paper Tables 1–2) use the default loaders and do **not** need this folder.
