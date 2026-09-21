# Frozen results contract

The following files are the canonical outputs of v0.2.0:

- `results/baseline_v017/test_per_image.csv`
- `results/payload_freeze/validation_capacity.csv`
- `config/frozen_payloads.json`
- `results/models/validation_block_scores.csv`
- `results/models/allocator_alpha_validation.csv`
- `config/frozen_allocator.json`
- `results/frozen_test/per_image.csv`
- `results/detectors/test_detectability.csv`
- `results/detectors/adaptive_srm_lite_primary_payload.csv`
- `results/publication/table_test_metrics.csv`
- `results/publication/table_ablation.csv`
- `results/publication/experiment_audit.json`

After `results/frozen_test/per_image.csv` contains any test rows, changing a frozen payload level, allocator alpha, block size, source manifest, side-information accounting rule or codec behavior requires a new version rather than overwriting v0.2.0.
