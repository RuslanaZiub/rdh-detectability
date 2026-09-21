# Detectability-Aware Reversible Data Hiding

Reproducibility repository for the manuscript:

**Detectability-Aware Reversible Data Hiding with Steganalyzer-Guided Local Risk Allocation**

## Scope

The repository contains the source code, frozen experimental configuration,
analysis notebooks, tests, protocol documentation, and compact publication-level
result summaries used in the manuscript.

The main frozen operating point is 0.009 net bpp with allocation weight
alpha = 0.25. The repository also contains the secondary SRNet robustness
workflow and post-hoc reviewer-facing sensitivity diagnostics.

## Data

The image dataset itself is **not redistributed in this repository**.
Obtain BOSSbase / the source dataset separately under its original terms and
prepare it using the project preprocessing workflow.

Do not commit private local paths, API keys, Docker virtual disks, caches,
per-image context caches, or large training checkpoints.

## Reproducibility

Recommended order:

1. Create the Python/Docker environment.
2. Prepare the dataset using the configured split.
3. Run the training/development notebooks.
4. Freeze the allocator and detector artifacts.
5. Run the frozen TEST analysis.
6. Run the secondary robustness and post-hoc diagnostics only with their
   documented status and restrictions.

See docs/ and the notebooks for the exact frozen protocol.

## Statistical interpretation

The primary TEST contrast is confirmatory under the frozen protocol.
SRNet is a post-hoc secondary robustness detector.
The alpha sweep, complexity baseline, and reviewer-facing diagnostics are
explicitly exploratory/post-hoc and are not used to retune the frozen allocator.

## Large model files

Large checkpoints are intentionally excluded from normal Git history.
If archival model weights are released, place them in a GitHub Release and/or
the associated Zenodo record rather than committing them to the repository.

## Citation

A versioned archival DOI should be added here after the GitHub release is
archived in Zenodo.

## License

No software license is granted unless a LICENSE file is added by the authors.
