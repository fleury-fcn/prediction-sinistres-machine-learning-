# Data

## `insurance_claims.csv` (real, benchmark)

Public "insurance claims" dataset published by Derrick Mwiti
(<https://github.com/mwitiderrick/insurancedata>): 1,000 automobile claims, 39
columns, binary target `fraud_reported` (247 frauds, 24.7 %). Missing values are
encoded as `?`. Incidents span January-March 2015.

The provenance of the original records is not documented by the publisher; several
properties (uniform-looking marginals, two hobbies with > 70 % fraud) suggest the
data may itself be partly simulated. It is used here because it is the standard
public benchmark for this task, and all conclusions are limited accordingly.

## `synthetic/` (SYNTHETIC, generated)

Created by `make synth` from generators fitted on the 800-row real training split.
Only `SYNTHETIC_sample_1000.csv` is versioned; the 500,000-row parquet file is
regenerated locally (see `results/metrics/synthetic_dataset.json`). Synthetic rows
never enter the real-data benchmark.
