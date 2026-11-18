# Cantill.ai spend pipeline

This local tool ingests one or more CSV sources described by JSON, normalizes net spend and dates, applies configured exclusions and duplicate rules, groups suppliers/items, and classifies included lines against the runtime taxonomy.

## Setup and run

Python 3.10+ and `numpy`, `scikit-learn` are required. Install with:

```powershell
python -m pip install numpy scikit-learn
```

From any directory, run both supplied companies with one command:

```powershell
python run_all.py
```

Pass `--name YourName` to change the output filename prefix. To run one company or a later company config, use `python pipeline.py --config path\to\company.json --output path\to\output.csv`.

For the explicit zero-label check on A, run:

```powershell
python pipeline.py --config company_A.json --output A_zero_label.csv --no-labels
```

This uses only taxonomy names and is intended as a cold-start fallback, not as
a substitute for labelled training.

The CSV inputs are intentionally ignored by Git. Keep the supplied data locally and put later blind-test files in the paths named by their configs. Make a new JSON config for a different source layout and point it to that company's tree.

Labels are optional. If a config provides a labels file, the classifier uses character n-gram nearest-neighbour matching against those labelled descriptions. Without labels, it selects the closest category path based only on the category names in the supplied tree.


