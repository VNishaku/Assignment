import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the configured A and B pipelines")
    parser.add_argument("--name", default="NishaKumari", help="Output filename prefix")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for company in ("A", "B"):
        subprocess.run(
            [
                sys.executable, "-B", str(root / "pipeline.py"),
                "--config", str(root / f"company_{company}.json"),
                "--output", str(root / f"{args.name}__{company}.csv"),
            ],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
