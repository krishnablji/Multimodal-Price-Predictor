"""One-command pipeline runner for demo or full dataset training."""

import subprocess
import sys
from pathlib import Path

def main():
    train_path = "data/samples/demo_train.csv"
    test_path = "data/samples/demo_test.csv"
    
    # Check if raw data exists
    if Path("data/raw/train.csv").exists():
        train_path = "data/raw/train.csv"
        print("💡 Using dataset from data/raw/train.csv")
    if Path("data/raw/test.csv").exists():
        test_path = "data/raw/test.csv"
        print("💡 Using dataset from data/raw/test.csv")

    cmd = [
        sys.executable, "-m", "src.price_predictor.cli", "run",
        "--train-csv", train_path,
        "--test-csv", test_path,
        "--config", "configs/final.json",
        "--artifact-dir", "artifacts/final"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
