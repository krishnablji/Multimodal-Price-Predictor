"""Download e-commerce product dataset from Kaggle."""

import argparse
import subprocess
import sys
from pathlib import Path


def download_dataset(dataset_handle: str = "suvroo/amazon-ml", dest_dir: str = "data/raw", unzip: bool = True):
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Downloading dataset '{dataset_handle}' to '{dest_path}'...")
    cmd = ["kaggle", "datasets", "download", "-d", dataset_handle, "-p", str(dest_path)]
    if unzip:
        cmd.append("--unzip")

    try:
        subprocess.run(cmd, check=True)
        print("✅ Dataset downloaded and extracted successfully!")
    except FileNotFoundError:
        print("❌ Error: 'kaggle' CLI is not installed or not in PATH.")
        print("👉 Install it with: pip install kaggle")
        print("👉 Ensure your ~/.kaggle/kaggle.json API token is in place.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download Multimodal E-Commerce Dataset")
    parser.add_argument("--dataset", default="suvroo/amazon-ml", help="Dataset identifier on Kaggle")
    parser.add_argument("--dest", default="data/raw", help="Target directory for raw data")
    parser.add_argument("--no-unzip", action="store_true", help="Do not automatically unzip archive")
    args = parser.parse_args()

    download_dataset(dataset_handle=args.dataset, dest_dir=args.dest, unzip=not args.no_unzip)


if __name__ == "__main__":
    main()
