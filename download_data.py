import argparse
from src.dataset import prepare_and_cache_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess a subset of Gretel Synthetic Text-to-SQL")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to stream and cache")
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    print(f"Fetching {args.samples} samples...")
    train_path, val_path = prepare_and_cache_dataset(
        output_dir=args.out_dir,
        num_samples=args.samples,
        val_split=0.1
    )
    print(f"Data ready at:\n  Train: {train_path}\n  Val:   {val_path}")
