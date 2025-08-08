from dataclasses import dataclass
import safetensors.torch
import torch
import argparse
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass
class DareConfig:
    drop_rate: float = 0.9
    delta_threshold: float = 0.002
    verbose: bool = True


class DareMerger:
    def __init__(
        self, base_file: str, merge_files: list[str], config: DareConfig = None
    ):
        """
        Memory-efficient DARE model merger

        Args:
        - base_file: SafeTensors file of base model
        - merge_files: List of SafeTensors files to merge
        - config: DareConfig instance
        """
        self.base_file = base_file
        self.merge_files = merge_files
        self.config = config or DareConfig()

    def get_all_keys(self) -> set[str]:
        """Get all available keys from base model"""
        with safetensors.safe_open(self.base_file, framework="pt") as f:
            return set(f.keys())

    def load_weight(self, file_path: str, key: str) -> torch.Tensor:
        """Load single weight tensor from file"""
        with safetensors.safe_open(file_path, framework="pt") as f:
            return f.get_tensor(key)

    def compute_merged_delta(self, base_weight: torch.Tensor, key: str) -> torch.Tensor:
        """
        Compute aggregated delta for a specific key

        Args:
        - base_weight: Base model weight tensor
        - key: Weight key being processed

        Returns: Average delta across all merge files
        """
        deltas = []

        for merge_file in self.merge_files:
            try:
                merge_weight = self.load_weight(merge_file, key)
                delta = merge_weight - base_weight
                deltas.append(delta)
                del merge_weight
            except Exception as e:
                if self.config.verbose:
                    print(f"Skipping {key} in {merge_file}: {e}")

        if not deltas:
            return torch.zeros_like(base_weight)

        return torch.mean(torch.stack(deltas), dim=0)

    def apply_dare_transform(
        self, base_weight: torch.Tensor, delta_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply DARE transformation: drop parameters and rescale

        Args:
        - base_weight: Original base model weight
        - delta_weight: Computed delta weight

        Returns: Transformed weight
        """
        # Check delta magnitude
        max_delta = delta_weight.abs().max().item()
        if max_delta > self.config.delta_threshold and self.config.verbose:
            print(
                f"Warning: Large delta {max_delta:.6f} > {self.config.delta_threshold}"
            )

        # Create random drop mask
        mask = torch.rand_like(delta_weight) < self.config.drop_rate

        # Drop parameters and rescale remaining
        dropped_delta = delta_weight.clone()
        dropped_delta[mask] = 0
        rescaled_delta = dropped_delta / (1 - self.config.drop_rate)

        return base_weight + rescaled_delta

    def merge(self, output_file: str):
        """
        Execute memory-efficient DARE merge

        Args:
        - output_file: Path to save merged SafeTensors file
        """
        all_keys = self.get_all_keys()
        merged_weights = {}

        for i, key in enumerate(all_keys):
            # Load base weight for this key
            base_weight = self.load_weight(self.base_file, key)

            # Compute merged delta across all files
            delta_weight = self.compute_merged_delta(base_weight, key)

            # Apply DARE transformation
            merged_weight = self.apply_dare_transform(base_weight, delta_weight)

            del base_weight
            del delta_weight

            merged_weights[key] = merged_weight

            # Progress logging
            if self.config.verbose and (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(all_keys)} keys")

        # Save merged model
        safetensors.torch.save_file(merged_weights, output_file)

        if self.config.verbose:
            print(f"Merge complete! Saved to {output_file}")

    def analyze(self, sample_keys: int = 50):
        """
        Analyze model differences without performing merge (dry run)

        Args:
        - sample_keys: Number of keys to sample for analysis
        """
        all_keys = list(self.get_all_keys())
        total_keys = len(all_keys)

        if self.config.verbose:
            print(f"Model Analysis (Dry Run)")
            print(f"=" * 50)
            print(f"Total keys: {total_keys}")
            print(f"Analyzing sample of {min(sample_keys, total_keys)} keys...")
            print()

        # Sample keys for analysis
        import random

        sample_keys_list = random.sample(all_keys, min(sample_keys, total_keys))

        delta_stats = {
            "min_delta": float("inf"),
            "max_delta": 0.0,
            "avg_delta": 0.0,
            "large_deltas": 0,
            "total_params": 0,
            "key_stats": [],
        }

        for i, key in enumerate(sample_keys_list):
            # Load base weight
            base_weight = self.load_weight(self.base_file, key)

            # Compute delta
            delta_weight = self.compute_merged_delta(base_weight, key)

            # Calculate statistics
            max_delta = delta_weight.abs().max().item()
            mean_delta = delta_weight.abs().mean().item()
            param_count = delta_weight.numel()

            delta_stats["min_delta"] = min(delta_stats["min_delta"], max_delta)
            delta_stats["max_delta"] = max(delta_stats["max_delta"], max_delta)
            delta_stats["avg_delta"] += mean_delta
            delta_stats["total_params"] += param_count

            if max_delta > self.config.delta_threshold:
                delta_stats["large_deltas"] += 1

            delta_stats["key_stats"].append(
                {
                    "key": key,
                    "shape": list(base_weight.shape),
                    "params": param_count,
                    "max_delta": max_delta,
                    "mean_delta": mean_delta,
                }
            )

            if self.config.verbose and (i + 1) % 10 == 0:
                print(f"Analyzed {i + 1}/{len(sample_keys_list)} keys")

        # Calculate final statistics
        delta_stats["avg_delta"] /= len(sample_keys_list)

        # Display results
        if self.config.verbose:
            print(f"\nAnalysis Results:")
            print(f"=" * 50)
            print(f"Sample size: {len(sample_keys_list)}/{total_keys} keys")
            print(f"Total parameters analyzed: {delta_stats['total_params']:,}")
            print(
                f"Delta range: {delta_stats['min_delta']:.6f} to {delta_stats['max_delta']:.6f}"
            )
            print(f"Average delta: {delta_stats['avg_delta']:.6f}")
            print(
                f"Keys with large deltas: {delta_stats['large_deltas']}/{len(sample_keys_list)}"
            )
            print(
                f"Large delta ratio: {delta_stats['large_deltas'] / len(sample_keys_list) * 100:.1f}%"
            )
            print()

            if delta_stats["max_delta"] > self.config.delta_threshold:
                print(
                    f"⚠️  Warning: Large deltas detected (>{self.config.delta_threshold})"
                )
                print(f"   Consider adjusting --delta-threshold or --drop-rate")
                print(f"   Current drop rate: {self.config.drop_rate}")
                print()

            # Show top 5 layers with largest deltas
            sorted_stats = sorted(
                delta_stats["key_stats"], key=lambda x: x["max_delta"], reverse=True
            )
            print(f"Top 5 layers with largest deltas:")
            print(f"{'-' * 80}")
            print(f"{'Key':<40} {'Shape':<20} {'Max Delta':<12} {'Mean Delta':<12}")
            print(f"{'-' * 80}")
            for stat in sorted_stats[:5]:
                shape_str = f"{stat['shape']}"[:18]
                print(
                    f"{stat['key'][:38]:<40} {shape_str:<20} {stat['max_delta']:<12.6f} {stat['mean_delta']:<12.6f}"
                )

        return delta_stats


def load_config_from_toml(config_path: str) -> tuple[DareConfig, str, list[str], str]:
    """
    Load configuration from TOML file

    Returns:
    - DareConfig instance
    - base_file path
    - list of merge_files paths
    - output_file path
    """
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)

    # Load DARE config
    dare_config = config_data.get("dare", {})
    config = DareConfig(
        drop_rate=dare_config.get("drop_rate", 0.9),
        delta_threshold=dare_config.get("delta_threshold", 0.002),
        verbose=dare_config.get("verbose", True),
    )

    # Load model paths
    models = config_data.get("models", {})
    base_file = models.get("base_file", "")
    merge_files = models.get("merge_files", [])
    output_file = models.get("output_file", "merged_model.safetensors")

    if not base_file:
        raise ValueError("base_file must be specified in TOML config")
    if not merge_files:
        raise ValueError("merge_files must be specified in TOML config")

    return config, base_file, merge_files, output_file


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="DARE Model Merger - Merge language models using Drop And REscale method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use TOML config file
  python main.py --config merge_config.toml
  
  # Use command line arguments
  python main.py --base base.safetensors --merge math.safetensors code.safetensors --output merged.safetensors
  
  # Override TOML config with CLI args
  python main.py --config merge_config.toml --drop-rate 0.95 --verbose
        """,
    )

    # Config file option
    parser.add_argument(
        "--config", "-c", type=str, help="Path to TOML configuration file"
    )

    # Model file options
    parser.add_argument(
        "--base", "-b", type=str, help="Path to base model SafeTensors file"
    )

    parser.add_argument(
        "--merge",
        "-m",
        nargs="+",
        type=str,
        help="Paths to model SafeTensors files to merge",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="merged_model.safetensors",
        help="Output path for merged model (default: merged_model.safetensors)",
    )

    # DARE configuration options
    parser.add_argument(
        "--drop-rate",
        type=float,
        help="Proportion of delta parameters to drop (0.0-1.0)",
    )

    parser.add_argument(
        "--delta-threshold",
        type=float,
        help="Warning threshold for large delta parameters",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Disable verbose output"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze models without performing merge (shows delta statistics)",
    )

    return parser


def main():
    parser = create_cli_parser()
    args = parser.parse_args()

    # Load from config file if provided
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Config file {args.config} does not exist")
            return 1

        try:
            config, base_file, merge_files, output_file = load_config_from_toml(
                args.config
            )
        except Exception as e:
            print(f"Error loading config file: {e}")
            return 1
    else:
        # Use CLI arguments
        if not args.base or not args.merge:
            print("Error: --base and --merge are required when not using --config")
            parser.print_help()
            return 1

        config = DareConfig()
        base_file = args.base
        merge_files = args.merge
        output_file = args.output

    # Override config with CLI arguments if provided
    if args.drop_rate is not None:
        config.drop_rate = args.drop_rate
    if args.delta_threshold is not None:
        config.delta_threshold = args.delta_threshold
    if args.verbose:
        config.verbose = True
    if args.quiet:
        config.verbose = False

    # Validate files exist
    if not Path(base_file).exists():
        print(f"Error: Base file {base_file} does not exist")
        return 1

    for merge_file in merge_files:
        if not Path(merge_file).exists():
            print(f"Error: Merge file {merge_file} does not exist")
            return 1

    # Create and run merger
    if config.verbose:
        print(f"DARE Configuration:")
        print(f"  Drop rate: {config.drop_rate}")
        print(f"  Delta threshold: {config.delta_threshold}")
        print(f"  Base file: {base_file}")
        print(f"  Merge files: {merge_files}")
        print(f"  Output file: {output_file}")
        print()

    merger = DareMerger(base_file=base_file, merge_files=merge_files, config=config)

    try:
        if args.dry_run:
            # Perform analysis instead of merge
            stats = merger.analyze(sample_keys=50)
            if config.verbose:
                print("\n🔍 Dry run complete! No files were modified.")
            return 0
        else:
            # Perform actual merge
            merger.merge(output_file)
            return 0
    except Exception as e:
        print(f"Error during {'analysis' if args.dry_run else 'merge'}: {e}")
        return 1


if __name__ == "__main__":
    main()
