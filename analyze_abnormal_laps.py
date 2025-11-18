#!/usr/bin/env python3
"""Analyze abnormal laps across train/valid/test splits."""

from pathlib import Path

import pandas as pd


def is_abnormal_lap(lap_file):
    """Check if a lap is abnormal (pit lap, red flag, etc.).

    Args:
        lap_file: Path to lap CSV file

    Returns:
        Tuple of (is_abnormal, stats_dict)
    """
    df = pd.read_csv(lap_file)

    # Criteria for abnormal lap:
    # 1. Duration > 300 seconds (5 minutes) - normal VIR lap is ~2-3 minutes
    # 2. Average speed < 60 km/h - too slow for racing
    # 3. Ends with very low speed < 20 km/h (pit entry/stop)
    # 4. More than 50% of time at < 60 km/h

    duration_sec = len(df) / 10.0  # 10Hz sampling
    avg_speed = df["speed"].mean()
    end_speed = df["speed"].iloc[-50:].mean()
    low_speed_pct = (df["speed"] < 60).sum() / len(df) * 100

    is_abnormal = duration_sec > 300 or avg_speed < 60 or end_speed < 20 or low_speed_pct > 50

    return is_abnormal, {
        "duration": duration_sec,
        "avg_speed": avg_speed,
        "end_speed": end_speed,
        "low_speed_pct": low_speed_pct,
    }


def main():
    """Analyze abnormal laps across train/valid/test splits."""
    # Scan all splits
    splits = ["train", "valid", "test"]
    results = {}

    for split in splits:
        print(f"\n=== Analyzing {split.upper()} set ===")

        # Get all lap files (from preprocessed source)
        split_dir = Path(f"datasets/drivingdatasets/input/{split}")
        x_files = list((split_dir / f"x_{split}").glob("*.csv"))

        abnormal_count = 0
        total_count = len(x_files)
        abnormal_laps = []

        for x_file in x_files:
            # Get corresponding preprocessed lap file
            # Extract lap info from filename
            # Format: VIR_R1_vehicle_GR86_002_2_lap_001_X_train.csv
            parts = x_file.stem.split("_")

            # Find original lap file
            race_parts = []
            for i, part in enumerate(parts):
                if part == "vehicle":
                    race_parts = parts[:i]
                    break

            if not race_parts:
                continue

            race_str = "_".join(race_parts)  # VIR_R1
            vehicle_idx = parts.index("vehicle")
            lap_idx = parts.index("lap")
            vehicle_str = "_".join(parts[vehicle_idx:lap_idx])  # vehicle_GR86_002_2
            lap_str = "_".join(parts[lap_idx : lap_idx + 2])  # lap_001

            # Find in preprocessed directory
            race_dir = f"datasets/preprocessed_10Hz/{race_parts[0]}/{race_parts[1]}"
            lap_pattern = f"{race_str}_{vehicle_str}_{lap_str}.csv"
            lap_files = list(Path(race_dir).glob(lap_pattern))

            if not lap_files:
                continue

            lap_file = lap_files[0]

            try:
                is_abn, stats = is_abnormal_lap(lap_file)

                if is_abn:
                    abnormal_count += 1
                    abnormal_laps.append({"file": lap_file.name, **stats})
            except Exception as e:
                print(f"Error processing {lap_file.name}: {e}")
                continue

        abnormal_pct = abnormal_count / total_count * 100 if total_count > 0 else 0
        print(f"Total laps: {total_count}")
        print(f"Abnormal laps: {abnormal_count} ({abnormal_pct:.1f}%)")

        if abnormal_count > 0:
            print("\nMost extreme abnormal laps:")
            # Sort by duration (longest first)
            sorted_laps = sorted(abnormal_laps, key=lambda x: x["duration"], reverse=True)
            for i, lap in enumerate(sorted_laps[:5]):
                print(f"  {i + 1}. {lap['file']}")
                print(
                    f"     Duration: {lap['duration']:.1f}s, "
                    f"Avg Speed: {lap['avg_speed']:.1f} km/h, "
                    f"End Speed: {lap['end_speed']:.1f} km/h, "
                    f"Low Speed: {lap['low_speed_pct']:.1f}%"
                )

        results[split] = {
            "total": total_count,
            "abnormal": abnormal_count,
            "abnormal_pct": abnormal_pct,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Abnormal Lap Distribution Across Splits")
    print("=" * 70)
    for split in splits:
        r = results[split]
        print(
            f"{split.capitalize():10s}: {r['abnormal']:3d}/{r['total']:3d} "
            f"({r['abnormal_pct']:5.1f}%) abnormal"
        )

    print("\n⚠️  Analysis:")
    if results["test"]["abnormal_pct"] > results["train"]["abnormal_pct"] * 1.5:
        print("   ❌ Test set has significantly MORE abnormal laps than train set!")
        print("   → This explains the performance drop")
        print("   → Model trained on normal laps, tested on many abnormal laps")
    elif results["test"]["abnormal_pct"] < results["train"]["abnormal_pct"] * 0.5:
        print("   ✓ Test set has FEWER abnormal laps (good)")
    else:
        print("   ~ Abnormal lap distribution is relatively balanced")

    print("\n💡 Recommendations:")
    total_abnormal = sum(r["abnormal"] for r in results.values())
    if total_abnormal > 10:
        print("   1. FILTER out abnormal laps before training")
        print('   2. Or create separate "abnormal lap detector" model')
        print("   3. Use only clean racing laps for driver identification")


if __name__ == "__main__":
    main()
