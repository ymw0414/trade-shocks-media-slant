"""
99_test_slant_convergence_yearly.py

[Description]
This script performs a Monte Carlo simulation to test how the weighted slant score
converges to the "True YEARLY Mean" as the sample size (N) increases.

[Logic]
1. Identifies the Top 3 newspapers with the highest total volume.
2. For each newspaper, selects the SINGLE YEAR with the maximum article count
   (e.g., "The NYT in 1996").
3. Treats that specific year's data as the population and draws random samples
   of size N (N = 10, 30, ..., 2000).
4. Plots the error bars to determine the minimum article count required per year.

[Output]
- Saves a plot to: data/analysis/plots/slant_convergence_yearly_test.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "slant"
PLOT_DIR = BASE_DIR / "data" / "analysis" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Sample sizes to test
SAMPLE_SIZES = [10, 30, 50, 100, 200, 300, 500, 800, 1000, 1500, 2000]
TRIALS_PER_SIZE = 50

def get_slant_files():
    return sorted(list(SLANT_DIR.glob("news_slant_congress_*.parquet")))

def calculate_weighted_slant(df_sample):
    if df_sample.empty:
        return np.nan
    weighted_sum = (df_sample['slant'] * df_sample['used_terms']).sum()
    total_weights = df_sample['used_terms'].sum()
    if total_weights == 0:
        return np.nan
    return weighted_sum / total_weights

def main():
    files = get_slant_files()
    if not files:
        print("Error: No slant files found.")
        return

    # 1. Identify Top 3 Newspapers
    print(">>> 1. Identifying Top 3 high-volume newspapers...")
    paper_counts = {}
    for f in tqdm(files, desc="Scanning Metadata"):
        try:
            # Read paper to find top targets
            df = pd.read_parquet(f, columns=['paper'])
            counts = df['paper'].value_counts()
            for p, c in counts.items():
                paper_counts[p] = paper_counts.get(p, 0) + c
        except:
            pass

    top_papers = sorted(paper_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    target_papers = [p[0] for p in top_papers]
    print(f"   Selected Targets: {target_papers}")

    # 2. Load ALL data for these targets first
    print(">>> 2. Loading article data for targets...")
    article_pool = {p: [] for p in target_papers}

    for f in tqdm(files, desc="Loading Articles"):
        try:
            # Load date to derive year
            df = pd.read_parquet(f, columns=['paper', 'slant', 'used_terms', 'date'])
            df_targets = df[df['paper'].isin(target_papers)].copy()

            if not df_targets.empty:
                # Ensure Year exists
                df_targets['date'] = pd.to_datetime(df_targets['date'], errors='coerce')
                df_targets['year'] = df_targets['date'].dt.year

                for p in target_papers:
                    subset = df_targets[df_targets['paper'] == p]
                    if not subset.empty:
                        article_pool[p].append(subset)
        except:
            pass

    # 3. Select PEAK YEAR and Run Simulation
    print("\n>>> 3. Selecting Peak Year & Running Simulation...")

    plt.figure(figsize=(15, 5))

    for i, paper in enumerate(target_papers):
        # Merge all chunks
        if not article_pool[paper]:
            print(f"Skipping {paper} (No data)")
            continue

        df_all_years = pd.concat(article_pool[paper], ignore_index=True)

        # [CRITICAL FIX] Find the year with the most articles
        year_counts = df_all_years['year'].value_counts()
        if year_counts.empty:
            print(f"Skipping {paper} (No valid years)")
            continue

        peak_year = year_counts.idxmax()
        peak_count = year_counts.max()

        # Filter for ONLY that year
        df_year = df_all_years[df_all_years['year'] == peak_year].copy()

        # Calculate True Mean for that year
        true_slant = calculate_weighted_slant(df_year)

        print(f"\n[{paper}]")
        print(f"   Peak Year: {int(peak_year)} (N={peak_count} articles)")
        print(f"   True Slant (Yearly): {true_slant:.4f}")

        means = []
        errors = []

        valid_sample_sizes = [n for n in SAMPLE_SIZES if n < peak_count]

        for n in valid_sample_sizes:
            trial_slants = []
            for _ in range(TRIALS_PER_SIZE):
                sample = df_year.sample(n=n, replace=False)
                slant = calculate_weighted_slant(sample)
                trial_slants.append(slant)

            means.append(np.mean(trial_slants))
            errors.append(np.std(trial_slants))

        # Plot
        ax = plt.subplot(1, 3, i+1)
        ax.errorbar(valid_sample_sizes, means, yerr=errors, fmt='-o', capsize=5,
                    label='Est. (Mean ± SD)', color='blue', ecolor='gray')
        ax.axhline(y=true_slant, color='r', linestyle='--', label=f'True (Year {int(peak_year)})')

        ax.set_title(f"{paper[:15]}...\n{int(peak_year)} (N={peak_count})")
        ax.set_xlabel("Sample Size (Articles)")
        if i == 0:
            ax.set_ylabel("Weighted Slant Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = PLOT_DIR / "slant_convergence_yearly_test.png"
    plt.savefig(output_path)
    print(f"\n>>> Analysis Complete. Plot saved to: {output_path}")

if __name__ == "__main__":
    main()