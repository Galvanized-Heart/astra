# -*- coding: utf-8 -*-
"""
End-to-End Dataset Splitting via Sequence Clustering with MMseqs2.

This script provides a comprehensive workflow to:
1. Read a CSV file containing biological sequences.
2. Cluster the sequences using MMseqs2 to group similar items.
3. Split the data into train/validation/test sets based on these clusters,
   preventing data leakage and ensuring robust model evaluation.
4. Generate a "capped" histogram to clearly visualize the distribution of
   the most common cluster sizes.
5. Create a CSV report detailing all clusters that exceed the specified size cap.
"""

import os
import random
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# SECTION 1: Core MMseqs2 & File Helpers
# ==============================================================================

def create_fasta_from_csv(csv_path: str, seq_col: str, fasta_path: str):
    """Creates a FASTA file from a specified column in a CSV file."""
    df = pd.read_csv(csv_path)
    with open(fasta_path, 'w') as f:
        for idx, row in df.iterrows():
            f.write(f">{idx}\n{row[seq_col]}\n")


def cluster_and_split(fasta_path: str, seq_id: float, split_ratios: dict, 
                      coverage: float, cov_mode: int, threads: int, 
                      cluster_mode: int, seed: int):
    """Runs MMseqs2 to cluster sequences and assigns them to splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/DB"
        cluster_path = f"{tmpdir}/clu"
        tsv_path = f"{tmpdir}/clusters.tsv"
        
        cmds = [
            ["mmseqs", "createdb", fasta_path, db_path],
            ["mmseqs", "cluster", db_path, cluster_path, f"{tmpdir}/tmp",
             "--min-seq-id", str(seq_id), "-c", str(coverage), "--cov-mode", str(cov_mode), 
             "--threads", str(threads), "--cluster-mode", str(cluster_mode)],
            ["mmseqs", "createtsv", db_path, db_path, cluster_path, tsv_path]
        ]
        
        for cmd in cmds:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"MMseqs2 failed at command: {' '.join(cmd)}\nStderr: {e.stderr}")
                raise
        
        clusters = {}
        with open(tsv_path) as f:
            for line in f:
                rep, member = line.strip().split('\t')
                clusters.setdefault(rep, []).append(member)

        cluster_reps = list(clusters.keys())
        rng = random.Random(seed)
        rng.shuffle(cluster_reps)
        
        n = len(cluster_reps)
        train_end = int(split_ratios['train'] * n)
        valid_end = train_end + int(split_ratios['valid'] * n)
        
        splits = {
            'train': cluster_reps[:train_end],
            'valid': cluster_reps[train_end:valid_end],
            'test': cluster_reps[valid_end:]
        }

        split_map = {}
        cluster_id_map = {}
        for split_name, reps in splits.items():
            for rep in reps:
                for member in clusters[rep]:
                    cluster_id_map[member] = rep
                    split_map[member] = split_name

        num_clusters = len(clusters)
        total_members = sum(len(members) for members in clusters.values())
        average_size = total_members / num_clusters if num_clusters > 0 else 0
        stats = {
            'num_clusters': num_clusters,
            'average_cluster_size': f"{average_size:.2f}",
            'sequences_clustered': total_members,
        }
        return split_map, cluster_id_map, stats, clusters


# ==============================================================================
# SECTION 2: Visualization & Reporting Helpers
# ==============================================================================

def plot_cluster_size_distribution_capped(clusters: dict, output_path: str, xlim_max: int):
    """Generates a histogram of cluster sizes with a capped x-axis."""
    if not clusters:
        return

    cluster_sizes = pd.Series([len(members) for members in clusters.values()])
    plt.figure(figsize=(12, 7))
    bins = np.arange(1, xlim_max + 2)
    plt.hist(cluster_sizes, bins=bins, alpha=0.8, label='Cluster Frequency')
    
    mean_size = cluster_sizes.mean()
    plt.axvline(mean_size, color='r', linestyle='--', linewidth=2, label=f'Overall Mean Size: {mean_size:.2f}')
    
    plt.xlim(0, xlim_max)
    plt.title(f'Distribution of Cluster Sizes (View Capped at Size {xlim_max})')
    plt.xlabel(f'Number of Sequences per Cluster (Capped at {xlim_max})')
    plt.ylabel('Frequency (Number of Clusters)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_and_plot_clusters(clusters: dict, size_threshold: int, output_dir: str):
    """
    Analyzes cluster sizes, creating a capped plot and a report for large clusters.
    """
    large_clusters_info = [
        {'representative_id': rep, 'cluster_size': len(members)}
        for rep, members in clusters.items() if len(members) > size_threshold
    ]

    report_path = os.path.join(output_dir, 'large_clusters_report.csv')
    if large_clusters_info:
        large_clusters_df = pd.DataFrame(large_clusters_info).sort_values('cluster_size', ascending=False)
        large_clusters_df.to_csv(report_path, index=False)
        print(f"-> Found {len(large_clusters_df)} clusters larger than {size_threshold}. Report saved to: {report_path}")
    else:
        print(f"-> No clusters found larger than {size_threshold}.")

    plot_path = os.path.join(output_dir, f'cluster_dist_capped_at_{size_threshold}.png')
    plot_cluster_size_distribution_capped(clusters, plot_path, size_threshold)
    print(f"-> Capped distribution plot saved to: {plot_path}")


# ==============================================================================
# SECTION 3: Main Orchestration Function
# ==============================================================================

def mmseqs2_split_data_into_files(
    input_csv_path: str,
    output_dir: str,
    seq_col: str = "protein_sequence",
    seq_id: float = 0.8,
    split_ratios: dict = None,
    coverage: float = 0.8,
    cov_mode: int = 0,
    cluster_mode: int = 0,
    threads: int = 4,
    seed: int = 42,
    cluster_size_cap: int = 50
):
    """
    Orchestrates the full data splitting and analysis workflow.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_dir (str): Directory for all output files.
        seq_col (str): Name of the column containing sequences.
        seq_id (float): Sequence identity threshold for clustering.
        split_ratios (dict): Ratios for train/valid/test splits.
        coverage (float): Coverage threshold for clustering.
        cov_mode (int): Coverage mode for MMseqs2.
        cluster_mode (int): Cluster mode for MMseqs2.
        threads (int): Number of CPU threads for MMseqs2.
        seed (int): Random seed for reproducible splitting.
        cluster_size_cap (int): Threshold for reporting large clusters and
                                for capping the visualization plot.
    """
    print("--- Starting Data Splitting and Analysis Process ---")
    
    # --- 1. Validation and Setup ---
    if not os.path.exists(input_csv_path): raise FileNotFoundError(f"Input not found: {input_csv_path}")
    if not shutil.which("mmseqs"): raise EnvironmentError("MMseqs2 not found in PATH.")
    if split_ratios is None: split_ratios = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
    if abs(sum(split_ratios.values()) - 1.0) > 1e-6: raise ValueError("Split ratios must sum to 1.0.")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv_path)

    # --- 2. Create FASTA and Run Clustering ---
    with tempfile.NamedTemporaryFile(mode='w', suffix=".fasta", delete=False) as fasta_file:
        fasta_path = fasta_file.name
    try:
        print(f"1. Creating temporary FASTA file...")
        create_fasta_from_csv(input_csv_path, seq_col, fasta_path)

        print("2. Clustering sequences with MMseqs2...")
        split_map, cluster_map, stats, clusters = cluster_and_split(
            fasta_path, seq_id, split_ratios, coverage, cov_mode, threads, cluster_mode, seed
        )
    finally:
        if os.path.exists(fasta_path): os.remove(fasta_path)

    # --- 3. Analyze and Visualize Cluster Distribution ---
    print(f"3. Analyzing cluster distribution with a size cap of {cluster_size_cap}...")
    analyze_and_plot_clusters(clusters, cluster_size_cap, output_dir)
        
    # --- 4. Assign Splits and Save Data Files ---
    print("4. Assigning splits and saving partitioned CSV files...")
    df['cluster_id'] = df.index.astype(str).map(cluster_map)
    df['split'] = df.index.astype(str).map(split_map)
    
    if df['split'].isnull().sum() > 0:
        print(f"-> Warning: {df['split'].isnull().sum()} sequences were unclustered. Assigning to 'train'.")
        df['split'].fillna('train', inplace=True)

    output_files = {}
    for split_name in ['train', 'valid', 'test']:
        split_df = df[df['split'] == split_name].copy().drop(columns=['split'])
        output_path = os.path.join(output_dir, f"{split_name}.csv")
        split_df.to_csv(output_path, index=False)
        output_files[split_name] = {'path': output_path, 'count': len(split_df)}

    # --- 5. Final Report ---
    print("\n--- Process Complete ---")
    print("Clustering Statistics:")
    for key, value in stats.items(): print(f"  - {key.replace('_', ' ').title()}: {value}")
    
    print("\nOutput Files:")
    for split_name, info in output_files.items(): print(f"  - {info['path']} ({info['count']} rows)")
    print(f"  - {os.path.join(output_dir, f'cluster_dist_capped_at_{cluster_size_cap}.png')}")
    if os.path.exists(os.path.join(output_dir, 'large_clusters_report.csv')):
        print(f"  - {os.path.join(output_dir, 'large_clusters_report.csv')}")
    print("------------------------\n")

# ==============================================================================
# SECTION 4: Example Usage
# ==============================================================================

if __name__ == '__main__':
    # --- Create a dummy CSV for demonstration ---
    print("Creating a dummy 'input.csv' for demonstration...")
    np.random.seed(42)
    small = [1] * 100 + [2] * 20 + list(np.random.randint(3, 50, 20))
    medium = [55, 67, 120]
    all_sizes = small + medium
    sequences = []
    for i, size in enumerate(all_sizes):
        base_seq = ''.join(random.choices('ACGT', k=30))
        for j in range(size):
            # Create slight variations for each member
            mutated_seq = list(base_seq)
            mutated_seq[j % 30] = random.choice('ACGT')
            sequences.append(''.join(mutated_seq))
            
    dummy_df = pd.DataFrame({'SEQ': sequences})
    dummy_input_csv = "input_for_splitting.csv"
    dummy_df.to_csv(dummy_input_csv, index=False)
    
    OUTPUT_DIRECTORY = "final_split_output"
    
    # --- Run the main function if MMseqs2 is available ---
    if shutil.which("mmseqs"):
        mmseqs2_split_data_into_files(
            input_csv_path=dummy_input_csv,
            output_dir=OUTPUT_DIRECTORY,
            seq_col="SEQ",
            seq_id=0.7, # Lower identity to find more clusters
            cluster_size_cap=50 # Set the cap for reporting/plotting
        )
    else:
        print("\n" + "="*50)
        print("WARNING: MMseqs2 executable not found. Main script was skipped.")
        print("Please install MMseqs2 and ensure it's in your system's PATH.")
        print("="*50 + "\n")