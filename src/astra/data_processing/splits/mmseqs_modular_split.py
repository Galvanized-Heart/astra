"""
End-to-End Dataset Splitting via Sequence Clustering with MMseqs2.

This script provides a comprehensive workflow to:
1. Read a CSV file containing biological sequences and associated data (kcat, KM, Ki).
2. Define a "signature" for each data point based on which values are present.
3. Cluster the sequences using MMseqs2 to group similar items.
4. Split the data into N cross-validation folds based on these clusters using one of three strategies:
   - random: The original method, randomly shuffling clusters.
   - stratified: A fast method that balances the *dominant* signature of each cluster.
   - greedy: A powerful heuristic that aims to perfectly balance the full signature composition of each fold.
5. Generate reports to verify split balance and analyze cluster sizes.
"""
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# ==============================================================================
# SECTION 1: Core MMseqs2 & File Helpers (Largely Unchanged)
# ==============================================================================

def create_fasta_from_csv(csv_path: str, seq_col: str, fasta_path: str):
    """Creates a FASTA file from a specified column in a CSV file."""
    df = pd.read_csv(csv_path)
    with open(fasta_path, 'w') as f:
        for idx, row in df.iterrows():
            f.write(f">{idx}\n{row[seq_col]}\n")


def run_mmseqs2_clustering(fasta_path: str, seq_id: float, coverage: float, cov_mode: int, threads: int, cluster_mode: int):
    """Runs MMseqs2 to cluster sequences and returns the cluster dictionary."""
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
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        clusters = {}
        with open(tsv_path) as f:
            for line in f:
                rep, member = line.strip().split('\t')
                clusters.setdefault(rep, []).append(member)
        return clusters

# ==============================================================================
# SECTION 2: NEW - Advanced Splitting Algorithms
# ==============================================================================

def get_signature(row, kcat_col, km_col, ki_col):
    """Determines the data signature (e.g., 'kcat_km_only') for a row."""
    has_kcat = pd.notna(row[kcat_col])
    has_km = pd.notna(row[km_col])
    has_ki = pd.notna(row[ki_col])
    
    parts = []
    if has_kcat: parts.append("kcat")
    if has_km: parts.append("km")
    if has_ki: parts.append("ki")
    
    return "_".join(parts) if parts else "no_data"


def split_clusters_stratified(clusters: dict, df_with_signatures: pd.DataFrame, n_folds: int, seed: int):
    """
    Splits clusters using StratifiedKFold based on the DOMINANT signature of each cluster.
    
    Nuance: This is a fast and effective approximation. It ensures that the number of
    "kcat-dominant" clusters, "km-dominant" clusters, etc., is balanced across folds.
    It does NOT balance the detailed composition within each cluster, making it less precise
    than the greedy algorithm but much better than random splitting.
    """
    print("--> Using 'stratified' splitting mode.")
    # 1. Determine the dominant signature for each cluster
    cluster_reps = []
    cluster_strata = [] # The 'y' variable for stratification
    for rep, members in clusters.items():
        member_indices = [int(m) for m in members]
        member_signatures = df_with_signatures.loc[member_indices, 'signature']
        dominant_signature = Counter(member_signatures).most_common(1)[0][0]
        
        cluster_reps.append(rep)
        cluster_strata.append(dominant_signature)

    # 2. Use StratifiedKFold to get fold assignments
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_assignments = {}
    # The 'X' is a placeholder; 'y' (the strata) is what matters
    for fold_num, (_, test_indices) in enumerate(skf.split(cluster_reps, cluster_strata)):
        for i in test_indices:
            rep_id = cluster_reps[i]
            fold_assignments[rep_id] = fold_num
            
    return fold_assignments


def split_clusters_greedy(clusters: dict, df_with_signatures: pd.DataFrame, n_folds: int):
    """
    Splits clusters using a greedy algorithm to achieve the most balanced folds.
    
    Nuance: This method provides the best balance. It considers the full, detailed
    signature composition of every cluster when assigning it. By sorting clusters from
    largest to smallest, it places the most "difficult" items first, which is a powerful
    heuristic for finding a near-optimal solution. It is more computationally
    intensive than stratification but more accurate.
    """
    print("--> Using 'greedy' splitting mode.")
    # 1. Characterize global data and target distribution
    all_signatures = df_with_signatures['signature'].unique()
    target_dist = df_with_signatures['signature'].value_counts() / n_folds

    # 2. Characterize each cluster
    cluster_profiles = []
    for rep, members in clusters.items():
        member_indices = [int(m) for m in members]
        signature_counts = df_with_signatures.loc[member_indices, 'signature'].value_counts().to_dict()
        cluster_profiles.append({
            "rep": rep,
            "size": len(members),
            "counts": Counter(signature_counts)
        })

    # Sort clusters from largest to smallest (crucial heuristic)
    cluster_profiles.sort(key=lambda x: x['size'], reverse=True)

    # 3. Iteratively assign clusters to the best-fit fold
    folds = [Counter() for _ in range(n_folds)]
    fold_assignments = {}

    for cluster in cluster_profiles:
        costs = []
        for i in range(n_folds):
            # Calculate Sum of Squared Errors cost for placing this cluster in fold i
            cost = 0
            for sig in all_signatures:
                current_count = folds[i].get(sig, 0)
                cluster_count = cluster['counts'].get(sig, 0)
                target_count = target_dist.get(sig, 0)
                # Error of the fold *after* adding the cluster
                error = (current_count + cluster_count) - (len(fold_assignments) / n_folds * target_count)
                cost += error ** 2
            costs.append(cost)
        
        best_fold_idx = np.argmin(costs)
        folds[best_fold_idx].update(cluster['counts'])
        fold_assignments[cluster['rep']] = best_fold_idx

    return fold_assignments


def split_clusters_random(clusters: dict, n_folds: int, seed: int):
    """Original random splitting method, adapted for N folds."""
    print("--> Using 'random' splitting mode.")
    cluster_reps = list(clusters.keys())
    random.Random(seed).shuffle(cluster_reps)
    
    fold_assignments = {}
    for i, rep in enumerate(cluster_reps):
        fold_assignments[rep] = i % n_folds
    return fold_assignments


# ==============================================================================
# SECTION 3: Reporting & Verification
# ==============================================================================

def print_split_report(df: pd.DataFrame, n_folds: int):
    """Prints a detailed report of the signature distribution across folds."""
    print("\n--- Split Balance Report ---")
    
    report = df.groupby('fold')['signature'].value_counts(normalize=True).unstack(fill_value=0)
    report = (report * 100).round(2) # Convert to percentage
    
    # Add a 'Total' column for sequence counts per fold
    report['Total Sequences'] = df['fold'].value_counts()

    # Add a 'Global' row for reference
    global_dist = df['signature'].value_counts(normalize=True)
    global_dist = (global_dist * 100).round(2)
    global_dist['Total Sequences'] = len(df)
    report.loc['Global Avg.'] = global_dist
    
    print("Signature Distribution (%) across Folds:")
    print(report.to_string())
    print("----------------------------\n")

# Main plotting and analysis functions from the original script can remain here...
def plot_cluster_size_distribution_capped(clusters: dict, output_path: str, xlim_max: int):
    """Generates a histogram of cluster sizes with a capped x-axis."""
    if not clusters: return
    cluster_sizes = pd.Series([len(members) for members in clusters.values()])
    plt.figure(figsize=(12, 7)); plt.hist(cluster_sizes, bins=np.arange(1, xlim_max + 2), alpha=0.8)
    mean_size = cluster_sizes.mean()
    plt.axvline(mean_size, color='r', linestyle='--', linewidth=2, label=f'Mean Size: {mean_size:.2f}')
    plt.xlim(0, xlim_max); plt.title(f'Cluster Size Dist. (Capped at {xlim_max})'); plt.xlabel('Cluster Size'); plt.ylabel('Frequency')
    plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.6); plt.tight_layout(); plt.savefig(output_path); plt.close()

def analyze_and_plot_clusters(clusters: dict, size_threshold: int, output_dir: str):
    """Analyzes cluster sizes, creating a capped plot and a report for large clusters."""
    large_clusters_info = [{'rep_id': rep, 'size': len(m)} for rep, m in clusters.items() if len(m) > size_threshold]
    if large_clusters_info:
        pd.DataFrame(large_clusters_info).sort_values('size', ascending=False).to_csv(f"{output_dir}/large_clusters.csv", index=False)
    plot_cluster_size_distribution_capped(clusters, f"{output_dir}/cluster_dist_capped_{size_threshold}.png", size_threshold)

# ==============================================================================
# SECTION 4: Main Orchestration Function
# ==============================================================================

def main_workflow(
    input_csv_path: str,
    output_dir: str,
    seq_col: str,
    kcat_col: str,
    km_col: str,
    ki_col: str,
    split_mode: str = 'greedy', # 'random', 'stratified', or 'greedy'
    n_folds: int = 5,
    seq_id: float = 0.8,
    coverage: float = 0.8,
    cov_mode: int = 0,
    cluster_mode: int = 0,
    threads: int = 4,
    seed: int = 42,
    cluster_size_cap: int = 50
):
    """Orchestrates the full data splitting and analysis workflow."""
    print("--- Starting Data Splitting and Analysis Process ---")
    
    # --- 1. Validation and Setup ---
    if not os.path.exists(input_csv_path): raise FileNotFoundError(f"Input not found: {input_csv_path}")
    if not shutil.which("mmseqs"): raise OSError("MMseqs2 not found in PATH.")
    if split_mode not in ['random', 'stratified', 'greedy']: raise ValueError("split_mode must be 'random', 'stratified', or 'greedy'")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv_path)

    # --- 2. Create Signatures and Run Clustering ---
    print("1. Defining data signatures (kcat/km/ki)...")
    df['signature'] = df.apply(get_signature, axis=1, args=(kcat_col, km_col, ki_col))
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".fasta", delete=False) as fasta_file:
        fasta_path = fasta_file.name
    try:
        print("2. Creating temporary FASTA and clustering with MMseqs2...")
        create_fasta_from_csv(input_csv_path, seq_col, fasta_path)
        clusters = run_mmseqs2_clustering(fasta_path, seq_id, coverage, cov_mode, threads, cluster_mode)
    finally:
        if os.path.exists(fasta_path): os.remove(fasta_path)

    # --- 3. Perform Cluster-Aware Splitting ---
    print(f"3. Splitting {len(clusters)} clusters into {n_folds} folds using '{split_mode}' mode...")
    if split_mode == 'random':
        fold_assignments = split_clusters_random(clusters, n_folds, seed)
    elif split_mode == 'stratified':
        fold_assignments = split_clusters_stratified(clusters, df, n_folds, seed)
    else: # greedy
        fold_assignments = split_clusters_greedy(clusters, df, n_folds)
    
    # --- 4. Map Fold Assignments back to DataFrame ---
    print("4. Mapping fold assignments to sequences...")
    # Create a map from each sequence member ID to its assigned fold
    split_map = {}
    for rep, members in clusters.items():
        fold_num = fold_assignments.get(rep)
        if fold_num is not None:
            for member_id in members:
                split_map[member_id] = fold_num
    
    df['fold'] = df.index.astype(str).map(split_map)
    
    unassigned_count = df['fold'].isnull().sum()
    if unassigned_count > 0:
        print(f"-> Warning: {unassigned_count} unclustered sequences found. Assigning them randomly to folds.")
        unassigned_indices = df[df['fold'].isnull()].index
        random_folds = np.random.randint(0, n_folds, size=len(unassigned_indices))
        df.loc[unassigned_indices, 'fold'] = random_folds

    df['fold'] = df['fold'].astype(int)

    # --- 5. Analyze Clusters, Verify Splits, and Save Files ---
    print("5. Analyzing cluster distribution...")
    analyze_and_plot_clusters(clusters, cluster_size_cap, output_dir)
    
    print("6. Verifying split balance and saving partitioned CSV files...")
    print_split_report(df, n_folds)
    
    for fold_num in range(n_folds):
        fold_df = df[df['fold'] == fold_num].copy()
        output_path = os.path.join(output_dir, f"fold_{fold_num}.csv")
        fold_df.to_csv(output_path, index=False)

    print("\n--- Process Complete ---")
    print(f"Find your {n_folds} split files and analysis reports in: {output_dir}")
    print("------------------------\n")


# ==============================================================================
# SECTION 5: Example Usage
# ==============================================================================

if __name__ == '__main__':
    # --- Create a dummy CSV for demonstration ---
    print("Creating a dummy 'input.csv' for demonstration...")
    # Let's create a more complex dataset to really test the splitting
    sequences = []
    metadata = []

    # Group 1: 100 sequences, mostly kcat_km
    for i in range(100):
        sequences.append(''.join(random.choices('ACGT', k=20)) + 'GROUPONE')
        metadata.append({'kcat': 1.0, 'km': 1.0, 'ki': np.nan if i % 10 != 0 else 1.0}) # 10% have ki

    # Group 2: 50 sequences, mostly ki_only
    for i in range(50):
        sequences.append(''.join(random.choices('ACGT', k=20)) + 'GROUPTWO')
        metadata.append({'kcat': np.nan, 'km': np.nan, 'ki': 1.0})

    # Group 3: 80 sequences, mixed kcat_only and no_data
    for i in range(80):
        sequences.append(''.join(random.choices('ACGT', k=20)) + 'GROUPTHREE')
        metadata.append({'kcat': 1.0 if i % 2 == 0 else np.nan, 'km': np.nan, 'ki': np.nan})

    dummy_df = pd.DataFrame(metadata)
    dummy_df['protein_sequence'] = sequences
    dummy_input_csv = "input_for_splitting.csv"
    dummy_df.to_csv(dummy_input_csv, index=False)
    
    # --- Run the main function if MMseqs2 is available ---
    if shutil.which("mmseqs"):
        # --- Example 1: Greedy Algorithm (Recommended) ---
        main_workflow(
            input_csv_path=dummy_input_csv,
            output_dir="split_output_greedy",
            seq_col="protein_sequence",
            kcat_col="kcat", km_col="km", ki_col="ki",
            split_mode='greedy',
            n_folds=5,
            seq_id=0.8,
            cluster_size_cap=50
        )
        
        # --- Example 2: Stratified Sampling ---
        main_workflow(
            input_csv_path=dummy_input_csv,
            output_dir="split_output_stratified",
            seq_col="protein_sequence",
            kcat_col="kcat", km_col="km", ki_col="ki",
            split_mode='stratified',
            n_folds=5,
            seq_id=0.8,
            cluster_size_cap=50
        )
    else:
        print("\n" + "="*50)
        print("WARNING: MMseqs2 executable not found. Main script was skipped.")
        print("Please install MMseqs2 and ensure it's in your system's PATH.")
        print("="*50 + "\n")