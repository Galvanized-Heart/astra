# Notes

### Greedy fold assignment
*A description of the algorithm that happens in the mmseqs2_split_w_ratio.py*

- Goal: Preserve the **ratio** of kinetic parameters from the **input dataset** in each of the folds for k-fold cross validation splits.
    - for every fold, the % of each kinetic column (kcat, KM, Ki) should match the original dataset’s percentages.

- Method: Greedy assignment
    - Make the locally best change available at each step - accept a swap only if it reduces deviation.

### Objective

Minimise the total L1 deviation between each fold’s parameter ratio and the original dataset:

$$
\mathcal{D}(\mathcal{F})=\sum_{f=1}^{K}\ \sum_{p\in P}\left|\, r_{p,f}-r_{p,\mathrm{orig}} \right|
$$

Lower $\mathcal{D}$ indicates better balance.

#### Where

- **$\mathcal{D}$** — Total L1 deviation (objective value).
- **$\mathcal{F}$** — Function that assigns each cluster to one of the $K$ folds (greedy assignment).
- **$K$** — Number of folds.
- **$P$** — Set of kinetic parameters (Kcat, Ki, KM).
- $r_{p,f}$ — Fraction of samples in fold $f$ that contain a non-missing measurement for kinetic parameter $p$
- $r_{p,\mathrm{orig}}$ — The same fraction computed over the entire dataset.



### Code 

#### Supporting functions
- `calculate_kinetic_distribution(df, kinetic_cols)`: % non‑null per kinetic column
- `calculate_distribution_deviation(target, current)`: L1 gap (sum of absolute differences) between two % vectors


#### Main functions 
- `evaluate_fold_assignment(df, fold_assignments, n_folds, target, kinetic_cols)`
1. For each fold: collect rows whose cluster_id is assigned to that fold

2. Compute that fold’s distribution and its deviation from target

3. Return total deviation (sum across folds) and per‑fold stats

- `greedy_fold_assignment`
1. Initial assignment is done with `i % n_folds`
    e.g. if `n_folds` is 3 and you have cluster indices 0, 1, 2, 3...:
    ```
    Index 0: 0 % 3 = 0 (Fold 0)
    Index 1: 1 % 3 = 1 (Fold 1)
    Index 2: 2 % 3 = 2 (Fold 2)
    Index 3: 3 % 3 = 0 (Fold 0)
    Index 4: 4 % 3 = 1 (Fold 1)
    Index 5: 5 % 3 = 2 (Fold 2)
    ```
2. Calculate initial score stored in `best_score`, and keep a deep copy as `best_assignments`
3. Iterative search in `range(max_iterations)`
    - Pick two distinct clusters at random: cluster1, cluster2.
    - Skip if they are already in the same fold.
    - Swap their fold labels (two‑way swap).
    - Re‑score the entire assignment via evaluate_fold_assignment.
    - Accept if better: If new_score < best_score
    - If NO improvement: put both clusters back to their original folds.
*All rows with the same cluster_id move together, preventing similar‑sequence leakage across folds*