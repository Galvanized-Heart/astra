# Notes

### Greedy fold assignment
*A description of the algorithm that happens in the mmseqs2_split_w_ratio.py*

- Goal: Preserve the **ratio** of kinetic parameters from the **input dataset** in each of the folds for k-fold cross validation splits.
    - for every fold, the % of each kinetic column (kcat, KM, Ki) should match the original dataset’s percentages.

- Method: Greedy search
    - Make the locally best change available at each step - accept a swap only if it reduces deviation.

### Objective

Minimize total L1 deviation from the target across folds:

$$
J(a) = \sum_{f=1}^{k} \sum_{p \in P} \left| t_p - d_p\left(D_f(a)\right) \right|
$$

Lower $J(a)$ indicates better balance.

#### Where

- **$J(a)$** — Objective value (total L1 deviation across all folds).
- **$a$** — Assignment function mapping each cluster to a fold.
- **$k$** — Number of folds.
- **$P$** — Set of kinetic parameters considered.
- **$t_p$** — Target (global) percent of non‑null values for parameter $p$ in the full dataset.
- **$D_f(a)$** — Subset of rows assigned to fold $f$ under assignment $(a)$.
- **$d_p(S)$** — Percent of non‑null values for parameter $p$ within set $S$.




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