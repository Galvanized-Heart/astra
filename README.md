# Astra

Astra is a multi-task learning framework for enzyme kinetics predictiong. Given a protein sequence and a substrate SMILES string, Astra jointly predicts $`k_\text{cat}`$, $`K_M`$, and $`K_i`$ by learning the underlying elementary rate constants from which these observable quantities are derived, enforcing physical consistency directly in the model's parameterization.

---

## Motivation

Standard deep learning approaches to enzyme kinetics treat $`k_\text{cat}`$, $`K_M`$, and $`K_i`$ as independent regression targets, ignoring both the mechanistic relationships between them and the natural overlap in their kinetic components.

Consider the standard Michaelis-Menten mechanism:

```math
\text{E} + \text{S} \underset{k_{-1}}{\stackrel{k_1}{\rightleftharpoons}} \text{ES} \xrightarrow{k_2} \text{E} + \text{P}
```

The three observable kinetic parameters are not independent quantities. They are all derived from the same small set of elementary rates:

```math
K_M = \frac{k_{-1} + k_2}{k_1}, \qquad k_\text{cat} = k_2, \qquad K_i = \frac{k_{-1}}{k_1}
```

Each observed parameter shares components with the others. $`k_\text{cat}`$ is $`k_2`$ directly. $`K_i`$ is the ratio of $`k_{-1}`$ to $`k_1`$. $`K_M`$ is a function of all three rates and therefore contains both $`k_\text{cat}`$ and $`K_i`$ as embedded subexpressions. Joint learning is therefore not a convenience but a consequence of the underlying kinetic structure. A model that predicts $`k_\text{cat}`$ and $`K_M`$ separately has no mechanism to enforce that the $`k_2`$ they each imply is the same value.

Astra takes a different approach. Rather than learning the observed parameters directly, the model predicts the elementary rate constants from which they are derived, then applies a closed-form recomposition to obtain $`k_\text{cat}`$, $`K_M`$, and $`K_i`$. This mechanistic decomposition serves as an inductive bias that constrains the hypothesis space to solutions that are biochemically consistent by construction.

An additional practical benefit is that not every protein-ligand pair in real datasets has measurements for all three parameters. Astra handles this with masked loss functions that compute gradients only over observed labels, allowing the model to learn from partially annotated data without discarding any observations.

---

## Architecture

![Architecture Diagram](images/architecture.svg)

### Featurization

| Modality | Method |
|---|---|
| Protein sequence | ESM2 650M embeddings |
| Substrate | 2048-bit ECFP4 |

Both featurizers are modular. Custom featurizers and backbone architectures can be substituted via the registry system *(documentation in progress)*.

### Model Architectures

Astra evaluates four backbone architectures adapted from [CPI-Pred](https://doi.org/10.1101/2025.01.16.633372):

| Architecture | Description |
|---|---|
| `LinearBaselineModel` | Mean-pooled protein embedding concatenated with ligand fingerprint, passed through a two-layer MLP |
| `CpiPredConvModel` | 1D convolution over the protein sequence with dual-branch residual structure |
| `CpiPredSelfAttnModel` | Self-attention over protein tokens with attention-weighted pooling |
| `CpiPredCrossAttnModel` | Cross-attention where the ligand queries the protein sequence |

---

## Michaelis-Menten Recomposition

The recomposition layer is the core architectural contribution of Astra. The model predicts elementary rates in $`\log_{10}`$ space and derives the observable kinetic constants through closed-form expressions that operate entirely in log space, avoiding the need to exponentiate intermediate values.

### Basic Recomposition (3-rate mechanism)

For the standard Michaelis-Menten mechanism introduced above, the model predicts $`[\log k_1,\ \log k_{-1},\ \log k_2]`$ and computes:

```math
\log k_\text{cat} = \log k_2
```

```math
\log K_i = \log k_{-1} - \log k_1
```

```math
\log K_M = \log(k_{-1} + k_2) - \log k_1
```

The sum $`\log(k_{-1} + k_2)`$ is computed using a numerically stable log-sum-exp identity. Naively exponentiating, summing, and taking the log risks catastrophic cancellation when the two rates differ by many orders of magnitude. The stable form avoids this:

```math
\log_{10}(a + b) = \max(\log a,\ \log b) + \log_{10}\!\left(1 + 10^{\min(\log a, \log b) - \max(\log a, \log b)}\right)
```

### Advanced Recomposition (5-rate mechanism)

For a more verbose Michaelis-Menten mechanism including a reversible enzyme-product complex intermediate:

```math
\text{E} + \text{S} \underset{k_{-1}}{\stackrel{k_1}{\rightleftharpoons}} \text{ES} \underset{k_{-2}}{\stackrel{k_2}{\rightleftharpoons}} \text{EP} \xrightarrow{k_3} \text{E} + \text{P}
```

The model predicts $`[\log k_1,\ \log k_{-1},\ \log k_2,\ \log k_{-2},\ \log k_3]`$ and derives:

```math
\log k_\text{cat} = \log k_2 + \log k_3 - \log(k_2 + k_{-2} + k_3)
```

```math
\log K_M = \log(k_{-1}k_{-2} + k_{-1}k_3 + k_2 k_3) - \log k_1 - \log(k_2 + k_{-2} + k_3)
```

```math
\log K_i = \log k_{-1} - \log k_1
```

All multi-term sums are computed through the same numerically stable log-sum-exp identity.

---

## Multi-Task Learning

Jointly training on $`k_\text{cat}`$, $`K_M`$, and $`K_i`$ introduces competing gradient signals. Naive loss summation allows dominant tasks to suppress learning in others. Astra addresses this with two complementary strategies.

### Uncertainty-Weighted Loss

Based on [Kendall et al. (CVPR 2018)](https://arxiv.org/abs/1705.07115). Each task is assigned a learned log-variance parameter $`s_i = \log \sigma_i^2`$. The total loss becomes:

```math
\mathcal{L} = \sum_{i} \frac{1}{2} e^{-s_i} \mathcal{L}_i + \frac{1}{2} s_i
```

The model learns to downweight tasks it is uncertain about rather than relying on hand-tuned loss coefficients. The precision term $`e^{-s_i}`$ scales the task loss, while the regularization term $`\frac{1}{2} s_i`$ prevents the model from collapsing all weights to zero.

### Conflict-Averse Gradient Descent (CAGrad)

Based on [Liu et al. (NeurIPS 2021)](https://arxiv.org/abs/2110.14048). Standard multi-task optimizers update along the average gradient ($`g_0 = \frac{1}{K}\sum_i \nabla \mathcal{L}_i`$), which can actively harm individual tasks when gradients point in conflicting directions. CAGrad instead finds the optimal update vector ($`d^*`$) that maximizes the minimum per-task descent rate, subject to staying within a trust region around $`g_0`$:

```math
\max_{d} \min_{i \in [K]} \langle g_i, d \rangle \quad \text{s.t.} \quad \|d - g_0\| \leq c\|g_0\|
```

where $`c \in [0, 1)`$ controls the size of the constraint region. Solving this directly in parameter space is intractable for large networks, so CAGrad solves the dual problem over a weight vector $`w \in \mathbb{R}^K`$ on the probability simplex instead:

```math
w^* = \arg\min_{w \in \mathcal{W}}\ g_w^\top g_0 + \sqrt{\phi}\ \|g_w\|, \quad \text{where } g_w = \textstyle\sum_i w_i g_i,\ \phi = c^2 \|g_0\|^2
```

The optimal update is then $`d^* = g_0 + \frac{\sqrt{\phi}}{\|g_{w^*}\|} g_{w^*}`$. Because $`w`$ has dimension equal to the number of tasks rather than the number of parameters, this inner optimization is cheap regardless of model size.

---

## Data Pipeline

### Sequence-Similarity Splitting

To prevent data leakage between train and validation folds, Astra clusters protein sequences by sequence identity using [MMseqs2](https://github.com/soedinglab/MMseqs2) before splitting. This ensures that no protein with high sequence similarity to a validation sequence appears in the training set, producing a more realistic estimate of generalization to novel enzymes.

Three splitting strategies are available:

- **Random:** shuffles clusters and assigns to folds
- **Stratified:** balances the dominant kinetic signature (kcat-only, KM-only, all-three, etc.) across folds
- **Greedy:** iteratively assigns clusters to minimize deviation from the target kinetic parameter distribution across all folds simultaneously

### Masked Loss for Partial Labels

Real enzyme kinetics datasets are sparsely annotated. Many protein-ligand pairs have measurements for only one or two of the three target parameters. Astra uses masked MSE loss functions that compute gradients only over observed labels, allowing full use of the available data without imputation or label dropping.

---

## Installation

```bash
git clone https://github.com/Galvanized-Heart/astra.git
cd astra

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install MMseqs2 (required for data splitting)
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xvfz mmseqs-linux-avx2.tar.gz
mv mmseqs/bin/mmseqs ~/.local/bin/
rm -rf mmseqs mmseqs-linux-avx2.tar.gz

# Create environment and install dependencies
uv venv --python 3.11
uv sync --locked
uv sync --dev
```

*GPU strongly advised for ESM2 featurization.*

---

## Usage

*Demo and full usage examples coming soon.*

---

## Roadmap

- [x] Basic Michaelis-Menten recomposition (3-rate mechanism)
- [x] Advanced Michaelis-Menten recomposition (5-rate mechanism)
- [x] Numerically stable log-space recomposition
- [x] Joint $`k_\text{cat}`$ / $`K_M`$ / $`K_i`$ prediction with masked loss
- [x] Uncertainty-weighted multi-task loss (Kendall et al.)
- [x] MMseqs2-based sequence-similarity data splitting
- [x] W&B experiment tracking with Hydra configuration
- [ ] CAGrad optimizer integration (Liu et al.)
- [ ] Custom featurizer API and documentation
- [ ] Custom architecture API and documentation
- [ ] Demo interface

---

## References

- Xu, Z., Barghout, R. A., Wu, J., Garg, D., Song, Y. S., Mahadevan, R. (2025) [CPI-Pred: A deep learning framework for predicting functional parameters of compound-protein interactions.](https://doi.org/10.1101/2025.01.16.633372) *bioRxiv*.
- Kendall, A., Gal, Y., and Cipolla, R. (2018). [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.](https://arxiv.org/abs/1705.07115) *CVPR*.
- Liu, B., Liu, X., Jin, X., Stone, P., and Liu, Q. (2021). [Conflict-Averse Gradient Descent for Multi-task Learning.](https://arxiv.org/abs/2110.14048) *NeurIPS*.
- Lin, Z., et al. (2023). [Evolutionary-scale prediction of atomic-level protein structure with a language model.](https://www.science.org/doi/10.1126/science.ade2574) *Science* (ESM2).
- Steinegger, M. and Söding, J. (2017). [MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets.](https://www.nature.com/articles/nbt.3988) *Nature Biotechnology* (MMseqs2).
