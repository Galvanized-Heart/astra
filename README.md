# Astra
Astra is a multi-task model for predicting Michaelis-Menten kinetic parameters from elementary kinetic rate decompositions.

## Getting Started
Astra uses `uv` for development. To install `uv`, you can follow the instructions <a href=https://docs.astral.sh/uv/getting-started/installation>here</a> for updated instructions. At the time of writing, you can follow the instructions below to install `uv` and create the correct environment:
```
# Download and install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Download and install mmseqs2
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xvfz mmseqs-linux-avx2.tar.gz
export PATH=$(pwd)/mmseqs/bin/:$PATH
mv mmseqs/bin/mmseqs ~/.local/bin/
rm -rf mmseqs mmseqs-linux-avx2.tar.gz

# Create .venv
uv venv

# Sync uv .venv with uv.lock
uv sync --locked

# Sync addtional dev dependencies
uv sync --dev
```

## Brief Tutorial for uv Usage
Run scripts (e.g. Python or shell commands)
```
uv run <COMMAND>
```
<br>

Add dependencies to `pyproject.toml`
```
uv add <PACKAGE>
```
<br>

Remove dependencies from `pyproject.toml` and `uv.lock`
```
uv remove <PACKAGE>>
```
<br>

Sync environment with dependencies from `pyproject.toml`
```
uv sync
```
- `--locked` flag is used to sync environment from `uv.lock`. 
<br>

Update lockfile
```
uv lock
```
<br>

## NeurIPs 2025 Worshop Planning:

### Key Milestones

| Milestone                              | Date              |
| :------------------------------------- | :---------------- |
| **Target Submission Deadline**         | **Fri, Aug 22**   |
| Mandatory Accept/Reject Notification   | **Mon, Sep 22**   |
| NeurIPS 2025 Conference                | **Dec 2 - Dec 6** |

<br>

NeurIPs 2025 Potential Workshops:
- <a href=https://icml.cc/virtual/2025/workshop/39959>2nd Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences</a> (temporary link)
- <a href=https://ai4sciencecommunity.github.io/neurips25>AI for Science: The Reach and Limits of AI for Scientific Discovery</a>
    - Abstract Submission - Aug 18, 2025
    - Paper Submission - Aug 25, 2025
    - Review Bidding Period - Aug 25, 2025 to Aug 27, 2025
    - Review Deadline - Sep 18, 2025
    - Accept/Reject Notification - Sep 22, 2025
    - Workshop - Dec 6, 2025 to Dec 7, 2025 

<br>
This plan outlines the necessary tasks, subtasks, and deadlines to prepare a submission for the NeurIPS 2025 workshop by August 22, 2025 .

### ✍️ Writing & Publication

| Task                            | Subtasks                                                                                                                                                             | Deadline        | Status |
| :------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :----: |
| **Conduct Literature Review**   | - Identify and summarize 10-15 key papers.<br>- Create an annotated bibliography.<br>- Clearly define how our work is positioned relative to existing literature.         | **Wed, Jul 24** |   ☐    |
| **Write Detailed Paper Outline**| - Draft section headers (problem statement, proposed methods, and expected results).<br>- Write 2-4 bullet points of key content for each section.<br>- Explicitly list the paper's 3-4 key contributions.   | **Thu, Jul 25** |   ☐    |
| **Write Abstract**              | - Draft one sentence for: context, problem, method, result, and implication.<br>- Refine into a cohesive paragraph under the word limit.                                 | **Fri, Jul 26** |   ☐    |
| **Write First Draft**           | - Draft **Method** section first.<br>- Draft **Introduction** and **Related Work**.<br>- Draft **Experiments** section with planned setup.<br>- Create placeholders for all figures and tables. | **Fri, Aug 1**  |   ☐    |
| **Get & Implement Feedback**    | - Identify 2-3 reviewers and send the draft.<br>- Consolidate all feedback into a single document.<br>- Triage changes (Critical, High-Priority, Minor).<br>- Systematically implement revisions. | **Tue, Aug 19** |   ☐    |
| **Generate Figures & Tables**   | - Create a script (`plots.py`) to generate all visuals from saved results.<br>- Generate main performance and ablation tables.<br>- Generate interpretability plots.<br>- Write clear captions for everything. | **Wed, Aug 20** |   ☐    |
| **Final Proofread and Polish**  | - Perform a full grammar and spelling check.<br>- Read the entire paper aloud to catch awkward phrasing.<br>- Ensure formatting matches the workshop's style guide.  | **Thu, Aug 21** |   ☐    |

---

### 🏗️ Data & Infrastructure

| Task                                | Subtasks                                                                                                                                                                        | Deadline        | Status |
| :---------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------- | :----: |
| **Data Curation & Preprocessing**   | - Write scripts to download and parse raw data.<br>- Standardize data formats; handle missing values.<br>- Save the final, processed datasets to be used in training.              | **Mon, Jul 29** |   ☐    |
| **Create Data Splitting Logic**     | - Implement the random splitting function.<br>- Implement `mmseqs2` wrapper and cluster-based splitting.<br>- Write tests to ensure no data leakage between train/validation sets. | **Tue, Jul 30** |   ☐    |
| **Setup Experiment Tracking**       | - ~~Initialize a Weights & Biases project.~~<br>- Define standard metrics to log (loss, accuracy, F1, etc.).<br>- Ensure hyperparameters are automatically logged on every run. | **Wed, Jul 31** |   ☐    |

---

### 💻 Software Implementation

| Task                            | Subtasks                                                                                                                                                                 | Deadline        | Status |
| :------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :----: |
| **Create Model Architectures**  | - Implement the baseline model(s) (e.g., XGBoost, Linear).<br>- Implement the `sklearn.multioutput.MultiOutputRegressor` wrapper for XGBoost.<br>- ~~Implement the "advanced recomp" model as a PyTorch `nn.Module`~~.<br>- Add clear docstrings explaining input/output shapes. | **Fri, Aug 1**  |   ☐    |
| **Build Training & Loss Logic** | - ~~Create PyTorch `Dataset` and `DataLoader` classes.~~<br>- Build the main training script with `argparse` for key parameters.<br>- Implement training and validation loops.<br>- Integrate experiment tracking calls within the loops. | **Fri, Aug 1**  |   ☐    |
| **Optimize Hyperparameters**    | - Define the hyperparameter search space (learning rate, weight decay, etc.).<br>- Set up and run a sweep using a tool like Optuna or W&B Sweeps.<br>- Analyze results to find the best configuration. | **Wed, Aug 13** |   ☐    |

---

### 🔬 Experiments & Analysis

| Task                              | Subtasks                                                                                                                                                         | Deadline        | Status |
| :-------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :----: |
| **Run Baseline Experiments**      | - Run "individual models" experiment.<br>- Run "naive combined" experiment.<br>- Save all model checkpoints and a `results.json` summary for each run.               | **Fri, Aug 9**  |   ☐    |
| **Run Core Method Experiments**   | - Run "basic recomp" experiment.<br>- Run "advanced recomp" experiment.<br>- Document all results in the experiment tracking system.                               | **Wed, Aug 14** |   ☐    |
| **Run Interpretability Analysis** | - Write a script to load a trained model and a data sample.<br>- Extract and save feature coefficients/importance from linear/XGBoost models.<br>- Extract and visualize attention weights from attention-based models. | **Fri, Aug 16** |   ☐    |
