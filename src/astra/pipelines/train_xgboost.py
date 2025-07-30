import torch
import pandas as pd
import numpy as np
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from astra.data_processing.datasets import ProteinLigandDataset


def load_embeddings_from_manifest(manifest_path: str):
    """Load all embeddings and targets from a manifest into numpy arrays."""
    dataset = ProteinLigandDataset(manifest_path)

    protein_embeddings = []
    ligand_embeddings = []
    targets = []

    for sample in dataset:
        protein_embeddings.append(sample['protein_embedding'].numpy())
        ligand_embeddings.append(sample['ligand_embedding'].numpy())
        targets.append(sample['targets'].numpy())

    # Concatenate protein and ligand embeddings
    X = np.concatenate([np.stack(protein_embeddings), np.stack(ligand_embeddings)], axis=1)
    y = np.stack(targets)

    return X, y


def train_xgboost(train_manifest_path: str, valid_manifest_path: str, seed: int = 42):
    print("Loading training data...")
    X_train, y_train = load_embeddings_from_manifest(train_manifest_path)
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")

    print("Loading validation data...")
    X_valid, y_valid = load_embeddings_from_manifest(valid_manifest_path)
    print(f"Validation shape: X={X_valid.shape}, y={y_valid.shape}")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Set up XGBoost regressor (multi-output using sklearn wrapper)
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=seed,
        verbosity=1
    )
    model = MultiOutputRegressor(base_model)

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred, multioutput='uniform_average')

    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation R²: {r2:.4f}")

    return model

