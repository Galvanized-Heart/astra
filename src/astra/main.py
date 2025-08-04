import os
from pathlib import Path

import click

from astra.constants import PROJECT_ROOT
from astra.pipelines.run_train import run_training_engine

################################
########### WARNING ############
################################
###  DO NOT IMPORT torch OR  ###
### lightning INTO THIS FILE ###
################################
###  DOING SO WILL RUIN THE  ###
###       DETERMINISTIC      ###
###       FUNCTIONALIY       ###
################################

@click.group()
def cli():
    """Astra."""
    pass


# Test example of click cli
@cli.command()
@click.option('--name', default='User', help='The name to greet.')
def hello(name):
    """A simple hello command to test the CLI setup."""
    click.echo(f"Hello, {name}!")


@cli.command()
@click.option('--input_path', default=Path.joinpath(PROJECT_ROOT, "data", "split", "cpipred", "pangenomic", "mmseqs", "train.csv"), help='The path to data you want to train on.')
def manifest(input_path):
    # TODO: Use create_feature_manifest() to convert protein sequences and 
    # ligand SMILES to features and save feature paths inside manifest.csv
    pass


# Training script for Astra
@cli.command()
@click.option("--config_path", default=f"{PROJECT_ROOT}configs/experiments/test_config.yaml", help="The string path to config file used for training.")
def train(config_path):
    """CLI function for training Astra model."""
    # WARNING: Do not import torch or lightning into main.py directly or indirectly
    run_training_engine(config_path=config_path)


def predict(test_path):
    # TODO: Use Trainer.validate() for inference on individual predictions and for testing datasets
    pass

if __name__ == '__main__':
    cli()