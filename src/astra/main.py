from pathlib import Path

import click

from astra.constants import PROJECT_ROOT

DATA_PATH = Path.joinpath(PROJECT_ROOT, "data", "interim", "cpipred", "CPI_all_brenda_core_enriched.csv")

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
@click.option('--train_path', default=Path.joinpath(PROJECT_ROOT, "data", "split", "cpipred", "pangenomic", "mmseqs", "train.csv"), help='The path to data you want to train on.')
@click.option('--valid_path', default=Path.joinpath(PROJECT_ROOT, "data", "split", "cpipred", "pangenomic", "mmseqs", "valid.csv"), help='The path to data you want to validate on.')
@click.option('--batch_size', default=32, help='The batch size you want to train with.')
def train(train_path, valid_path, batch_size):
    """Base function for training Astra model."""
    from astra.pipelines.train import train

    click.echo(f"Setting up training for {train_path}.")
    click.echo(f"Using {valid_path} for validation.")

    # TODO: Write train.py to contain all this logic instead
    # NOTE: Having these as global imports slows the entire CLI signifiantly!!!
    #import lightning as L this is a BIG import
    #from astra.model.models import AstraModule
    train(train_path, valid_path, batch_size)

    click.echo("Training complete!")


def predict(test_path):
    # TODO: Use Trainer.validate() for inference on individual predictions and for testing datasets
    pass

if __name__ == '__main__':
    cli()