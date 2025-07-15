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
@click.option('--name', default='World', help='The name to greet.')
def hello(name):
    """A simple hello command to test the CLI setup."""
    click.echo(f"Hello, {name}!")

# Training script for Astra
@cli.command()
@click.option('--input_path', default=Path.joinpath(PROJECT_ROOT, "data", "interim", "cpipred", "CPI_all_brenda_core_enriched.csv"), help='The path to data you want to train on.')
def train(input_path):
    """Base function for training Astra model."""
    click.echo(f"Setting up training for {input_path}")
    # TODO: Use data_processing/ to create protein/ligand encodings/embeddings, Dataset, and DataLoader
    # TODO: Instantiate model architecture from model/
    # TODO: Initiate training pipelines/train.py
        # This should setup wandb tracking, save checkpoints, (other stuff?)

if __name__ == '__main__':
    cli()