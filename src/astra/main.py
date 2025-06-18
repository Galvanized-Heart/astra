import click

@click.group()
def cli():
    """Astra."""
    pass

@cli.command()
@click.option('--name', default='World', help='The name to greet.')
def hello(name):
    """A simple hello command to test the CLI setup."""
    click.echo(f"Hello, {name}!")

if __name__ == '__main__':
    cli()