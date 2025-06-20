# Repository Design
This design.md is meant to be a point of organization for the M.A.Sc. research of Maxim Kirby under codename Astra. In this document, directory structure, data processing, model training, model inference, model architecture, package management, data acquisition, model weights acquisition, and experimental logging will all be detailed for high level conceptualization and reproducibility. A lot of this architecture is inspired by the setup **Botlz-2** used in June 2025 and the **cookiecutter-data-science** template.

## Directory Architecture
This is the layout for this machine learning experiment repository:

```
astra
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/astra
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── model
│   │   ├── layer
│   │   ├── modules
│   │   ├── models
│   │   ├── loss
│   │   └── optim
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── __init__.py
│   └── main.py
├── configs/
│   ├── data/
│   └── experiments/
│       └── exp001.yaml
├── scripts/
│   ├── download_data
│   └── run_experiment
├── tests/
├── docs/
├── README.md
├── design.md
├── pyproject.toml
└── .gitignore
```

### data
`data` will be a directory used to store the data used in these experiments. Often times, data files are very large and cannot be uploaded to GitHub. For that reason, this folder will be in `.gitignore` and `docs` will specifiy how to dowload the data for reproducing these experiments. Data is often stored on servers like zenodo, google cloud, or amazon web services to be downloaded later and `scripts` is used to make it more accessible to users. 

There will be a `raw` folder containing the data in its raw format and `scripts` will be used to process the data into intermediate and analysis ready formats in the `interim` and `processed` folders respectively.

### src/astra
In `src/astra`, there will be a `main.py` file for a command line interface (CLI) for running the model. The command line script will need to be defined in `pyproject.toml` as: 
```
[project.scripts]
astra = "astra.main:cli"
```
and will include flags for customizing predictions. The CLI will be built using <a href=https://click.palletsprojects.com/en/stable/>click</a>.

Additionally, there will be a `data_processing` folder for harbouring all the data required for this project. Python scripts will be used to clean this data, split it, and handle data processing in reproducible ways (e.g.converting data formats, featurizing data).

There will be a `model` folder for keeping all the logic related to model architecture. This would include Python files for creating layers in PyTorch, modules built from these layers, model architectures built from these modules, desired metrics, customizable loss functions, and customizable optimizer settings.

Lastly, `pipelines` will contain the main logic for training loop, inference, and evaluations.

### configs
`configs` is meant to be a place to store settings for experimental runs. There will be run using YAML files to configure model hyperparamters for training experiments and to configure data preprocessing. There are also other ways to use YAML files in ML pipelines such as defining model architecture and deploying models for higher throughput experiments. All these are outlined pretty nicely in this <a href=https://rumn.medium.com/simplifying-machine-learning-workflow-with-yaml-files-e146cb3d481a>Medium post</a> and they also discuss how YAML syntax works.

In the Boltz-2 repository, they use YAML or fasta files for data formatting. For example, this `ligand.yaml` in the repository at the time of writing this:
```
version: 1  # Optional, defaults to 1
sequences:
  - protein:
      id: [A, B]
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ
      msa: ./examples/msa/seq1.a3m
  - ligand:
      id: [C, D]
      ccd: SAH
  - ligand:
      id: [E, F]
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
```
It might also be wise to use <a href=https://docs.pydantic.dev/latest/>Pydantic</a> to make this processes more robust. Converting the YAML into a Pydantic model with type hints should allow for better error handling. 
- TODO: Look into Pydantic for configs.

### scripts
In `scripts`, there will be logic used to run preconfigured pipelines within this repository at a high level. This will involve downloading raw data, downloading model weights, processing the raw data, training a model from scratch, fine-tuning a pretrained model, running inference on a model, evaluating a model on a given dataset. This will help streamline many of the processes involved in experimenting and also help users experiment with the code once it is published.

### tests
`tests` will be a folder containing test scripts for ensuring components from this repository work as intented. This could look at testing layers, loss functions, data processing, model on example data as a method of verifying that the program runs smoothly. This development will be done using <a href=https://docs.pytest.org/en/7.1.x/contents.html>pytest</a>.

### docs
`docs` will contain markdown files of how to navigate this repository and how to make use of the scripts for training, evaluating, and running their own predictions.

### pyproject.toml
`pyproject.toml` is a development file for ensuring a reproducible setup. It is similar to a `requirements.txt` file, but much more verbose. This file is meant to manage package versions during development as well as create a CLI. For more details about `pyproject.toml` the <a href=https://packaging.python.org/en/latest/guides/writing-pyproject-toml/>Python docs</a> have a page dedicated to it. For managing the packages in `pyproject.toml` and developing using a virtual environment, there are several options but many recommend using <a href=https://docs.astral.sh/uv/>uv</a> since it is written in rust (a low level language, making it super light weight).

Using `pyproject.toml` will also ensure users can install this repository reproducibly and run inference more easily.

## 