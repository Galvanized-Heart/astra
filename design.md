# Repo Design
This design.md is meant to be a point of organization for the M.A.Sc. research of Maxim Kirby under codename Astra. In this document, directory structure, data processing, model training, model inference, model architecture, package management, data acquisition, model weights acquisition, and experimental logging will all be detailed for high level conceptualization and reproducibility. A lot of this architecture is inspired by the setup **Botlz-2** used in June 2025 and the **cookiecutter-data-science** template.

## Directory Architecture
This layout of this machine learning experiment repository:

```
src/astra
├── data
│   ├── stored_data
│   ├── processing
│   ├── training
│   ├── inference
├── model
│   ├── layer
│   ├── modules
│   ├── models
│   ├── loss
│   ├── optim
main.py
tests (testing things to ensure code is working as intented (not sure what to test yet))
.gitignore (ignore certain files)
pyproject.toml (manage packages for effective development and reproducibility)
docs (documentation to help users run code in this repo)
examples (config files for example script usage (we can also used these for experiments))
scripts (scripts for running the code in this repo)
```

### src/astra
In `src/astra`, there will be a `main.py` file for a command line interface (CLI) for running the model. The command line script will need to be defined in `pyproject.toml` as: 
```
[project.scripts]
astra = "astra.main:cli"
```
and will include flags for customizing predictions. The CLI will be built using <a href=https://click.palletsprojects.com/en/stable/>click</a>.

Additionally, there will be a `data` folder for harbouring all the data required for this project. Python scripts will be used to split this data in reproducible ways, handle data processing (e.g.converting data formats, featurizing data), training models, and running inference on models. There might also need to be some logic for the user to download the data from a web server since github cannot host large amounts of data (something to keep in mind for usability).

Lastly, there will be a `model` folder for keeping all the logic related to model architecture. This would include Python files for creating layers in PyTorch, modules built from these layers, model architectures built from these modules, customizable loss functions, and customizable optimizer settings.