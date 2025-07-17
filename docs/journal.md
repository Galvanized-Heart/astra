### Jun 25, 2025:
Today, I've been conceptualizing how research papers could look like with different model architectures and how we could do multi-task learning for predicting kinetic parameters. 

The model architecture would likely be compared to existing models in the space that use embeddings + XGBoost (<a href=https://doi.org/10.1038/s41467-023-44113-1>UniKP</a>/<a href=https://doi.org/10.1038/s41467-023-39840-4>TuRNuP</a>) or things like embeddings + attention (<a href=https://doi.org/10.1038/s41467-025-57215-9>CatPred</a>/<a href=https://doi.org/10.1101/2025.01.16.633372>CPI-Pred</a>) to using 3D structures as tensors (method for this is undecided) or as graphs (with 3D structured embedded in them), the same could be done for co-structure generation from <a href=https://doi.org/10.1101/2025.06.14.659707>Boltz-2</a> (tensor/graph) and then last would be to using <a href=https://doi.org/10.1038/s42256-025-01031-9>SIMG*</a> (stereoelectronics-infused molecular graphs) graphs since they seem very robust. I wonder if we do need tensors for all this of if we can just go straight to graphs.

For the multi-task learning, this could look like a couple things and this it more thouroughly thought out at this point in time. The standard in the field right now it to created a model for each kinetic parameter (kcat, Ki, KM) and predict them. A naive approach to multi-task learning could be to predict kcat, Ki, and KM all in a single model with weightings for each loss and masks for which components are being predicted. A more robust approach would look at the components of kcat, Ki, and KM and how they interact: 
- $`k_{cat}=k_{cat}`$
- $`K_{i,competitive}=K_D= \frac{k_{-1}}{k_{+1}}`$
- $`K_M=\frac{k_{-1}+k_{cat}}{k_{+1}}`$

From these equations, I noticed you can express kcat, Ki competitive, KD, and KM as compounds of kcat (rate of catalysis), k-1 (rate of unbinding), and k+1 (rate of binding). Additionally, I'm not sure people making these models were accounting for the difference between competitive, non-competitive, and uncompetitive inhibition. It might be smartest to seperate them as different predictions, but this work would just use competitive inhibition since it is most notably binding. I know we definitely aren't predicting how complexes are working in this work of previous work, until something very effective comes out regarding this.

I've also seen Boltz-2 predict IC50 with continuous values and whether something binds as a binary function to get a confidence prediction. The loss could be extended further to predicting the probabilty something binds 
$`\hat{P}_{binding} \in \Set{0,1}`$
and something gets converted to product 
$`\hat{P}_{catalysis} \in \Set{0,1}`$.
 If we assume that the data passes through the model like this: 

[Protein Representation, Ligand Representation] -> Model -> [
    $`\hat{k}_{cat}`$, 
    $`\hat{k}_{-1}`$, 
    $`\hat{k}_{+1}`$, 
    $`\hat{P}_{binding}`$, 
    $`\hat{P}_{catalysis}`$]

Where we recreate kcat, Ki, and KM with the functions outlined earlier, we can create a loss function that looks like this to train the model:

$`\mathcal{L} = 
\alpha m_{k_{cat}} MSE(k_{cat},\hat{k}_{cat}) + 
\beta m_{K_{i}} MSE(K_{i},\hat{K}_{i}) + 
\gamma m_{K_{M}} MSE(K_{M},\hat{K}_{M}) + 
\delta m_{binding} BCE(P_{binding},\hat{P}_{binding}) + 
\epsilon m_{catalysis} BCE(P_{catalysis},\hat{P}_{catalysis})`$

Where $`\alpha, \beta, \gamma, \delta, \epsilon`$ are weights for the individual loss functions and $`m_{x} \in \Set {0,1}`$ is the mask for whichever prediction is being used since not every entry will have kcat, Ki, and/or KM associated with it. There's a nice paper outlining how to do this: https://doi.org/10.48550/arXiv.1705.07115. The two types of loss functions are $`MSE()`$ for regression tasks and $`BCE()`$ for binary classification tasks.

I still have to formulate the raw data for this problem. So far, BRENDA, SABIO-RK, and BindingDB are good sources for training data. But I want to find a good strategy for getting good quality data. CPI-Pred has shown that using the pangenomic dataset lead to poorer performance when the dataset was split using sequence similarity, but not when using compound similarity.



### Jun 26, 2025:
Today, I thought that the data I got from the SABIO-RK API didn't contain any values for kcat, KM, or Ki and I looked into how to fix that. But I ended up finding out that I had most of the data I needed and I learned about how I could scrape the SABIO-RK website for some missing info that I wanted using this address: https://sabiork.h-its.org/kineticLawEntry.jsp?viewData=true&kinlawid=[SABIO-RK ENTRY ID].

By querying https://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws/entryIDs, I am able to get all the entryIDs from the SABIO-RK API. The step I was taking after that was querying https://sabiork.h-its.org/sabioRestWebServices/kineticLaws with a batch of entryIDs and this gave me what I wanted.

After a while of searching, I came across <a href=https://github.com/KarrLab/datanator/blob/master/datanator/data_source/sabio_rk.py>this repo</a> from the Karr Lab, which uses a combination of webscraping and the API to build their dataset back in 2018. I want to adapt this for my purposes.

Ultimately, I thought about the data I want to collect and this seems sufficient and doable to me.

EntryID (Have by default), 

organismName (Supplement with webscraping using EntryID and https://sabiork.h-its.org/kineticLawEntry.jsp?viewData=true&kinlawid=x), 
reactionID (search SBML for rdf:li rdf:resource="https://identifiers.org/sabiork.reaction/x), 
uniprotID (search SBML for rdf:li rdf:resource="https://identifiers.org/uniprot/XXXXXX"), 
ecClass (search SBML for rdf:li rdf:resource="https://identifiers.org/ec-code/x.x.x.x"), 

reactantIDs (search SBML for <listOfReactants> to get id from species=x, then search SBML for <species id="SPC_1262_Cell" name="NADPH"...>, then find <rdf:li rdf:resource="https://identifiers.org/chebi/CHEBI:XXXXX"/> and/or <rdf:li rdf:resource="https://identifiers.org/kegg.compound/XXXXXX"/> resources), 
productIDs (repeat reactants step for <listOfProducts>), 
modifierIDs (repeat reactants step for <listOfModifiers>), 

kcatValue (search SBML for <listOfLocalParameters> and get id=Kcat... & value=x), 
kcatDeviation (Supplement with webscraping using EntryID and https://sabiork.h-its.org/kineticLawEntry.jsp?viewData=true&kinlawid=x), 
kcatUnits (search SBML for <listOfLocalParameters> and get id=Kcat... & units=x) (units might look weird -> Mwedgeoneswedgeone = M⁻¹s⁻¹), 
kcatSpecies (search SBML for <listOfLocalParameters> and it from the id as id=Kcat_XXX_XXXX_XXXX), 

kmValue (repeat kcat steps for id=Km...), 
kmDeviation, 
kmSpecies, 
kmUnits, 

vmaxValue (repeat kcat steps for id=Vmax...), 
vmaxDeviation, 
vmaxSpecies, 

kiValue (repeat kcat steps for id=Ki...), 
kiDeviation, 
kiUnits, 
kiSpecies,
kiType (requires SBML parsing of the <math> block to determine whether Ki is competitive, non-competitive, or mixed) (see below)
v = Vmax * [S] / (Km * (1 + [I]/Ki) + [S]) is competitive
v = Vmax * [S] / (Km + [S] * (1 + [I]/Ki)) and v = (Vmax / (1+[I]/Ki)) * [S] / ((Km / (1+[I]/Ki)) + [S]) are uncompetitive
v = (Vmax / (1 + [I]/Ki)) * [S] / (Km + [S]) is non-competitive
v = Vmax * [S] / (Km * (1 + [I]/Kic) + [S] * (1 + [I]/Kiu)) is mixed

temperature (search SBML for <sbrk:startValueTemperature>), 
tempUnits (search SBML for <sbrk:temperatureUnit>), 
pH (search SBML for <sbrk:startValuepH>), 
buffer (search SBML for <sbrk:buffer>), 
pubmedID (search SBML for <rdf:li rdf:resource="https://identifiers.org/pubmed/XXXXXXXX"/>)

I did notice that some entries on the website had something like ([uniprot ID])*n where n was sometimes "n" and sometimes a number, presumably to indicate the number of enzyme subunits in a complex. I couldn't find out where they sourced this information, so I opted not to use it. It would probably be really important to use for future work if it's possible to get that information.



### Jul 2, 2025:
Today I had a meeting with some colleagues to pitch the ideas I wrote from Jun 25, 2025. 

For idea number 1, where we use Boltz-2 co-structure generation and SIMG* graphs including orbitals to try and bolster the predictions made by embeddings + XGBoost models and embeddings + attention based models, I was told that this may be adding inductive bias but is the bias going to be informative if we haven't thoroughly identified weakenesses of these current models. From a narrative standpoint, why would we do this work if we haven't seen that these could make any meaningful improvements. Sure we may be providing the model with more information about what it might be working with, but does it need this to make better predictions or has it already been able to infer this?

For idea number 2, where we use multi-task learning with kcat and k-1 and k+1 to predict kinetic parameters, my colleagues were skeptical about whether the relationships held true for Ki and KM and whether the masking would be effective to get models to bridge between kcat and Ki. In most enzyme kinetics courses they teach you about these relationships, but perhaps the field has gotten more complex in this regard and these relationships are more complex. If that is the case, it still might not be impossible to say that framing the predictions like this would lead to improved predictions because this would be the first instance where people are mixing the prediction tasks where the weights could learn from each other. In <a href=https://doi.org/10.48550/arXiv.1705.07115>this paper</a> they find that multi-task learning for related concepts lead to better predictions on individual tasks. The narrative in this case could be to apply this to see if it works for predicting enzyme kinetic parameters. This paper was published on arXiv in 2017, so there might be more development in this field related to multi-task learning.



### Jul 3, 2025:
Today, I met with a colleague and we spoke about datasets.

I was talking to them about processing SABIO-RK and how I was processing it. They said they had previously run into problems where they weren't able to decipher which compound was the limiting reagent for a given reaction and I showed them that what I collected seemed to be able to do that. I was also speaking to them about how I thought there wasn't enough reproducible code to acquire the SABIO-RK data out there so I was making the effort to build it up myself. My PI had said to me before to just take the datasets we have and use them for getting results, but I felt like it was a bit hasty since I felt like I wanted to make my work as reproducible as possible. My colleague told me that they agreed with my PI and that I should probably just take some existing data and maybe try to expand it a bit later on. 

I've copied over the CPI-Pred "core" (which I'm coming to understand is just BRENDA's usable data points) and "scrn" (which is the pangenomic expansion of the BRENDA dataset to make use of organism names and EC numbers to identify uniprot IDs and compound SMILES). I'll be combining these datasets to have a signle table containing `protein_sequences, ligand_smiles, k_cat, K_M, K_i` for each the BRENDA and pangenomic BRENDA datasets with names that more accurately reflect this.

### Jul 4, 2025
Today I'm continuing to create the combined datasets for CPI-Pred.

I went on BRENDA to look for the units used for kcat, KM, and Ki.
- kcat units: s^-1
- KM units: mM
- Ki units: mM

I've been reading up on how kcat, KM, and kcat/KM are recorded because yesterday my senior colleague told me they believed that kcat, KM, and kcat/KM are from different assays but I felt they were from the same assay. My search showed that these values come from Michaelis-Menten kinetics experiments where they measure substrate concentration (M) vs rate of reaction (s^-1). Some main parameters that come from this experiment are Vmax (the rate at which the enzyme is fully saturated with substrate) and KM (the substrate concentration at which the enzyme is experiencing half of Vmax). kcat is a derivative of Vmax and is acquired by $`k_{cat} = \frac{V_{max}}{[E]}`$. Expanding this logic gives us kcat/KM simply by computing the ratio between kcat and KM. I did manage to find that there is a direct assay to determine kcat/KM but it is typically used to predict catalytic efficiency at low substrate concentrations. I did find that <a href=https://doi.org/10.1016/j.tibtech.2007.03.010>this one paper from 2007</a> said using kcat/KM is not a good metric for comparing enzymes since "*higher catalytic efficiency (i.e. kcat/KM value) can, at certain substrate concentrations, actually catalyze an identical reaction at lower rates than one having a lower catalytic efficiency*".

I did manage to investigate the data and found 1076 data points that had kcat/KM, but were missing one of either kcat or KM, so these values could be supplemented. There were 845 data points that had kcat/KM, but were missing both kcat and KM, so these might not be able to be recovered without some kind of imputation. Lastly, there were 4611 data points that had both kcat and KM, but were missing kcat/KM. This wouldn't change much for us as we are mainly predicting kcat and KM, and kcat/KM is a compound metric. I will fill in the blanks for these 5687 (1076 + 4611) data points to have a more complete dataset.

I've created `scripts/download_data/combine_cpipred_data.py` to successfully combine all the kinetic parameters (kcat, KM, kcat/KM, Ki) into a single file with enriched kcat, KM, and kcat/KM. I'm not sure if this is the right place for the script since there is data processing in `src/astra/data_processing`, but I thought that was reserved for `DataLoaders()`. Now that I have something to work with, I can try creating splitting scripts and work on downloading SABIO-RK, BRENDA, and BindingDB in the background. 

### Jul 11, 2025
This past week, I presented the idea from Jun 25, 2025 to my PI and he seemed enthusiastic about the idea of having informing the neural network with kinetic rates instead of having them be individual models, one for kcat/one for km/one for ki. He told me he wanted to expand upon the rates I had written since that model assumes E+S forming ES is reversible and ES generating E+P is irreversible. He told me to read through the kinetic chapters in Systems Biology 2nd Ed. by Edda Klipp and Enzyme Kinetics: Behavior and Analysis of Rapid Equilibrium and Steady-State Enzyme Systems 1st Ed. by Irwin Segel to improve the rate equations I had. I've also been trying to get CatPred predictions to work, but it seems like the CatPred repository wasn't designed to be reproduced with external data (at the time of writing) since they don't have documentation to reproduce their training but with splits created by the user. I found bugs in uploading the model in `utils.py` and I have since fixed that bug. But I continue to run into issues. After reading through all the potential arguments the script can take in, the ablation study from the CatPred paper shows that adding ESM features lead to the best model, but when it tries to validate on the model we train, it seems to run into some concatenation error. The student who designed CatPred has since graduted and we are struggling to get the answers to run effective and reproducible training on their architecture, which is unfortunate. I wanted to try offering to make the documentation ourselves as part of this learning process but I was told by my colleague not to edit any of their code since our changes might not be what the original authors want and could cause issues if we publish on their work that we edited. It could be considered sabotage or something like that, so I guess I'm not gonna get involved, even though it's a shame to not be able to reproduce the leading model in this area...

I just spoke to one of my senior colleagues about building up the model for this and they suggested that it is easiest to build from your own architecture instead of trying to build on someone else's model architecture since you would need to get an in-depth understanding of their model, which isn't always worth it in this case.

They also recommended just comparing to the models they will provide and not trying to retrofit their code to my task. I think this is a good approach and honestly what I would've done if I was left to my own devices. 

Since I have the dummy data from CPI-Pred, that I can always expand later on, I should start with that to build very simple models for proof of concept and build up from there. I will still need to try strengthening the kinetic mechanism model that I want to retrofit to if this simple assumption isn't enough.

### Jul 13, 2025
Today I've been thinking more about my reading committee meeting and how inductive biases are needed to make good use of a GNN, especially for the case where it is fully connected.

I've heard that Transformers are GNNs (Graph Neural Networks) a fair bit lately, and that's made me wonder exactly how that's true and if it means GNNs are obsolete as a result. As it so happens, Transformers represent a specific kind of GNN. If a GNN uses graphs with no edge embeddings and is fully connected, it is effectively a Transformer. Albeit, Transformers don't have as strong of an inductive bias for structure as GNNs do since Transformers require positional encodings to more loosely infer the structure of the data. In my mind, it seems more like the relationship between a square and a rectangle. Squares are a single special case rectangle where all the sides are even. The square is like the Transformer in that it is a special case GNN where all the nodes are fully connected. What's interesting is that some sources have said that, because the Transformer needs to learn inductive biases instead having concrete inductive biases in a GNN, Transformers require much more data to learn the structural patterns that a GNN gets for free.

### Jul 14, 2025
Today I spoke to my PI about the upgraded kinetic derivations and spoke to my senior colleagues about the data included in the dataset we have on hand. In each data point, only a single substrate is defined and these are the substrates provided by BRENDA. BRENDA doesn't include the reactions by default and kinetic parameters typically isolate for a single limiting reagent. 

```
                k₁                k₂                k₃
     E + S  <-------->   ES   <-------->   EP   <-------->   E + P
                k₋₁               k₋₂               k₋₃
      ^                  ^
      |                  |
  +I  | k₄, k₋₄      +I  | k₅, k₋₅
      |                  |
      v                  v

      EI                ESI
```

- $`k_{cat}=\frac{k_{+2} k_{+3}}{k_{-2}+k_{+3}+k_{+2}}`$
- $`K_M=\frac{k_{-1}k_{-2}+k_{-1}k_{+3}+k_{+2}k_{+3}}{k_{+1}(k_{-2}+k_{+3}+k_{+2})}`$
- $`K_{i,comp}= \frac{k_{-4}}{k_{+4}}`$
- $`K_{i,uncomp}= \frac{k_{-5}}{k_{+5}}`$
- $`K_D=\frac{k_{off}}{k_{on}}`$

*In vitro* assays for determining kinetic parameters will always start with [P] = 0 and so the equilibrium always favours the products, making the reactions effectively irreversible even though most enzymatic reactions are reversible. As a result, $`k_{-3}`$ is ignored in the elementary decompositions of these kinetic parameters. 

The irreversible form of the Michaelis-Menten kinetic formula is as follows:

$`v=\frac{V_{max}[S]}{K_M + [S]}`$

The reversible form of the Michaelis-Menten kinetic formula is as follows:

$`v=\frac{S(\frac{V_{max,f}}{K_{M,S}})-P(\frac{V_{max,r}}{K_{M,P}})}{1+\frac{S}{K_{M,S}}+\frac{P}{K_{M,P}}}`$

The reversible case can be constructed from the irreversible case if we consider both the forward and reverse reactions as two 'irreversible' reactions. From *in vitro* kinetic assays, this would require using two different sets of limiting reagents to assay the conversion to substrate and the conversion to product. 

My PI approves of these derivations and thinks I should push forward with building this project out. Yippy!

### Jul 15, 2025
Today, I'm learning about how to use wandb for logging model progress and checkpointing. I'm also learning about how to post a model on huggingface for people to be able to download the model down the line once the work is published. 

With wandb, I needed to install it using `uv add wandb`. Then, I added my key as a command line variable `export WANDB_API_KEY=<KEY>`. I was speaking with Gemini 2.5 Pro to learn about how I can make wandb logging optional and it gave me this boilerplate so that people won't be forced to use wandb once it's published.
```
# train.py
import argparse
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

def main(args):
    # --- Logger Setup ---
    if args.logger == "wandb":
        # Check if user is logged in, or prompt them.
        # This will error out if the key isn't set, which is good.
        import wandb
        wandb.login()
        logger = WandbLogger(project="my-awesome-project", log_model="all")
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger("tb_logs", name="my-model")
    else:
        # No logger
        logger = False

    # --- Trainer ---
    trainer = pl.Trainer(
        ...,
        logger=logger  # Pass the configured logger here
    )
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add other arguments for hyperparameters etc.
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard", # A safe, local default
        choices=["wandb", "tensorboard", "none"],
        help="Logger to use for experiment tracking."
    )
    args = parser.parse_args()
    main(args)
```
Having a single project name `WandbLogger(project="my-project")` will log multiple experiments to the same project name. This way, I won't get confused about where each training experiment goes and be able to find the best ones even with different architectures.

In huggingface, the models are stored in a separate repo and it is typically best to upload them as `.bin` files instead of `.ckpt` files because `.ckpt` is Pytorch Lightning specific. It seems like I don't have to worry about this right now though. I can always come back to uploading the model to huggingface later on. For now, I should focus on building the models.

### Jul 16, 2025
Today I worked on building up the `DataModule` class by reading through the <a herf=https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup>lightning docs</a> and the <a href=https://github.com/jwohlwend/boltz>boltz repository</a> to get an idea of how I could cleanly handle the data processing logic.

Reading through the boltz repository (at the time of writing), it seems like the boltz group created a custom class for their training config that parses their raw training YAML file into a useable format inside the `BoltzTrainingDataModule` class and use this class inside their `train()` function. The `train()` function takes some of the config arguments to setup hyperparameters, loading pretrained weights, checkpointing with a `Trainer()` callback (i.e. `ModelCheckpoint()`), and `WandbLogger()` for logging experimental progress.

I was able to build a functional first iteration of `AstraDataModule` that takes the `full_path` to a csv file. For now, it just sets the `train_dataloader` and `valid_dataloader` to the data in `full_path`. It will create the `ProteinLigandDataset` in the `setup()` method with tokenized proteins and ligand data points and `train_dataloader()` and `valid_dataloader()` methods wrap around this. Eventually, this class will need some logic for the use to make their own splits or with split functions that I'll create, that a user can create, or they can use their own train, valid, and test sets already presplit. 

Here's an output of what I have so far for the `train_dataloader`:
```
{'protein_input_ids': tensor([[ 2, 14, 13,  ...,  0,  0,  0],
         [ 2, 14,  6,  ...,  0,  0,  0],
         [ 2, 14,  9,  ...,  0,  0,  0],
         ...,
         [ 2, 14,  4,  ...,  0,  0,  0],
         [ 2, 14, 17,  ...,  0,  0,  0],
         [ 2, 22, 22,  ...,  0,  0,  0]]),
 'protein_attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         ...,
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0]]),
 'ligand_input_ids': tensor([[ 2, 43, 35,  ...,  0,  0,  0],
         [ 2, 38, 38,  ...,  0,  0,  0],
         [ 2, 68,  3,  ...,  0,  0,  0],
         ...,
         [ 2, 42, 38,  ...,  0,  0,  0],
         [ 2, 43, 35,  ...,  0,  0,  0],
         [ 2, 38, 35,  ...,  0,  0,  0]]),
 'ligand_attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         ...,
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0]]),
 'targets': tensor([[5.9000e+00, 1.9000e+01,        nan],
         [2.6833e-01, 2.9333e-03,        nan],
         [       nan,        nan, 1.7000e-01],
         [       nan, 6.7000e-01,        nan],
         [       nan, 6.3074e-01,        nan],
         [3.1500e-02, 1.5000e-02,        nan],
         [2.8000e-02, 2.0000e-04,        nan],
         [2.3000e-01, 1.2000e-01,        nan],
         [1.5000e-02, 4.1000e-02,        nan],
         [       nan,        nan, 8.0000e-02],
         [1.3000e-02, 7.9300e-01,        nan],
         [4.2500e+00, 5.0000e+00,        nan],
         [2.5120e-01, 2.7636e-03, 2.3504e-02],
         [7.0000e-01, 5.0000e+01,        nan],
         [1.0800e+03, 8.1000e-03,        nan],
         [1.2700e+00, 1.4500e-02,        nan],
         [       nan, 2.0000e-02,        nan],
         [       nan, 2.8000e-02,        nan],
         [4.9000e-01, 1.5987e+00,        nan],
         [2.1000e+00, 4.0100e+01,        nan],
         [9.5300e+00, 5.3300e+01,        nan],
         [       nan, 1.9875e-02,        nan],
         [       nan, 8.3500e-01,        nan],
         [       nan, 5.6700e-02,        nan],
         [4.0000e-03, 2.7200e-01,        nan],
         [       nan, 7.8000e-04,        nan],
         [       nan, 3.2450e-01,        nan],
         [1.3000e+00, 4.4100e-02,        nan],
         [       nan, 2.5000e+01,        nan],
         [       nan,        nan, 4.2000e-02],
         [3.1000e+01, 7.3000e-03,        nan],
         [       nan, 4.7000e-04,        nan]])}
```