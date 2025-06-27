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