### Jun 25, 2025:
Today, I've been conceptualizing how research papers could look like with different model architectures and how we could do multi-task learning for predicting kinetic parameters. 

The model architecture would likely be compared to existing models in the space that use embeddings + XGBoost (UniKP/TuRNuP) or things like embeddings + attention (CatPred/CPI-Pred) to using 3D structures as tensors (method for this is undecided) or as graphs (with 3D structured embedded in them), the same could be done for co-structure generation from Boltz-2 (tensor/graph) and then last would be to using SIMG* (stereoelectronics-infused molecular graphs) graphs (source: https://www.nature.com/articles/s42256-025-01031-9) since they seem very robust. I wonder if we do need tensors for all this of if we can just go straight to graphs.

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

Where $`\alpha, \beta, \gamma, \delta, \epsilon`$ are weights for the individual loss functions and $`m_{x} \in \Set {0,1}`$ is the mask for whichever prediction is being used since not every entry will have kcat, Ki, and/or KM associated with it. There's a nice paper outlining how to do this: https://arxiv.org/abs/1705.07115. The two types of loss functions are $`MSE()`$ for regression tasks and $`BCE()`$ for binary classification tasks.