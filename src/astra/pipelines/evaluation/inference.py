import pandas as pd
import torch
import lightning as L
from tqdm import tqdm
import umap # pip install umap-learn

class FeatureExtractor:
    def __init__(self, model, layer_name):
        self.features = None
        # Recursively find the layer by name and register the hook
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self.hook_fn)
                break

    def hook_fn(self, module, input, output):
        # Depending on the layer, output might be a tuple. Grab the tensor.
        self.features = output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else output[0].detach().cpu().numpy()

def extract_oof_data_for_fold(ckpt_path, val_dataloader, model_layer_to_hook="model.linear.2"):
    """
    Loads a checkpoint, runs validation data, and returns a dataframe.
    `model_layer_to_hook` should be the name of the layer right before your final output.
    For LinearBaselineModel, "model.linear.2" (the 2nd linear layer) is a good choice.
    """
    # Load model from checkpoint
    module = AstraModule.load_from_checkpoint(ckpt_path)
    module.eval()
    module.cuda() # Assuming GPU

    # Set up the hook to grab the penultimate embeddings
    extractor = FeatureExtractor(module, model_layer_to_hook)

    all_data = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Processing fold"):
            # Move batch to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Pop targets (assuming shape: batch_size x 3 for kcat, KM, Ki)
            targets = batch.pop("targets").cpu().numpy()
            
            # 1. Forward pass (gets log k1, log k-1, log k2)
            rates = module.model(**batch)
            
            # 2. Recomposition (gets log kcat, log KM, log Ki)
            preds = module.recomposition_func(rates) if module.recomposition_func else rates
            
            # 3. Grab the embeddings intercepted by our hook
            embeddings = extractor.features
            
            # 4. Move to CPU for pandas
            rates_np = rates.cpu().numpy()
            preds_np = preds.cpu().numpy()
            
            # Construct a row for each item in the batch
            for i in range(len(targets)):
                all_data.append({
                    "true_kcat": targets[i, 0],
                    "true_KM": targets[i, 1],
                    "true_Ki": targets[i, 2],
                    "pred_kcat": preds_np[i, 0],
                    "pred_KM": preds_np[i, 1],
                    "pred_Ki": preds_np[i, 2],
                    "log_k1": rates_np[i, 0],
                    "log_k_minus_1": rates_np[i, 1],
                    "log_k2": rates_np[i, 2],
                    "embedding": embeddings[i] # Array of size (embedding_dim,)
                })
                
    return pd.DataFrame(all_data)