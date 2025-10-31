import torch
import torch.nn as nn


# Orignial class name from CPI-Pred's Z05X_All_Models.py: SQembConv_CPenc_Model_v2
# TODO: Include github link once published

# cpi_pred_conv_model.py
# Place this file in the same directory as your DummyModel, e.g., astra/model/architectures/

import torch
import torch.nn as nn

class CpiPredConvModel(nn.Module):
    """
    An adaptation of the CPI-Pred model (SQembConv_CPenc_Model_v2) for the Astra pipeline.

    This model uses 1D convolutions over the protein sequence embedding and concatenates
    the result with a ligand fingerprint embedding before a final feed-forward network.
    """
    def __init__(self,
                 # --- Dimensions injected by PipelineBuilder ---
                 protein_emb_dim: dict,
                 ligand_emb_dim: dict,
                 out_dim: int,
                 # --- Hyperparameters from config ---
                 hid_dim: int,
                 kernal_1: int,
                 conv_out_dim: int,
                 kernal_2: int,
                 last_hid: int,
                 dropout: float = 0.1,
                 ):
        super().__init__()

        # --- Derive internal dimensions from pipeline specs ---
        # protein_emb_dim is {'embedding': (max_len, embedding_dim), 'attention_mask': ...}
        # ligand_emb_dim is {'embedding': (fp_size,)}
        in_dim_internal   = protein_emb_dim['embedding'][1]
        max_len_internal  = protein_emb_dim['embedding'][0]
        cmpd_dim_internal = ligand_emb_dim['embedding'][0]

        # --- Model Layers ---
        self.norm       = nn.BatchNorm1d(in_dim_internal)
        self.conv1      = nn.Conv1d(in_dim_internal, hid_dim, kernal_1, padding=int((kernal_1 - 1) / 2))
        self.dropout1   = nn.Dropout(dropout)
        
        # Branch 1
        self.conv2_1    = nn.Conv1d(hid_dim, conv_out_dim, kernal_2, padding=int((kernal_2 - 1) / 2))
        self.dropout2_1 = nn.Dropout(dropout)
        
        # Branch 2 (Residual Connection)
        self.conv2_2    = nn.Conv1d(hid_dim, hid_dim, kernal_2, padding=int((kernal_2 - 1) / 2))
        self.dropout2_2 = nn.Dropout(dropout)
        
        # Ligand Encoder
        self.cmpd_enc   = nn.Linear(cmpd_dim_internal, in_dim_internal)
        self.cmpd_drop  = nn.Dropout(dropout)

        # Early fusion branch (not used for final prediction in original code, we can ignore its output)
        self.fc_early   = nn.Linear(max_len_internal * hid_dim + in_dim_internal, 1)
        self.cls        = nn.Sigmoid()
        
        # Third convolution on the residual branch
        self.conv3      = nn.Conv1d(hid_dim, conv_out_dim, kernal_2, padding=int((kernal_2 - 1) / 2))
        self.dropout3   = nn.Dropout(dropout)
        
        # Final Feed-Forward Network
        # Input: flattened_conv_outputs + encoded_ligand
        ffn_in_dim = int(2 * max_len_internal * conv_out_dim + in_dim_internal)
        self.fc_1 = nn.Linear(ffn_in_dim, last_hid)
        self.fc_2 = nn.Linear(last_hid, last_hid)
        self.fc_3 = nn.Linear(last_hid, out_dim) # Use the pipeline-provided out_dim


    def forward(self, protein_embedding: torch.Tensor, ligand_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        # The pipeline provides protein_embedding with shape (batch, seq_len, embed_dim)
        # The Conv1D layer expects (batch, embed_dim, seq_len), so we transpose.
        output = protein_embedding.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout1(output)
        
        # Branch 1
        output_1 = nn.functional.relu(self.conv2_1(output))
        output_1 = self.dropout2_1(output_1)
        
        # Branch 2 with residual connection
        output_2 = nn.functional.relu(self.conv2_2(output)) + output
        output_2 = self.dropout2_2(output_2)
        
        # Encode ligand to match protein embedding space
        compound_encoded = self.cmpd_drop(nn.functional.relu(self.cmpd_enc(ligand_embedding)))

        # Optional early fusion (we compute but don't use the result for the final output)
        # single_conv_input = torch.cat((torch.flatten(output_2, 1), compound_encoded), 1)
        # single_conv_output = self.cls(self.fc_early(single_conv_input))
        
        # Continue with Branch 2
        output_2 = nn.functional.relu(self.conv3(output_2))
        output_2 = self.dropout3(output_2)
        
        # Concatenate the two convolutional branches
        output = torch.cat((output_1, output_2), 1)
        
        # Flatten and concatenate with the encoded ligand for final prediction
        output = torch.cat((torch.flatten(output, 1), compound_encoded), 1)
        
        # Final FFN
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        output = nn.functional.relu(output)
        output = self.fc_3(output)

        return output